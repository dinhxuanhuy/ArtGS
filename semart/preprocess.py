"""SemArt S0: VLM part list + per-view instance masks for one scene.

Runs in a modern Python/torch env (on Colab: the system python, NOT the ArtGS 3.10 venv):
    python -m semart.preprocess --scene_dir /content/data/dta/sapien/storage_47254 \
        --object_id dta/sapien/storage_47254 [--vlm Qwen/Qwen3-VL-4B-Instruct] [--segmenter sam3|qwen_sam2]

Output (content-addressed, tools/cache.py): cache/semseg/<object_id>/<hash>/
    vlm.json        raw VLM answer + parsed part list {object, movable_parts:[{name,count,joint}]}
    masks/<state>/<frame>.npz   labels uint8 HxW (0 none, i = instance i), cats [n], scores [n]
    summary.json    per-category instance counts per view, timings
    vis/            a few overlays for eyeballing
Prints the entry path on the last line ("SEMSEG_DIR=<path>").
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.cache import CacheEntry  # noqa: E402

STATES = ("start", "end")
PROMPT_VERSION = "v2"
PART_PROMPT = (
    "The images show ONE object from several viewpoints. The first {n} images show it in "
    "articulation state A, the last {n} images in state B. Some parts moved between the two states.\n"
    "List the object's movable parts: rigid parts that move relative to the main body (doors, drawers, "
    "lids, ...). Do not list the main body. Do not list small parts rigidly attached to a movable part "
    "(handles, knobs): they belong to that part. Group identical parts under one name with a count.\n"
    "joint: \"revolute\" if the part rotates about a hinge (doors, lids, flaps, even if they fold down), "
    "\"prismatic\" if it slides along a straight line (drawers, sliding trays). Compare the two states.\n"
    "Answer with JSON only, no prose:\n"
    '{{"object": "<category>", "movable_parts": [{{"name": "<short noun usable as a segmentation '
    'prompt, e.g. drawer>", "count": <int>, "joint": "revolute" or "prismatic"}}]}}'
)
GROUND_PROMPT = ('Locate every instance of these parts of the object: {names}. Output JSON only, one entry '
                 'per instance: [{{"bbox_2d": [x1, y1, x2, y2], "label": "<one of: {names}>"}}]. '
                 'Output [] if there is none.')


def log(*a):
    print("[preprocess]", *a, flush=True)


def list_views(scene_dir, state):
    d = os.path.join(scene_dir, state, "train", "rgba")
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def load_rgb(path, bg=255):
    im = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    a = im[..., 3:4] / 255.0
    rgb = im[..., :3] * a + bg * (1 - a)
    return Image.fromarray(rgb.astype(np.uint8)), (a[..., 0] > 0.5)


def fit(image, size, fg=None, margin=0.08):
    """Optionally crop to the foreground bbox, then resize so the longer side is `size`."""
    if fg is not None and fg.any():
        ys, xs = np.nonzero(fg)
        h, w = fg.shape
        my, mx = int(margin * h), int(margin * w)
        image = image.crop((max(xs.min() - mx, 0), max(ys.min() - my, 0),
                            min(xs.max() + mx, w), min(ys.max() + my, h)))
    s = size / max(image.size)
    return image.resize((max(1, round(image.size[0] * s)), max(1, round(image.size[1] * s))), Image.BICUBIC)


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in VLM answer: {text[:300]!r}")


def dtype_for_gpu():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


# ---------------------------------------------------------------- VLM
class VLM:
    def __init__(self, name):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype_for_gpu(), device_map="cuda").eval()

    @torch.no_grad()
    def ask(self, images, text, max_new_tokens=512):
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def normalise_parts(parsed):
    parts = []
    for p in parsed.get("movable_parts", []):
        name = str(p.get("name", "")).strip().lower()
        if not name:
            continue
        joint = str(p.get("joint", "revolute")).lower()
        parts.append({"name": name, "count": max(1, int(p.get("count", 1))),
                      "joint": "prismatic" if joint.startswith("pris") else "revolute"})
    merged = {}
    for p in parts:                                   # same name twice -> add counts
        if p["name"] in merged:
            merged[p["name"]]["count"] += p["count"]
        else:
            merged[p["name"]] = p
    return {"object": parsed.get("object", ""), "movable_parts": list(merged.values())}


# ---------------------------------------------------------------- segmenters
class Sam3Segmenter:
    def __init__(self, name="facebook/sam3", **_):
        from transformers import Sam3Model, Sam3Processor
        self.processor = Sam3Processor.from_pretrained(name)
        # fp16 + one vision pass shared by all prompts: 6.6x faster than fp32 per-prompt on a T4
        self.model = Sam3Model.from_pretrained(name).to("cuda").half().eval()

    @torch.no_grad()
    def segment_all(self, image, names, score_thresh):
        img = self.processor(images=image, return_tensors="pt").to("cuda")
        ve = self.model.get_vision_features(pixel_values=img["pixel_values"].half())
        out = []
        for n in names:
            ti = self.processor(text=n, return_tensors="pt").to("cuda")
            o = self.model(vision_embeds=ve, **ti)
            res = self.processor.post_process_instance_segmentation(
                o, threshold=score_thresh, mask_threshold=0.5, target_sizes=img["original_sizes"].tolist())[0]
            out += [(n, m, float(sc)) for m, sc in zip(res["masks"].bool().cpu().numpy(),
                                                       res["scores"].float().cpu().numpy())]
        return out


class QwenSam2Segmenter:
    """Qwen3-VL grounding boxes (0..1000 relative coords) -> SAM2.1 box prompts."""

    def __init__(self, vlm, name="facebook/sam2.1-hiera-large", ground_size=512, **_):
        self.ground_size = ground_size
        from transformers import Sam2Model, Sam2Processor
        self.vlm = vlm
        self.processor = Sam2Processor.from_pretrained(name)
        self.model = Sam2Model.from_pretrained(name).to("cuda").eval()
        self.raw = []

    @torch.no_grad()
    def segment_all(self, image, names, score_thresh):
        W, H = image.size
        ans = self.vlm.ask([fit(image, self.ground_size)], GROUND_PROMPT.format(names=", ".join(names)),
                           max_new_tokens=64 + 48 * 8)
        self.raw.append(ans)
        try:
            dets = extract_json(ans)
        except ValueError:
            return []
        if isinstance(dets, dict):
            dets = [dets]
        boxes, labels = [], []
        for d in dets:
            b = d.get("bbox_2d") if isinstance(d, dict) else None
            lab = str(d.get("label", "")).strip().lower() if isinstance(d, dict) else ""
            lab = lab if lab in names else (names[0] if len(names) == 1 else None)
            if b and len(b) == 4 and lab:
                x1, y1, x2, y2 = [float(v) for v in b]
                boxes.append([x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H])
                labels.append(lab)
        if not boxes:
            return []
        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt").to("cuda")
        out = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        masks = masks[:, 0].numpy().astype(bool)
        scores = out.iou_scores.cpu().float().numpy().reshape(len(boxes), -1)[:, 0]
        return [(l, m, float(s)) for l, m, s in zip(labels, masks, scores) if s >= score_thresh]


# ---------------------------------------------------------------- label maps
def build_label_map(dets, fg, iou_dup=0.7, min_area=64):
    """dets: list of (cat, mask HxW bool, score). Higher score wins overlaps; duplicates dropped."""
    dets = sorted(dets, key=lambda d: -d[2])
    kept = []
    for cat, m, sc in dets:
        m = m & fg
        if m.sum() < min_area:
            continue
        if any((m & k[1]).sum() / max((m | k[1]).sum(), 1) > iou_dup for k in kept):
            continue
        kept.append((cat, m, sc))
    kept = kept[::-1][:254]                           # paint low -> high score: high score wins
    lab = np.zeros(fg.shape, np.uint8)
    for i, (_, m, _) in enumerate(kept):
        lab[m] = i + 1
    out = np.zeros_like(lab)                          # compact ids, drop instances painted over
    cats, scores = [], []
    for i, (cat, _, sc) in enumerate(kept):
        m = lab == i + 1
        if m.sum() >= min_area:
            cats.append(cat)
            scores.append(sc)
            out[m] = len(cats)
    return out, cats, scores


def overlay(image, lab):
    rng = np.random.default_rng(0)
    pal = (rng.random((256, 3)) * 200 + 55).astype(np.uint8)
    im = np.array(image).copy()
    m = lab > 0
    im[m] = (0.45 * im[m] + 0.55 * pal[lab[m]]).astype(np.uint8)
    return Image.fromarray(im)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--object_id", required=True, help="<dataset>/<subset>/<scene>")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--segmenter", default="sam3", choices=["sam3", "qwen_sam2"])
    ap.add_argument("--sam3", default="facebook/sam3")
    ap.add_argument("--sam2", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--vlm_views", type=int, default=4, help="views per state shown to the VLM")
    ap.add_argument("--vlm_size", type=int, default=448, help="longer side of each VLM sheet image")
    ap.add_argument("--view_stride", type=int, default=1, help="segment every n-th train view")
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--parts_json", default="", help="skip the VLM: fixed part list (oracle ablation)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    config = {"code": f"semart-preprocess-{PROMPT_VERSION}", "vlm": a.vlm, "segmenter": a.segmenter,
              "seg_model": a.sam3 if a.segmenter == "sam3" else a.sam2, "seg_precision": "fp16" if a.segmenter == "sam3" else "fp32", "vlm_views": a.vlm_views,
              "view_stride": a.view_stride, "vlm_size": a.vlm_size, "score_thresh": a.score_thresh,
              "part_prompt": PART_PROMPT, "ground_prompt": GROUND_PROMPT if a.segmenter == "qwen_sam2" else "",
              "parts_json": a.parts_json}
    e = CacheEntry("semseg", a.object_id, config)
    if e.complete and not a.force:
        log("cache hit", e.path)
        print(f"SEMSEG_DIR={e.path}")
        return
    out = e.prepare()
    t0 = time.time()
    timings = {}

    views = {s: list_views(a.scene_dir, s) for s in STATES}
    n = a.vlm_views
    # per state: real captures can have a different number of views in each state
    idx = {s: np.linspace(0, len(views[s]) - 1, n).round().astype(int) for s in STATES}
    sheet = [fit(im, a.vlm_size, fg) for im, fg in (load_rgb(views[s][i]) for s in STATES for i in idx[s])]

    vlm = None
    if a.parts_json:
        parsed = normalise_parts(json.loads(a.parts_json))
        raw = "(parts_json)"
    else:
        vlm = VLM(a.vlm)
        raw = vlm.ask(sheet, PART_PROMPT.format(n=n))
        parsed = normalise_parts(extract_json(raw))
    timings["vlm_s"] = round(time.time() - t0, 1)
    log("VLM:", json.dumps(parsed))
    json.dump({"raw": raw, "parsed": parsed, "sheet_views": {s: idx[s].tolist() for s in STATES}}, open(out / "vlm.json", "w"), indent=2)
    if not parsed["movable_parts"]:
        raise SystemExit("VLM returned no movable parts")

    t1 = time.time()
    if a.segmenter == "sam3":
        if vlm is not None:
            vlm.close()
        seg = Sam3Segmenter(a.sam3)
    else:
        if vlm is None:
            vlm = VLM(a.vlm)
        seg = QwenSam2Segmenter(vlm, a.sam2)

    counts = {p["name"]: [] for p in parsed["movable_parts"]}
    os.makedirs(out / "vis", exist_ok=True)
    n_done = 0
    for s in STATES:
        os.makedirs(out / "masks" / s, exist_ok=True)
        for vi, path in enumerate(views[s]):
            if vi % a.view_stride:
                continue
            image, fg = load_rgb(path)
            dets = seg.segment_all(image, [p["name"] for p in parsed["movable_parts"]], a.score_thresh)
            lab, cats, scores = build_label_map(dets, fg)
            for c in counts:
                counts[c].append(sum(1 for x in cats if x == c))
            name = os.path.splitext(os.path.basename(path))[0]
            np.savez_compressed(out / "masks" / s / f"{name}.npz", labels=lab,
                                cats=np.array(cats, dtype=object), scores=np.array(scores, np.float32))
            if vi in idx[s]:
                overlay(image, lab).save(out / "vis" / f"{s}_{name}.jpg", quality=85)
            n_done += 1
        log(f"{s}: {n_done} views segmented ({time.time() - t1:.0f}s)")
    timings["seg_s"] = round(time.time() - t1, 1)
    summary = {"parts": parsed["movable_parts"], "counts_per_view": counts, "n_views": n_done,
               "timings": timings, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if a.segmenter == "qwen_sam2":
        summary["grounding_raw_sample"] = seg.raw[:5]
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    e.mark_complete(n_views=n_done, parts=parsed["movable_parts"])
    log(f"done in {time.time() - t0:.0f}s")
    print(f"SEMSEG_DIR={e.path}")


if __name__ == "__main__":
    main()
