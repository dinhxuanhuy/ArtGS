"""Benchmark report: SemArt (our runs, mean ± std over seeds) vs ArtGS / REArtGS / REArtGS++ (paper numbers).

    python3 semart/report_bench.py runs/<RUN> ... --method sem_v2_mask --out REPORT_BENCH.md
Pure stdlib. Paper numbers: semart/paper_baselines.json (sources and protocols listed there).
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ["ang", "pos", "motion", "CDs", "CDm", "CDw"]
HEAD = ["Ang", "Pos", "Motion", "CD-s", "CD-m", "CD-w"]
GROUPS = [("PARIS (tổng hợp)", "paris/sapien/"), ("PARIS (thật)", "paris/realscan/"),
          ("DTA-Multi", "dta/sapien/"), ("ArtGS-Multi", "artgs/sapien/")]
BASES = ["ArtGS (paper, depth)", "ArtGS (REArtGS++ tab, no depth)", "REArtGS (REArtGS++ tab, no depth)",
         "REArtGS++ (paper, no depth)"]


def ours(ev):
    parts = [p for p in ev["parts"] if "angle" in p]
    rev = [p for p in parts if p["gt_type"] == "r"]
    mean = lambda xs: st.mean(xs) if xs else None
    return {"ang": mean([p["angle"] for p in parts]), "pos": mean([p["distance"] for p in rev]),
            "motion": mean([p["theta_diff"] for p in parts if p["theta_diff"] == p["theta_diff"]]),
            "CDs": ev.get("CD_static"), "CDm": mean([p["CD_dynamic"] for p in parts if p.get("CD_dynamic") == p.get("CD_dynamic")]),
            "CDw": ev.get("CD_whole"), "K": ev["K_pred"], "Kgt": ev["K_gt"], "recall": ev["part_recall"], "type": ev["type_acc"]}


def f(v, n=3):
    return "-" if v is None else (f"{v:.{n}f}" if abs(v) < 100 else f"{v:.1f}")


def ms(vals, n=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f(st.mean(vals), n) + (f" ± {f(st.stdev(vals), n)}" if len(vals) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--method", default="sem_v2_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", default="", help="dir with <scene>.jpg figures (semart/postprocess_run.sh)")
    a = ap.parse_args()
    paper = json.loads((HERE / "paper_baselines.json").read_text())
    runs = defaultdict(list)                       # scene -> [ours dict per seed]
    for run in a.runs:
        for p in Path(run).rglob("semart_eval.json"):
            m = re.search(rf"(\w+/\w+/[\w.-]+)/{a.method}_s(\d+)/train/ours_\d+/semart_eval.json$", str(p))
            if m:
                runs[m.group(1)].append(ours(json.loads(p.read_text())))
    name = f"SemArt `{a.method}`"
    L = [f"# Benchmark: {name} vs ArtGS / REArtGS / REArtGS++", "",
         "**Ours:** 20k iteration, resolution 2, đánh giá ở iteration cuối, **không dùng GT** (không K GT, không "
         "chọn checkpoint bằng GT), có depth; trung bình ± độ lệch chuẩn trên các seed.",
         "**Paper:** ArtGS (paper, depth) = số trong paper ArtGS (có depth, res 1, chọn checkpoint theo GT, K GT). "
         "Ba dòng còn lại lấy từ bảng của REArtGS++ (CVPR'26), nơi **mọi method chạy không depth**; đơn vị Pos "
         "của REArtGS++ là **mm**, của ArtGS và của mình là ×10 đơn vị cảnh, nên **không so trực tiếp cột Pos** "
         "giữa hai nhóm.", "",
         "Chỉ số (thấp hơn là tốt hơn): Ang = sai số góc trục (°); Pos = sai số vị trí trục (chỉ khớp xoay); Motion = "
         "sai số trạng thái; CD ×1000 (s tĩnh, m part động, w toàn vật). Vật nhiều part: trung bình trên part động. "
         "\"không có\" = paper đó không báo cáo vật này; \"-\" = không áp dụng (vd. vị trí trục của khớp trượt).", ""]
    for gname, prefix in GROUPS:
        scenes = sorted(s for s in set(runs) | {k for k in paper if not k.startswith("_")} if s.startswith(prefix))
        if not scenes:
            continue
        L += [f"## {gname}", ""]
        for sc in scenes:
            L += [f"### {sc}", "", "| Nguồn | n | K | Loại khớp | " + " | ".join(HEAD) + " |", "|---" * 10 + "|"]
            for b in BASES:
                d = paper.get(sc, {}).get(b)
                if d:
                    L.append(f"| {b} | - | GT | - | " + " | ".join(f(d[k]) for k in KEYS) + " |")
                else:   # the paper does not report this object
                    L.append(f"| {b} | - | - | - | " + " | ".join("không có" for _ in KEYS) + " |")
            r = runs.get(sc, [])
            if r:
                K = f"{st.mean(x['K'] for x in r):.1f}/{r[0]['Kgt']}"
                L.append(f"| **{name}** | {len(r)} | {K} | {st.mean(x['type'] for x in r):.0%} | "
                         + " | ".join(ms([x[k] for x in r]) for k in KEYS) + " |")
            else:
                L.append(f"| **{name}** | 0 | - | - | " + " | ".join("chưa chạy" for _ in KEYS) + " |")
            fig = Path(a.figs) / (sc.replace("/", "_") + ".jpg") if a.figs else None
            if fig and fig.exists():
                L.append(f"![{sc}]({fig})")
            L.append("")
        # group mean over scenes that have every source
        L += [f"**Trung bình {gname}** (trên cùng tập vật SemArt đã chạy; nguồn không có số cho vật nào thì bỏ vật đó; "
              f"ours = trung bình seed rồi trung bình vật):", "",
              "| Nguồn | n vật | " + " | ".join(HEAD) + " |", "|---" * 8 + "|"]
        for b in BASES + [name]:
            vals = defaultdict(list)
            n = 0
            for sc in [x for x in scenes if runs.get(x)]:      # same object set for every source
                d = paper.get(sc, {}).get(b) if b != name else (
                    {k: st.mean([x[k] for x in runs[sc] if x[k] is not None]) if any(x[k] is not None for x in runs[sc]) else None
                     for k in KEYS} if runs.get(sc) else None)
                if not d:
                    continue
                n += 1
                for k in KEYS:
                    if d[k] is not None:
                        vals[k].append(d[k])
            if n:
                L.append(f"| {b} | {n} | " + " | ".join(f(st.mean(vals[k])) if vals[k] else "-" for k in KEYS) + " |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
