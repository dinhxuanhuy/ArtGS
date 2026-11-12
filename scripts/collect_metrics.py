#!/usr/bin/env python3
"""Turn a run record's raw outputs into one metrics.json.

    python scripts/collect_metrics.py runs/<RUN_ID>

Collects every render.py result.csv found under the record, at
.../<dataset>/<subset>/<scene>/<model>/<split>/ours_<iteration>/result.csv
(<subset> may be absent, e.g. official checkpoints <dataset>/<scene>/...), plus
stage timings and the train.py summary block (val_axis_ang, peak_vram_mb).

Metric meaning (render.py): angle_* = axis angle error (deg), distance_* = axis position
error, theta_diff_* = part motion error, CD_* = Chamfer distance (static / dynamic part /
whole), PSNR/SSIM/LPIPS = -1 when appearance eval was skipped. Lower is better except PSNR/SSIM.
"""
import csv, json, re, sys
from pathlib import Path


def read_result_csv(p: Path) -> dict:
    out = {}
    with open(p) as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0]:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


def parse_path(p: Path, root: Path) -> dict:
    parts = p.relative_to(root).parts[:-1]          # drop result.csv
    it = parts[-1].removeprefix("ours_")
    split, model = parts[-2], parts[-3]
    rest = parts[:-3]
    known = {"paris", "dta", "artgs"}
    idx = next((i for i, x in enumerate(rest) if x in known), None)
    if idx is None:
        return {"split": split, "model": model, "iteration": it, "path": str(p.relative_to(root))}
    ds, tail = rest[idx], rest[idx + 1:]
    subset, scene = (tail[0], tail[1]) if len(tail) >= 2 else (None, tail[0] if tail else model)
    if len(tail) == 0:  # official ckpt layout <dataset>/<scene>/<split>/...
        scene, model = model, "official"
    return {"dataset": ds, "subset": subset, "scene": scene, "model": model, "split": split,
            "iteration": it, "path": str(p.relative_to(root))}


def main(rec: Path) -> dict:
    results = []
    for p in sorted(rec.rglob("result.csv")):
        results.append({**parse_path(p, rec), "metrics": read_result_csv(p)})
    stages = {}
    for t in rec.rglob("stage_seconds.tsv"):
        for line in t.read_text().splitlines():
            k, _, v = line.partition("\t")
            if v.strip().isdigit():
                stages[k] = int(v)
    summary = {}
    log = rec / "job.log"
    if log.exists():
        txt = log.read_text(errors="replace")
        for k in ("val_axis_ang", "val_l1", "peak_vram_mb"):
            m = re.findall(rf"^{k}: ([-\d.eE]+)", txt, re.M)
            if m:
                summary[k] = [float(x) for x in m]
    exit_code = (rec / "exit_code").read_text().strip() if (rec / "exit_code").exists() else None
    out = {"run_id": rec.name, "exit_code": exit_code, "stage_seconds": stages,
           "train_summary": summary, "results": results}
    (rec / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    o = main(Path(sys.argv[1]))
    print(f"{o['run_id']}: exit={o['exit_code']} results={len(o['results'])} stages={o['stage_seconds']}")
