"""Table of semart_eval.json (+ lift reports) across methods / scenes / seeds of one or more runs.

    python3 semart/compare.py runs/<RUN_ID> [runs/<RUN_ID2> ...] [--md out.md]
Pure stdlib (runs locally without the ArtGS env).
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

COLS = [("K", "K_pred"), ("recall", "part_recall"), ("type", "type_acc"), ("ang", "mean_angle"),
        ("pos", "mean_distance"), ("state", "mean_theta_diff"), ("CDd", "mean_CD_dynamic"),
        ("CDs", "CD_static"), ("CDw", "CD_whole")]


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    rows = defaultdict(list)          # (scene, method) -> [eval dicts over seeds]
    lifts = {}
    for run in a.runs:
        for p in Path(run).rglob("semart_eval.json"):
            m = re.search(r"(\w+)/(\w+)/([\w.-]+)/(\w+?)_s(\d+)/train/ours_\d+/semart_eval.json$", str(p))
            if not m:
                continue
            ds, sub, scene, method, seed = m.groups()
            rows[(f"{ds}/{sub}/{scene}", method)].append(json.loads(p.read_text()))
        for p in Path(run).rglob("lift_*/lift_report.json"):
            r = json.loads(p.read_text())
            lifts[str(p.parent)] = r
    lines = ["| scene | method | n | " + " | ".join(c for c, _ in COLS) + " |",
             "|---" * (len(COLS) + 3) + "|"]
    for (scene, method), evs in sorted(rows.items()):
        cells = []
        for _, k in COLS:
            v = [e[k] for e in evs if e.get(k) is not None]
            if not v:
                cells.append("-")
            elif len(v) == 1:
                cells.append(fmt(v[0]))
            else:
                cells.append(f"{st.mean(v):.3g}±{st.stdev(v):.2g}")
        lines.append(f"| {scene} | {method} | {len(evs)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("ang = axis angle error (deg), pos = axis position error (x10), state = motion error, "
                 "CD x1000; means over matched GT parts; recall = matched GT parts / GT parts.")
    for k, r in sorted(lifts.items()):
        lines.append(f"- lift `{k}`: K={r.get('num_slots')} types={r.get('joint_types')} "
                     f"requested={r.get('k_requested')} eigengap={r.get('k_eigengap')} "
                     f"covered={r.get('covered_frac', 0):.2f} n_vlm={r.get('n_vlm')} n_sam={r.get('n_sam')}")
    text = "\n".join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
