"""Markdown report: ArtGS paper numbers vs our re-run of ArtGS vs SemArt methods, per scene.

    python3 semart/report.py runs/<RUN1> runs/<RUN2> ... --out report.md [--figs dir]
Pure stdlib. Averages follow the ArtGS paper: Axis Pos over revolute parts only, other joint metrics
and CD-m over movable parts; values are for matched GT parts (unmatched parts are listed in 'recall').
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD_ORDER = ["artgs", "sem_k", "sem_vlm", "sem_mask", "sem_full", "gtk_mask", "sem_v2", "sem_v2_mask"]
DESC = {"artgs": "ArtGS chạy lại (K GT, spectral, loại khớp từ chuyển động)",
        "sem_k": "K VLM + tâm lift, loại khớp từ chuyển động",
        "sem_vlm": "K VLM + tâm lift, loại khớp VLM",
        "sem_mask": "sem_k + loss mask 2D",
        "sem_full": "sem_vlm + loss mask 2D",
        "gtk_mask": "K GT + loss mask 2D",
        "sem_v2": "v2: K VLM sửa bằng chuyển động + loại khớp kết hợp VLM/chuyển động",
        "sem_v2_mask": "sem_v2 + loss mask 2D an toàn (không hạng static, λ 0.03, tắt sau 10k)"}


def summarise(ev):
    parts = [p for p in ev["parts"] if "angle" in p]
    rev = [p for p in parts if p["gt_type"] == "r"]
    mean = lambda xs: st.mean(xs) if xs else None
    return {"K": f"{ev['K_pred']}/{ev['K_gt']}", "recall": ev["part_recall"], "type": ev["type_acc"],
            "ang": mean([p["angle"] for p in parts]), "pos": mean([p["distance"] for p in rev]),
            "motion": mean([p["theta_diff"] for p in parts if p["theta_diff"] == p["theta_diff"]]),
            "CDs": ev["CD_static"], "CDm": mean([p["CD_dynamic"] for p in parts if p.get("CD_dynamic") == p.get("CD_dynamic")]),
            "CDw": ev["CD_whole"], "parts": sorted(ev["parts"], key=lambda p: p["gt_part"])}


def paper_row(pp):
    if pp is None:
        return None
    if "mean" in pp:
        m = pp["mean"]
        return {"ang": m["angle"], "pos": m["distance"], "motion": m["theta_diff"], "CDm": m["CD_dynamic"],
                "CDs": pp["CD_static"], "CDw": pp["CD_whole"], "parts": None}
    pr = pp["per_part"]
    rev = [p["distance"] for p in pr if p["type"] == "r" and p["distance"] is not None]
    return {"ang": st.mean(p["angle"] for p in pr), "pos": st.mean(rev) if rev else None,
            "motion": st.mean(p["theta_diff"] for p in pr), "CDm": st.mean(p["CD_dynamic"] for p in pr),
            "CDs": pp["CD_static"], "CDw": pp["CD_whole"], "parts": pr}


def f(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", default="", help="dir with <scene>.jpg figures to link")
    a = ap.parse_args()
    paper = json.loads((HERE / "paper_artgs.json").read_text())
    res = defaultdict(lambda: defaultdict(list))   # scene -> method -> [summary per run]
    lifts, runs_of = {}, defaultdict(set)
    for run in a.runs:
        for p in Path(run).rglob("semart_eval.json"):
            m = re.search(r"(\w+/\w+/[\w.-]+)/(\w+?)_s(\d+)/train/ours_\d+/semart_eval.json$", str(p))
            if m:
                res[m.group(1)][m.group(2)].append(summarise(json.loads(p.read_text())))
                runs_of[m.group(1)].add(Path(run).name)
        for p in Path(run).rglob("lift_*/lift_report.json"):
            m = re.search(r"(\w+/\w+/[\w.-]+)/lift_\w+/lift_report.json$", str(p))
            if m:
                lifts[m.group(1)] = json.loads(p.read_text())

    L = ["# SemArt vs ArtGS: báo cáo so sánh", "",
         "Giao thức chạy lại (mọi method như nhau): 20k iteration, resolution 2, seed 0, đánh giá ở iteration "
         "cuối bằng `semart/eval_robust.py` (ghép part Hungarian). Paper ArtGS: resolution 1, checkpoint "
         "'best' chọn theo sai số khớp so với GT, trung bình 10 seed (DTA) / 3 lần (ArtGS-Multi).",
         "", "Chỉ số (càng thấp càng tốt): Ang = sai số góc trục (độ); Pos = sai số vị trí trục ×10 "
         "(chỉ khớp xoay); Motion = sai số trạng thái (độ với khớp xoay, m với khớp trượt); CD ×1000: "
         "s = thân tĩnh, m = part động (trung bình), w = toàn vật. K = dự đoán/GT; Type = tỉ lệ loại khớp đúng.", ""]
    def merge(runs):
        """mean of numeric fields over repeated runs; spread kept for the axis angle"""
        out = dict(runs[0])
        for k in ("recall", "type", "ang", "pos", "motion", "CDs", "CDm", "CDw"):
            v = [r[k] for r in runs if r[k] is not None]
            out[k] = st.mean(v) if v else None
        out["n"] = len(runs)
        out["ang_all"] = [r["ang"] for r in runs]
        return out
    res = {sc: {m: merge(v) for m, v in d.items()} for sc, d in res.items()}
    avg = defaultdict(lambda: defaultdict(list))
    for scene in sorted(res):
        L += [f"## {scene}", "", f"Run: {', '.join(sorted(runs_of[scene]))}", ""]
        L += ["| Method | n | K | Recall | Type | Ang | Pos | Motion | CD-s | CD-m | CD-w |", "|---" * 11 + "|"]
        pr = paper_row(paper.get(scene))
        if pr:
            L.append(f"| **ArtGS (paper)** | - | - | - | - | {f(pr['ang'])} | {f(pr['pos'])} | {f(pr['motion'])} | "
                     f"{f(pr['CDs'], 2)} | {f(pr['CDm'], 2)} | {f(pr['CDw'], 2)} |")
            for k in ("ang", "pos", "motion", "CDs", "CDm", "CDw"):
                if pr[k] is not None:
                    avg["ArtGS (paper)"][k].append(pr[k])
        for m in [x for x in METHOD_ORDER if x in res[scene]]:
            s = res[scene][m]
            L.append(f"| `{m}` | {s['n']} | {s['K']} | {f(s['recall'], 2)} | {f(s['type'], 2)} | {f(s['ang'])} | {f(s['pos'])} | "
                     f"{f(s['motion'])} | {f(s['CDs'], 2)} | {f(s['CDm'], 2)} | {f(s['CDw'], 2)} |")
            for k in ("ang", "pos", "motion", "CDs", "CDm", "CDw", "type", "recall"):
                if s[k] is not None:
                    avg[m][k].append(s[k])
        if pr and pr["parts"]:
            L += ["", "Theo từng part (paper / chạy lại `artgs` / tốt nhất của SemArt theo CD-m):", "",
                  "| Part | Nguồn | Ang | Pos | Motion | CD-m |", "|---|---|---|---|---|---|"]
            sem = [m for m in res[scene] if m.startswith("sem")]
            best = min(sem, key=lambda m: res[scene][m]["CDm"] or 1e9) if sem else None
            for j, pp in enumerate(pr["parts"]):
                L.append(f"| {j} ({pp['type']}) | paper | {f(pp['angle'])} | {f(pp['distance'])} | "
                         f"{f(pp['theta_diff'])} | {f(pp['CD_dynamic'], 2)} |")
                for m in [x for x in ("artgs", best) if x and x in res[scene]]:
                    q = res[scene][m]["parts"][j] if j < len(res[scene][m]["parts"]) else {}
                    L.append(f"| | `{m}` | {f(q.get('angle'))} | {f(q.get('distance') if pp['type'] == 'r' else None)} | "
                             f"{f(q.get('theta_diff'))} | {f(q.get('CD_dynamic'), 2)} |")
        lr = lifts.get(scene)
        if lr:
            L += ["", f"Lift: VLM {lr.get('n_vlm')} → K={lr.get('num_slots')} `{lr.get('joint_types')}` "
                  f"(eigengap {lr.get('k_eigengap')}); điểm chuyển động {lr.get('n_moving')}, phủ mask "
                  f"{lr.get('covered_frac', 0):.0%}; instance SAM3 giữ {lr.get('inst_kept')} / bỏ {lr.get('inst_dropped')}."]
        if a.figs:
            fig = Path(a.figs) / (scene.replace("/", "_") + ".jpg")
            if fig.exists():
                L += ["", f"Hình: `{fig}`"]
        L.append("")
    L += ["## Trung bình trên các vật", "", "| Method | n vật | Type | Recall | Ang | Pos | Motion | CD-s | CD-m | CD-w |",
          "|---" * 10 + "|"]
    for m in ["ArtGS (paper)"] + METHOD_ORDER:
        if m not in avg:
            continue
        d = avg[m]
        mm = lambda k, n=3: f(st.mean(d[k]), n) if d.get(k) else "-"
        n = max(len(v) for v in d.values())
        L.append(f"| {m if m.startswith('ArtGS') else '`' + m + '`'} | {n} | {mm('type', 2)} | {mm('recall', 2)} | "
                 f"{mm('ang')} | {mm('pos')} | {mm('motion')} | {mm('CDs', 2)} | {mm('CDm', 2)} | {mm('CDw', 2)} |")
    L += ["", "Methods: " + "; ".join(f"`{k}` = {v}" for k, v in DESC.items()), ""]
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
