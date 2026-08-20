#!/usr/bin/env python3
"""Summarize structured leave-one-view-out (R7 ALVG follow-up).

Uses existing view_decomp.leave_one_out from matched CLIP+view_anat runs
(ALVG / GNN / FT). No GPU.

  python -u experiments/summarize_structured_loo.py \\
    --roots docs/paper/exp/_unpack docs/paper/exp/_unpack_beat_mil \\
    --latex --out-dir outputs/tertiary_r7_structured_loo
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("four_chamber", "lvot", "rvot", "vvt")
LABELS = {
    "four_chamber": "4CH",
    "lvot": "LVOT",
    "rvot": "RVOT",
    "vvt": "3VT",
}
METHOD_NAMES = {
    "anatomy_graph": "ALVG",
    "graph_transformer": "GNN",
    "feature_transformer": "FT",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            ROOT / "docs/paper/exp",
            ROOT / "outputs/tertiary_20241125_vs_chd_fusion_raw",
            ROOT / "outputs/tertiary_missing_view_heads",
        ],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_r7_structured_loo",
    )
    p.add_argument("--latex", action="store_true")
    return p.parse_args()


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [float(x) for x in xs if x is not None and x == x]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _tex(m: float | None, s: float | None = None) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "---"
    if s is None or s == 0.0:
        return f"{m:.4f}"
    return f"{m:.4f}\\pm{s:.4f}"


def discover(roots: list[Path]) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("view_token_results_*.json"):
            key = str(p.resolve())
            if key in seen:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            method = d.get("method") or ""
            if method not in METHOD_NAMES:
                continue
            extra = str(d.get("extra_feats") or "")
            if "view_anat" not in extra and "view_anat" not in str(p):
                continue
            vd = d.get("view_decomp")
            if not vd and d.get("folds"):
                vd = (d["folds"][0] or {}).get("view_decomp")
            if not vd or not vd.get("leave_one_out"):
                continue
            seen.add(key)
            full = ((vd.get("keep_k") or {}).get("4") or {}).get("all") or {}
            full_m = full.get("val_f1_tuned") or {}
            loo = {}
            for v, modes in (vd.get("leave_one_out") or {}).items():
                m = modes.get("val_f1_tuned") or {}
                loo[v] = {"f1": m.get("f1"), "auc": m.get("auc")}
            rows.append(
                {
                    "method": method,
                    "seed": d.get("seed"),
                    "path": str(p),
                    "full_f1": full_m.get("f1"),
                    "full_auc": full_m.get("auc"),
                    "loo": loo,
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    rows = discover(args.roots)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)

    summary: dict = {"n_dumps": len(rows), "methods": {}}
    for method, name in METHOD_NAMES.items():
        rs = by.get(method) or []
        if not rs:
            continue
        full_f1 = _mean_std([r["full_f1"] for r in rs])
        block = {
            "display": name,
            "n_seeds": len(rs),
            "full_f1_mean": full_f1[0],
            "full_f1_std": full_f1[1],
            "leave_one_out": {},
        }
        deltas = []
        for v in VIEWS:
            f1 = _mean_std([r["loo"].get(v, {}).get("f1") for r in rs])
            auc = _mean_std([r["loo"].get(v, {}).get("auc") for r in rs])
            delta = None
            if full_f1[0] is not None and f1[0] is not None:
                delta = float(f1[0] - full_f1[0])
                deltas.append(delta)
            block["leave_one_out"][v] = {
                "label": LABELS[v],
                "f1_mean": f1[0],
                "f1_std": f1[1],
                "auc_mean": auc[0],
                "auc_std": auc[1],
                "delta_f1": delta,
            }
        block["mean_delta_f1"] = sum(deltas) / len(deltas) if deltas else None
        block["worst_delta_f1"] = min(deltas) if deltas else None
        summary["methods"][method] = block

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = ["# Structured leave-one-view-out (matched CLIP+view_anat)", ""]
    for method, name in METHOD_NAMES.items():
        b = summary["methods"].get(method)
        if not b:
            continue
        md.append(
            f"## {name} (n={b['n_seeds']})  full F1={b['full_f1_mean']:.4f}±{b['full_f1_std']:.4f}"
        )
        md.append("| leave out | F1 | ΔF1 |")
        md.append("|---|---:|---:|")
        for v in VIEWS:
            lo = b["leave_one_out"][v]
            md.append(
                f"| {lo['label']} | {lo['f1_mean']:.4f}±{lo['f1_std']:.4f} | {lo['delta_f1']:+.4f} |"
            )
        md.append(
            f"mean Δ={b['mean_delta_f1']:+.4f}  worst Δ={b['worst_delta_f1']:+.4f}"
        )
        md.append("")
    (args.out_dir / "AGGREGATE.md").write_text("\n".join(md), encoding="utf-8")

    if args.latex:
        lines = [
            "% Auto-generated by summarize_structured_loo.py",
            "% tab:structured_loo",
        ]
        for method, name in METHOD_NAMES.items():
            b = summary["methods"].get(method)
            if not b:
                continue
            lines.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{name}:}}}}\\\\")
            lines.append(
                f"Full 4 views & ${_tex(b['full_f1_mean'], b['full_f1_std'])}$ & --- \\\\"
            )
            for v in VIEWS:
                lo = b["leave_one_out"][v]
                d = lo["delta_f1"]
                dtex = "---" if d is None else f"{d:+.4f}"
                lines.append(
                    f"Leave out {lo['label']} & ${_tex(lo['f1_mean'], lo['f1_std'])}$ & ${dtex}$ \\\\"
                )
            lines.append("\\midrule")
        (args.out_dir / "LATEX_SNIPPET.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    print(f"OK n={len(rows)} → {args.out_dir}/summary.json")
    for method, name in METHOD_NAMES.items():
        b = summary["methods"].get(method)
        if b:
            print(
                f"  {name}: meanΔ={b['mean_delta_f1']:+.4f} worstΔ={b['worst_delta_f1']:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
