#!/usr/bin/env python3
"""Summarize missing-view robustness for ALVG vs other gated heads.

Reads view_decomp.keep_k from view_token_results_*.json (no GPU).

keep_k=4 → full bag; keep_k=3 → random-drop 1 view (combo mean);
keep_k=2 → random-drop 2; keep_k=1 → single-view mean.

  python -u experiments/summarize_alvg_missing_views.py \\
    --roots docs/paper/exp/_unpack_beat_mil/outputs/tertiary_20241125_vs_chd_fusion_raw \\
            outputs/tertiary_missing_view_heads \\
    --latex --out-dir outputs/tertiary_missing_view_summary
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "val_f1_tuned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[
            ROOT / "outputs" / "tertiary_20241125_vs_chd_fusion_raw",
            ROOT / "outputs" / "tertiary_missing_view_heads",
        ],
    )
    p.add_argument(
        "--methods",
        default="anatomy_graph,feature_transformer,graph_transformer",
        help="Comma list of fusion method directory prefixes",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "tertiary_missing_view_summary")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--metric-mode", default=PRIMARY)
    return p.parse_args()


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [x for x in xs if x == x]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _fmt(m: float | None, s: float | None = None) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "—"
    if s is None or s == 0.0:
        return f"{m:.4f}"
    return f"{m:.4f}±{s:.4f}"


def _tex(m: float | None, s: float | None = None) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "---"
    if s is None or s == 0.0:
        return f"{m:.4f}"
    return f"{m:.4f}\\pm{s:.4f}"


def _recipe_from_path(p: Path) -> str:
    m = re.search(r"__r-([^/]+)", str(p))
    if m:
        return m.group(1)
    name = p.parent.name
    m2 = re.match(r"([a-z_]+)_seed", name)
    return m2.group(1) if m2 else name


def _keep_metric(keep_block: dict, mode: str, key: str) -> float | None:
    vals: list[float] = []
    for _name, modes in (keep_block or {}).items():
        if not isinstance(modes, dict):
            continue
        vt = modes.get(mode) or modes.get(PRIMARY) or {}
        if isinstance(vt, dict) and vt.get(key) is not None:
            vals.append(float(vt[key]))
    return statistics.mean(vals) if vals else None


def load_run(path: Path, mode: str) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    folds = data.get("folds") or []
    vd = None
    if folds and isinstance(folds[0], dict):
        vd = folds[0].get("view_decomp")
    if not vd:
        vd = data.get("view_decomp")
    if not isinstance(vd, dict):
        return None
    keep = vd.get("keep_k") or {}
    row = {
        "seed": data.get("seed"),
        "method": data.get("method") or _recipe_from_path(path),
        "m0": data.get("m0_features"),
        "extra": data.get("extra_feats"),
        "path": str(path),
        "keep_f1": {},
        "keep_auc": {},
    }
    for k, block in keep.items():
        if not isinstance(block, dict):
            continue
        f1 = _keep_metric(block, mode, "f1")
        auc = _keep_metric(block, mode, "auc")
        if f1 is not None:
            row["keep_f1"][str(k)] = f1
        if auc is not None:
            row["keep_auc"][str(k)] = auc
    return row if row["keep_f1"] else None


def collect(roots: list[Path], methods: list[str], mode: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("view_token_results_*.json")):
            parent = p.parent.name
            if not any(parent.startswith(f"{m}_") for m in methods):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            row = load_run(p, mode)
            if not row:
                continue
            seen.add(key)
            xa = row.get("extra") or "none"
            m0 = row.get("m0") or "?"
            tag = f"{row['method']}|m0={m0}|xa={xa}"
            by[tag].append(row)
    return by


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "f1": {}, "auc": {}, "delta_f1": {}}
    for k in ("4", "3", "2", "1"):
        f1s = [r["keep_f1"][k] for r in rows if k in r["keep_f1"]]
        aucs = [r["keep_auc"][k] for r in rows if k in r["keep_auc"]]
        out["f1"][k] = _mean_std(f1s)
        out["auc"][k] = _mean_std(aucs)
    full_m, _ = out["f1"].get("4", (None, None))
    if full_m is not None:
        for k in ("3", "2", "1"):
            m, _ = out["f1"].get(k, (None, None))
            out["delta_f1"][k] = None if m is None else m - full_m
    return out


def main() -> int:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    by = collect(args.roots, methods, args.metric_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ranked: list[tuple[float, str, dict, list[dict]]] = []
    for tag, rows in by.items():
        agg = aggregate(rows)
        ranked.append((agg["f1"].get("4", (0.0, 0.0))[0] or 0.0, tag, agg, rows))
    ranked.sort(reverse=True)

    lines = [
        "# Missing-view robustness (keep_k)",
        "",
        f"Metric mode: `{args.metric_mode}`",
        f"Roots: {', '.join(str(r) for r in args.roots)}",
        "",
        "| method | n | keep4 F1 | keep3 (drop1) | keep2 (drop2) | keep1 | Δdrop1 | Δdrop2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    payload: dict = {"metric_mode": args.metric_mode, "methods": {}}
    for _score, tag, agg, rows in ranked:
        d1 = agg["delta_f1"].get("3")
        d2 = agg["delta_f1"].get("2")
        lines.append(
            "| `{tag}` | {n} | {f4} | {f3} | {f2} | {f1} | {d1} | {d2} |".format(
                tag=tag,
                n=agg["n"],
                f4=_fmt(*agg["f1"]["4"]),
                f3=_fmt(*agg["f1"]["3"]),
                f2=_fmt(*agg["f1"]["2"]),
                f1=_fmt(*agg["f1"]["1"]),
                d1=("—" if d1 is None else f"{d1:+.4f}"),
                d2=("—" if d2 is None else f"{d2:+.4f}"),
            )
        )
        payload["methods"][tag] = {
            "n": agg["n"],
            "keep_f1": {
                k: {"mean": agg["f1"][k][0], "std": agg["f1"][k][1]} for k in ("4", "3", "2", "1")
            },
            "delta_vs_full": agg["delta_f1"],
            "seeds": sorted({r["seed"] for r in rows if r.get("seed") is not None}),
        }

    md_path = args.out_dir / "MISSING_VIEW.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out_dir / "missing_view_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(md_path.read_text(encoding="utf-8"))

    if args.latex:
        tex_lines = [
            "% Auto-generated by summarize_alvg_missing_views.py",
            "\\begin{table}[!t]",
            "\\centering",
            "\\caption{Missing-view robustness under the gated protocol "
            "(combo-mean over keep-$k$ subsets; validation F1-tuned). "
            "ALVG uses anatomy-adjacent imputation; drop-1/2 correspond to keep-3/2.}",
            "\\label{tab:missing_main}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Setting & F1 & AUC \\\\",
            "\\midrule",
        ]
        alvg_tag = next((t for _, t, _, _ in ranked if t.startswith("anatomy_graph")), None)
        if alvg_tag:
            agg = aggregate(by[alvg_tag])
            for label, k in (
                ("Full 4 views (ALVG)", "4"),
                ("Random drop 1 view (ALVG)", "3"),
                ("Random drop 2 views (ALVG)", "2"),
            ):
                fm, fs = agg["f1"][k]
                am, as_ = agg["auc"][k]
                tex_lines.append(f"{label} & ${_tex(fm, fs)}$ & ${_tex(am, as_)}$ \\\\")
        # fair comparators if present
        for pref, nice in (
            ("feature_transformer|", "Feature Transformer"),
            ("graph_transformer|", "GNN"),
        ):
            tag = next((t for _, t, _, _ in ranked if t.startswith(pref)), None)
            if not tag:
                continue
            agg = aggregate(by[tag])
            tex_lines.append("\\midrule")
            for label, k in (
                (f"Full 4 views ({nice})", "4"),
                (f"Random drop 1 view ({nice})", "3"),
                (f"Random drop 2 views ({nice})", "2"),
            ):
                fm, fs = agg["f1"][k]
                am, as_ = agg["auc"][k]
                tex_lines.append(f"{label} & ${_tex(fm, fs)}$ & ${_tex(am, as_)}$ \\\\")
        tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
        tex_path = args.out_dir / "missing_view_table.tex"
        tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
        print(f"Wrote {tex_path}")

    if not ranked:
        print("WARNING: no view_decomp.keep_k found under roots")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
