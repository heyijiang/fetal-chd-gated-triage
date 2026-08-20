#!/usr/bin/env python3
"""Rank fusion runs vs Attention-MIL baseline (gated tertiary protocol).

Scans fusion output dirs (supports RUN_TAG_SUFFIX __r-<recipe>).

  python -u experiments/summarize_fusion_vs_mil.py \
    --fusion-root outputs/tertiary_20241125_vs_chd_fusion_raw \
    --baseline attention_mil

  python -u experiments/summarize_fusion_vs_mil.py \
    --fusion-root outputs/tertiary_fusion_beat_mil \
    --also-root outputs/tertiary_20241125_vs_chd_fusion_raw
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize fusion F1/AUC vs MIL baseline")
    p.add_argument(
        "--fusion-root",
        type=Path,
        action="append",
        default=[],
        help="Fusion output root(s); repeat for multiple",
    )
    p.add_argument(
        "--also-root",
        type=Path,
        action="append",
        default=[],
        help="Additional roots (e.g. existing fusion_raw for MIL baseline)",
    )
    p.add_argument("--baseline", default="attention_mil", help="Baseline fusion method name")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown report (default: <first-root>/BEAT_MIL.md)",
    )
    p.add_argument("--metric", default="f1", choices=["f1", "auc"])
    return p.parse_args()


def mean_std(xs: list[float]) -> tuple[float, float, int]:
    a = np.array(xs, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return (
        float(a.mean()),
        float(a.std(ddof=1) if a.size > 1 else 0.0),
        int(a.size),
    )


def parse_run_dir(name: str) -> dict:
    """anatomy_graph_seed42__r-alvg_core3 → method, seed, recipe."""
    m = re.match(r"^(.+)_seed(\d+)(?:__r-(.+))?$", name)
    if not m:
        return {"method": name, "seed": None, "recipe": None, "dir": name}
    return {
        "method": m.group(1),
        "seed": int(m.group(2)),
        "recipe": m.group(3),
        "dir": name,
    }


def load_metrics(json_path: Path) -> dict | None:
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    f0 = (d.get("folds") or [{}])[0]
    n_te = f0.get("n_test_patients")
    counts = f0.get("test_cohort_counts") or {}
    return {
        "f1": float(f0.get("f1_fusion") or float("nan")),
        "auc": float(f0.get("auc_fusion") or float("nan")),
        "n_te": n_te,
        "counts": counts,
        "m0": d.get("m0_features"),
        "extra": d.get("extra_feats"),
        "method": d.get("method") or d.get("fusion"),
        "seed": d.get("seed"),
        "path": str(json_path),
    }


def collect_roots(roots: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            jsons = sorted(run_dir.glob("view_token_results_*.json"))
            if not jsons:
                continue
            meta = parse_run_dir(run_dir.name)
            m = load_metrics(jsons[0])
            if m is None:
                continue
            rows.append({**meta, **m})
    return rows


def group_key(row: dict) -> str:
    if row.get("recipe"):
        return f"{row['recipe']} ({row['method']})"
    return str(row.get("method") or row.get("dir"))


def main() -> int:
    args = parse_args()
    roots = list(args.fusion_root or []) + list(args.also_root or [])
    if not roots:
        roots = [ROOT / "outputs/tertiary_20241125_vs_chd_fusion_raw"]
    rows = collect_roots(roots)
    if not rows:
        print("No fusion JSONs found.")
        return 1

    metric = args.metric
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[group_key(r)].append(r)

    baseline_key = None
    baseline_vals: list[float] = []
    for k, rs in by_group.items():
        if rs[0].get("method") == args.baseline and rs[0].get("recipe") in (None, "mil_ref"):
            baseline_key = k
            baseline_vals = [r[metric] for r in rs if r[metric] == r[metric]]
            break
    if not baseline_vals:
        for k, rs in by_group.items():
            if rs[0].get("method") == args.baseline:
                baseline_key = k
                baseline_vals = [r[metric] for r in rs if r[metric] == r[metric]]
                break

    b_mean, b_std, b_n = mean_std(baseline_vals)

    ranked = []
    for k, rs in by_group.items():
        vals = [r[metric] for r in rs if r[metric] == r[metric]]
        m, s, n = mean_std(vals)
        ntes = {r.get("n_te") for r in rs}
        ranked.append(
            {
                "group": k,
                "method": rs[0].get("method"),
                "recipe": rs[0].get("recipe"),
                "m0": rs[0].get("m0"),
                "extra": rs[0].get("extra"),
                "mean": m,
                "std": s,
                "n": n,
                "n_te": ntes,
                "delta": m - b_mean if b_mean == b_mean else float("nan"),
                "beats_mil": m > b_mean if b_mean == b_mean else False,
            }
        )
    ranked.sort(key=lambda x: (-x["mean"], x["group"]))

    lines = [
        "# Fusion vs Attention-MIL",
        "",
        f"Baseline **{args.baseline}**: {metric.upper()} = **{b_mean:.4f}±{b_std:.4f}** (n={b_n})",
        "",
        f"| rank | recipe / method | {metric.upper()} mean±std | n | n_te | Δ vs MIL | beats MIL |",
        "|---:|---|---:|---:|---|---:|:---:|",
    ]
    for i, r in enumerate(ranked, 1):
        flag = "✓" if r["beats_mil"] and r["group"] != baseline_key else ""
        lines.append(
            f"| {i} | `{r['group']}` | {r['mean']:.4f}±{r['std']:.4f} | {r['n']} | "
            f"{','.join(str(x) for x in sorted(r['n_te']))} | {r['delta']:+.4f} | {flag} |"
        )

  # detail: per-seed for top candidates
    lines.extend(["", "## Top candidates (per-seed)", ""])
    shown = 0
    for r in ranked:
        if r["group"] == baseline_key:
            continue
        if shown >= 8:
            break
        rs = by_group[r["group"]]
        lines.append(f"### {r['group']}")
        lines.append("")
        lines.append("| seed | F1 | AUC | n_te |")
        lines.append("|---:|---:|---:|---:|")
        for row in sorted(rs, key=lambda x: x.get("seed") or 0):
            lines.append(
                f"| {row.get('seed')} | {row['f1']:.4f} | {row['auc']:.4f} | {row.get('n_te')} |"
            )
        lines.append("")
        shown += 1

    out_path = args.out
    if out_path is None:
        out_path = roots[0] / "BEAT_MIL.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Baseline {args.baseline} {metric}={b_mean:.4f}±{b_std:.4f} (n={b_n})")
    print(f"Wrote {out_path}")
    for r in ranked[:6]:
        mark = " <-- BEATS MIL" if r["beats_mil"] and r["group"] != baseline_key else ""
        print(f"  {r['mean']:.4f}±{r['std']:.4f}  {r['group']}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
