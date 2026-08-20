#!/usr/bin/env python3
"""Summarize CARDIUM in-domain fusion runs (ALVG / MIL / FT + C2-b baselines).

  python -u experiments/summarize_cardium_in_domain.py \
    --fusion-root outputs/cardium_in_domain_fusion
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fusion-root",
        type=Path,
        default=ROOT / "outputs/cardium_in_domain_fusion",
    )
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def mean_std(xs: list[float]) -> tuple[float, float, int]:
    a = np.array([float(x) for x in xs if x is not None and x == x], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0), int(a.size)


def parse_dir(name: str) -> dict:
    m = re.match(r"^(.+)_seed(\d+)(?:__r-(.+))?$", name)
    if not m:
        return {"method": name, "seed": None, "recipe": None}
    return {"method": m.group(1), "seed": int(m.group(2)), "recipe": m.group(3)}


def load_run(json_path: Path) -> dict | None:
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    folds = d.get("folds") or []
    if not folds:
        return None
    f1s, aucs = [], []
    c2b_f1s, c2b_aucs = [], []
    for f0 in folds:
        f1s.append(f0.get("f1_fusion"))
        aucs.append(f0.get("auc_fusion"))
        c2b_f1s.append(f0.get("f1_c2b_mean"))
        c2b_aucs.append(f0.get("auc_c2b_mean"))
    return {
        "m0": d.get("m0_features"),
        "extra": d.get("extra_feats"),
        "n_folds": len(folds),
        "f1_mean": float(np.nanmean(f1s)),
        "auc_mean": float(np.nanmean(aucs)),
        "c2b_f1": float(np.nanmean(c2b_f1s)),
        "c2b_auc": float(np.nanmean(c2b_aucs)),
        "seed": d.get("seed"),
        "method": d.get("method"),
    }


def main() -> int:
    args = parse_args()
    rows = []
    if not args.fusion_root.is_dir():
        print(f"missing {args.fusion_root}")
        return 1
    for run_dir in sorted(args.fusion_root.iterdir()):
        if not run_dir.is_dir():
            continue
        jsons = sorted(run_dir.glob("view_token_results_*.json"))
        if not jsons:
            continue
        meta = parse_dir(run_dir.name)
        m = load_run(jsons[0])
        if m is None:
            continue
        rows.append({**meta, **m})

    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = f"{r['recipe']} ({r['method']})" if r.get("recipe") else str(r.get("method"))
        by[key].append(r)

    lines = [
        "# CARDIUM in-domain fusion",
        "",
        "Train+test on CARDIUM folds (protocol transfer of tertiary ALVG recipe).",
        "",
        "| recipe | F1 mean±std | AUC mean±std | n | C2-b mean F1† |",
        "|---|---:|---:|---:|---:|",
    ]
    ranked = []
    for k, rs in by.items():
        fm, fs, fn = mean_std([r["f1_mean"] for r in rs])
        am, as_, _ = mean_std([r["auc_mean"] for r in rs])
        cm, cs, _ = mean_std([r["c2b_f1"] for r in rs])
        ranked.append((fm, k, fs, fn, am, as_, cm))
    ranked.sort(reverse=True)
    for fm, k, fs, fn, am, as_, cm in ranked:
        lines.append(
            f"| `{k}` | {fm:.4f}±{fs:.4f} | {am:.4f}±{as_:.4f} | {fn} | {cm:.4f} |"
        )
    lines.append("")
    lines.append("† C2-b patient-mean from the same runs (appearance pooling baseline).")
    lines.append("")
    lines.append("## Per-seed")
    for fm, k, *_ in ranked:
        lines.extend(["", f"### {k}", "", "| seed | F1 | AUC | C2-b F1 |", "|---:|---:|---:|---:|"])
        for r in sorted(by[k], key=lambda x: x.get("seed") or 0):
            lines.append(
                f"| {r.get('seed')} | {r['f1_mean']:.4f} | {r['auc_mean']:.4f} | {r['c2b_f1']:.4f} |"
            )

    out = args.out or (args.fusion_root / "CARDIUM_IN_DOMAIN.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    for fm, k, fs, fn, am, as_, cm in ranked:
        print(f"  {fm:.4f}±{fs:.4f}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
