#!/usr/bin/env python3
"""Summarize R6 year-holdout patient-mean runs.

  python -u experiments/summarize_tertiary_year_holdout.py \\
    --root outputs/tertiary_r6_year_holdout
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "outputs/tertiary_r6_year_holdout")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None, None
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


def main() -> int:
    args = parse_args()
    out = args.out or (args.root / "summary.json")
    inv_path = args.root / "manifests" / "inventory.json"
    inventory = json.loads(inv_path.read_text()) if inv_path.is_file() else {}

    rows = []
    for hold_dir in sorted(args.root.glob("hold_*")):
        year = hold_dir.name.replace("hold_", "")
        seed_jsons = sorted(hold_dir.glob("patient_mean/seed*/results.json"))
        if not seed_jsons:
            continue
        f1s, aucs = [], []
        n_pos = n_neg = None
        for p in seed_jsons:
            d = json.loads(p.read_text(encoding="utf-8"))
            m = d.get("test_patient_mean") or {}
            if m.get("f1") is not None:
                f1s.append(float(m["f1"]))
            if m.get("auc") is not None:
                aucs.append(float(m["auc"]))
            n_pos = m.get("n_pos", n_pos)
            n_neg = m.get("n_neg", n_neg)
        f1_mu, f1_sd = mean_std(f1s)
        auc_mu, auc_sd = mean_std(aucs)
        rows.append({
            "year": int(year) if year.isdigit() else year,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_seeds": len(seed_jsons),
            "f1_mean": f1_mu,
            "f1_std": f1_sd,
            "auc_mean": auc_mu,
            "auc_std": auc_sd,
        })

    summary = {
        "inventory": inventory,
        "rows": rows,
        "n_years": len(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Year-holdout ablation (patient-mean)",
        "",
        "| hold year | n_pos/n_neg | F1 | AUC | seeds |",
        "|---:|---:|---:|---:|---:|",
    ]
    latex_lines = []
    for r in rows:
        f1s = f"{r['f1_mean']:.4f}" if r["f1_mean"] is not None else "—"
        if r["f1_std"] and r["n_seeds"] > 1:
            f1s = f"{r['f1_mean']:.4f}±{r['f1_std']:.4f}"
        aucs = f"{r['auc_mean']:.4f}" if r["auc_mean"] is not None else "—"
        if r["auc_std"] and r["n_seeds"] > 1:
            aucs = f"{r['auc_mean']:.4f}±{r['auc_std']:.4f}"
        md.append(
            f"| {r['year']} | {r['n_pos']}/{r['n_neg']} | {f1s} | {aucs} | {r['n_seeds']} |"
        )
        if r["f1_mean"] is not None:
            latex_lines.append(
                f"{r['year']} & {r['n_pos']}/{r['n_neg']} & "
                f"{r['f1_mean']:.4f} & --- \\\\"
            )

    md_path = args.root / "AGGREGATE.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (args.root / "LATEX_SNIPPET.txt").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
