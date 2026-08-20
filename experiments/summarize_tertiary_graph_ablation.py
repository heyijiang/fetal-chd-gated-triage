#!/usr/bin/env python3
"""Aggregate tertiary graph-ablation dumps (patient-level fusion F1).

  python -u experiments/summarize_tertiary_graph_ablation.py
  python -u experiments/summarize_tertiary_graph_ablation.py \\
      --root outputs/tertiary_graph_ablation
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
        "--root",
        type=Path,
        default=ROOT / "outputs/tertiary_graph_ablation",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON dump (default: <root>/summary_graph_ablation.json)",
    )
    return p.parse_args()


def _f1(blob: dict) -> float | None:
    if blob.get("mean_f1_fusion") is not None:
        return float(blob["mean_f1_fusion"])
    sm = ((blob.get("summary_by_mode") or {}).get("fusion") or {})
    for key in ("youden", "f1_max", "default", "val_youden"):
        row = sm.get(key) or {}
        if row.get("f1") is not None:
            return float(row["f1"])
    if sm:
        first = next(iter(sm.values()))
        if isinstance(first, dict) and first.get("f1") is not None:
            return float(first["f1"])
    return None


def _auc(blob: dict) -> float | None:
    if blob.get("mean_auc_fusion") is not None:
        return float(blob["mean_auc_fusion"])
    sm = ((blob.get("summary_by_mode") or {}).get("fusion") or {})
    for row in sm.values():
        if isinstance(row, dict) and row.get("auc") is not None:
            return float(row["auc"])
    return None


def _arm_from_dir(name: str) -> str:
    m = re.search(r"__adj_([A-Za-z0-9_]+)$", name)
    if m:
        return m.group(1)
    if name.startswith("view_mean"):
        return "view_mean"
    if name.startswith("anatomy_graph"):
        return "anatomy"
    return name


def main() -> int:
    args = parse_args()
    root = args.root
    if not root.is_dir():
        print(f"ERROR missing {root}")
        return 1
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for js in sorted(root.glob("*/view_token_results_*.json")):
        blob = json.loads(js.read_text(encoding="utf-8"))
        arm = str(blob.get("graph_adj") or _arm_from_dir(js.parent.name))
        if js.parent.name.endswith("__adj_view_mean") or blob.get("method") == "view_mean":
            arm = "view_mean"
        f1 = _f1(blob)
        if f1 is None:
            print(f"WARN no F1 in {js}")
            continue
        by_arm[arm].append(
            {
                "seed": int(blob.get("seed") or 0),
                "f1": f1,
                "auc": _auc(blob),
                "path": str(js),
                "method": blob.get("method"),
            }
        )

    summary = {}
    print(f"{'arm':12s}  n  F1 mean±std          AUC mean±std")
    order = ["anatomy", "random", "full", "mean", "view_mean"]
    arms = [a for a in order if a in by_arm] + [a for a in sorted(by_arm) if a not in order]
    for arm in arms:
        rows = by_arm[arm]
        f1s = np.array([r["f1"] for r in rows], dtype=np.float64)
        aucs = np.array([r["auc"] for r in rows if r["auc"] is not None], dtype=np.float64)
        rec = {
            "n_seeds": int(len(rows)),
            "seeds": [int(r["seed"]) for r in rows],
            "f1_mean": float(f1s.mean()),
            "f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
            "auc_mean": float(aucs.mean()) if len(aucs) else None,
            "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else (0.0 if len(aucs) else None),
            "runs": rows,
        }
        summary[arm] = rec
        auc_s = (
            f"{rec['auc_mean']:.3f}±{rec['auc_std']:.3f}"
            if rec["auc_mean"] is not None
            else "n/a"
        )
        print(
            f"{arm:12s}  {rec['n_seeds']:2d}  "
            f"{rec['f1_mean']:.3f}±{rec['f1_std']:.3f}     {auc_s}"
        )

    if "anatomy" in summary and "random" in summary:
        d = summary["anatomy"]["f1_mean"] - summary["random"]["f1_mean"]
        print(
            f"\nΔF1 anatomy−random = {d:+.4f}  "
            "(do not claim anatomy≫random unless CI excludes 0)"
        )

    out_path = args.out or (root / "summary_graph_ablation.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
