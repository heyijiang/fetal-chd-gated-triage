#!/usr/bin/env python3
"""Same-patient hybrid F1: gated score if in fusion dump, else full-frame.

Uses each scorer's own val-F1 threshold (two cuts, one decision rule).
Does not mix scores onto one ROC.

  python -u experiments/eval_tertiary_hybrid_sameset.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fullframe-json",
        type=Path,
        default=ROOT
        / "docs/paper/exp/outputs/tertiary_20241125_vs_chd_patient_full_raw_seeds/seed42/results.json",
    )
    p.add_argument(
        "--alvg-root",
        type=Path,
        default=ROOT / "docs/paper/exp/_unpack_beat_mil/outputs/tertiary_20241125_vs_chd_fusion_raw",
    )
    p.add_argument("--alvg-glob", default="anatomy_graph_seed*")
    p.add_argument(
        "--mil-root",
        type=Path,
        default=ROOT / "docs/paper/exp/_unpack_final/outputs/tertiary_feat_ablation",
    )
    p.add_argument("--mil-glob", default="attention_mil_seed*__r-mil_clip_anat")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/tertiary_hybrid_sameset.json",
    )
    return p.parse_args()


def f1_at(y, pred) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_ff(path: Path) -> tuple[dict[str, tuple[int, float]], float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("test_patient_mean_scores") or []
    scores = {str(r["patient_id"]): (int(r["label"]), float(r["score"])) for r in rows}
    thr = float(data.get("threshold_patient_mean_val_tuned") or 0.5)
    return scores, thr


def find_json(run_dir: Path) -> Path | None:
    hits = sorted(run_dir.glob("view_token_results_*.json"))
    return hits[0] if hits else None


def load_fusion_run(jpath: Path) -> dict | None:
    data = json.loads(jpath.read_text(encoding="utf-8"))
    fold = (data.get("folds") or [{}])[0]
    ps = fold.get("patient_scores") or {}
    block = ps.get("fusion") or {}
    pids = ps.get("pids_test") or []
    y = block.get("y_test") or []
    p = block.get("p_test") or []
    if not pids or not y or not p:
        return None
    thr = float((fold.get("fusion") or {}).get("val_f1_tuned", {}).get("threshold") or 0.5)
    scores = {str(a): (int(b), float(c)) for a, b, c in zip(pids, y, p)}
    return {"seed": data.get("seed"), "scores": scores, "thr": thr, "path": str(jpath)}


def collect(root: Path, glob_pat: str) -> list[dict]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.glob(glob_pat)):
        if not d.is_dir():
            continue
        j = find_json(d)
        if j is None:
            continue
        run = load_fusion_run(j)
        if run:
            out.append(run)
    return out


def eval_run(ff: dict, thr_ff: float, run: dict) -> dict:
    gated = run["scores"]
    thr_g = run["thr"]
    pids = sorted(ff)
    y, pred, gated_flag = [], [], []
    n_fb = 0
    for pid in pids:
        lab, s_ff = ff[pid]
        y.append(lab)
        if pid in gated:
            pred.append(int(gated[pid][1] >= thr_g))
            gated_flag.append(1)
        else:
            pred.append(int(s_ff >= thr_ff))
            gated_flag.append(0)
            n_fb += 1
    y = np.asarray(y)
    pred = np.asarray(pred)
    m = f1_at(y, pred)
    m["n_fallback"] = n_fb
    m["n_gated"] = int(sum(gated_flag))
    # full-frame restricted to gated pids
    y_g, s_g = [], []
    for pid, (lab, sc) in ff.items():
        if pid in gated:
            y_g.append(lab)
            s_g.append(sc)
    y_g = np.asarray(y_g)
    pred_g = (np.asarray(s_g) >= thr_ff).astype(np.int64)
    m_ff_g = f1_at(y_g, pred_g)
    m["fullframe_on_gated"] = m_ff_g
    auc_g = float(roc_auc_score([gated[p][0] for p in gated], [gated[p][1] for p in gated]))
    m["gated_only_auc"] = auc_g
    return m


def pack(xs: list[float]) -> dict:
    xs = [x for x in xs if x == x]
    if len(xs) == 1:
        return {"mean": xs[0], "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)),
        "n": len(xs),
    }


def summarize(name: str, ff, thr_ff, runs: list[dict]) -> dict:
    rows = []
    for run in runs:
        m = eval_run(ff, thr_ff, run)
        m["seed"] = run["seed"]
        rows.append(m)
    return {
        "name": name,
        "hybrid_f1": pack([r["f1"] for r in rows]),
        "hybrid_recall": pack([r["recall"] for r in rows]),
        "hybrid_precision": pack([r["precision"] for r in rows]),
        "fullframe_on_gated_f1": pack([r["fullframe_on_gated"]["f1"] for r in rows]),
        "n_fallback": rows[0]["n_fallback"] if rows else None,
        "n": rows[0]["n"] if rows else None,
        "n_pos": rows[0]["n_pos"] if rows else None,
        "seeds": rows,
    }


def main() -> int:
    args = parse_args()
    ff, thr_ff = load_ff(args.fullframe_json)
    alvg = collect(args.alvg_root, args.alvg_glob)
    mil = collect(args.mil_root, args.mil_glob)
    if not mil:
        mil = collect(
            ROOT / "docs/paper/exp/_unpack_beat_mil/outputs/tertiary_20241125_vs_chd_fusion_raw",
            "attention_mil_seed*",
        )
    out = {
        "fullframe_n": len(ff),
        "fullframe_thr": thr_ff,
        "alvg": summarize("alvg", ff, thr_ff, alvg),
        "mil": summarize("mil", ff, thr_ff, mil),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "alvg" or True}, indent=2)[:2500])
    print("alvg hybrid", out["alvg"]["hybrid_f1"])
    print("alvg ff-on-gated", out["alvg"]["fullframe_on_gated_f1"])
    print("mil hybrid", out["mil"]["hybrid_f1"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
