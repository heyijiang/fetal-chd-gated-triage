#!/usr/bin/env python3
"""Unified test-set metrics on the same manifest patients (review 🔴1).

Combines:
  - full-frame patient-mean scores (existing seed results.json)
  - fusion unified re-runs (--include-empty-view-patients, pids_test in JSON)
  - hybrid fallback: fusion score if gated, else full-frame

CPU-only aggregation after GPU unified fusion jobs finish.

  python -u experiments/eval_tertiary_unified_testset.py
  python -u experiments/eval_tertiary_unified_testset.py \\
    --unified-fusion-root outputs/tertiary_20241125_vs_chd_unified_eval
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "val_f1_tuned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified manifest test metrics")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument(
        "--audit-json",
        type=Path,
        default=ROOT / "outputs/tertiary_unified_audit/coverage.json",
    )
    p.add_argument(
        "--fullframe-root",
        type=Path,
        default=ROOT / "outputs/tertiary_20241125_vs_chd_patient_full_raw_seeds",
    )
    p.add_argument(
        "--fullframe-seed",
        type=int,
        default=42,
        help="LR probe is seed-invariant; any seed is fine",
    )
    p.add_argument(
        "--unified-fusion-root",
        type=Path,
        default=ROOT / "outputs/tertiary_20241125_vs_chd_unified_eval",
    )
    p.add_argument(
        "--fusion-globs",
        default=(
            "attention_mil_seed42,attention_mil_seed43,attention_mil_seed44,"
            "feature_transformer_seed42,graph_transformer_seed42"
        ),
        help="Comma-separated run dirs under --unified-fusion-root",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_unified_eval",
    )
    return p.parse_args()


def load_manifest_test(manifest: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("split") == "test":
                out[str(s["patient_id"])] = int(s.get("label_binary", 0))
    return out


def load_fullframe_scores(path: Path) -> dict[str, tuple[int, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("test_patient_mean_scores") or []
    return {str(r["patient_id"]): (int(r["label"]), float(r["score"])) for r in rows}


def find_fusion_json(run_dir: Path) -> Path | None:
    hits = sorted(run_dir.glob("view_token_results_*.json"))
    return hits[0] if hits else None


def load_fusion_unified(path: Path, mode: str = PRIMARY) -> dict[str, tuple[int, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fold = (data.get("folds") or [{}])[0]
    ps = fold.get("patient_scores") or {}
    pids = ps.get("pids_test")
    block = (ps.get("fusion") or {})
    y = block.get("y_test")
    p = block.get("p_test")
    if not pids or not y or not p:
        return {}
    return {
        str(pid): (int(lab), float(sc))
        for pid, lab, sc in zip(pids, y, p)
    }


def tune_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    if len(y_val) < 2 or len(set(y_val.tolist())) < 2:
        return 0.5
    order = np.argsort(-s_val)
    best_thr, best_f1 = 0.5, -1.0
    for i in order:
        thr = float(s_val[i])
        pred = (s_val >= thr).astype(np.int64)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr


def metrics_at_thr(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) >= 2 else float("nan")
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auc": auc,
        "f1": f1,
        "sensitivity": rec,
        "specificity": spec,
        "ppv": prec,
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def build_vector(
    manifest: dict[str, int],
    scores: dict[str, tuple[int, float]],
    *,
    impute_fn=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pids = sorted(manifest)
    y, s = [], []
    for pid in pids:
        if pid in scores:
            lab, sc = scores[pid]
            y.append(lab)
            s.append(sc)
        elif impute_fn is not None:
            lab = manifest[pid]
            y.append(lab)
            s.append(float(impute_fn(pid, lab)))
        else:
            continue
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64), pids


def hybrid_scores(
    manifest: dict[str, int],
    fusion: dict[str, tuple[int, float]],
    fullframe: dict[str, tuple[int, float]],
    gated_pids: set[str],
) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    for pid, lab in manifest.items():
        if pid in fusion and pid in gated_pids:
            out[pid] = fusion[pid]
        elif pid in fullframe:
            out[pid] = fullframe[pid]
        elif pid in fusion:
            out[pid] = fusion[pid]
    return out


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
    if s is None:
        return f"{m:.4f}"
    return f"{m:.4f}±{s:.4f}"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_test(args.manifest)
    ff_path = args.fullframe_root / f"seed{args.fullframe_seed}" / "results.json"
    if not ff_path.is_file():
        print(f"ERROR: missing fullframe {ff_path}")
        return 1
    fullframe = load_fullframe_scores(ff_path)

    gated_pids: set[str] = set()
    if args.audit_json.is_file():
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        gated_pids = {
            pid
            for pid, c in (audit.get("patients") or {}).items()
            if c.get("fusion_gated")
        }

    # --- manifest-complete with impute unscored as 0 (conservative triage) ---
    y_m, s_m, _ = build_vector(
        manifest,
        fullframe,
        impute_fn=lambda _pid, _lab: 0.0,
    )
    thr_ff = tune_threshold(y_m, s_m)  # placeholder; use val from JSON if needed
    thr_ff = float(
        json.loads(ff_path.read_text(encoding="utf-8")).get("threshold_patient_mean_val_tuned", thr_ff)
    )
    m_manifest_ff = metrics_at_thr(y_m, s_m, thr_ff)

    y_ep, s_ep, _ = build_vector(manifest, fullframe)  # embed-present only
    m_embed_ff = metrics_at_thr(y_ep, s_ep, thr_ff)

    fusion_runs: list[dict] = []
    for name in [x.strip() for x in args.fusion_globs.split(",") if x.strip()]:
        run_dir = args.unified_fusion_root / name
        jpath = find_fusion_json(run_dir)
        if jpath is None:
            print(f"WARN: no fusion JSON in {run_dir}")
            continue
        scores = load_fusion_unified(jpath)
        if not scores:
            print(f"WARN: no pids_test in {jpath}")
            continue
        y_u, s_u, _ = build_vector(manifest, scores)
        # threshold from val in same JSON
        fold = json.loads(jpath.read_text(encoding="utf-8")).get("folds", [{}])[0]
        ps = fold.get("patient_scores", {})
        yv = np.asarray(ps.get("fusion", {}).get("y_val", []), dtype=np.int64)
        sv = np.asarray(ps.get("fusion", {}).get("p_val", []), dtype=np.float64)
        thr = tune_threshold(yv, sv) if len(yv) else 0.5
        mode_block = (fold.get("fusion") or {}).get(PRIMARY) or {}
        if mode_block.get("threshold") is not None:
            thr = float(mode_block["threshold"])
        fusion_runs.append({
            "run": name,
            "manifest_unified": metrics_at_thr(y_u, s_u, thr),
            "json": str(jpath),
        })

    first_fusion_scores: dict[str, tuple[int, float]] | None = None
    for name in [x.strip() for x in args.fusion_globs.split(",") if x.strip()]:
        run_dir = args.unified_fusion_root / name
        jpath = find_fusion_json(run_dir)
        if jpath:
            first_fusion_scores = load_fusion_unified(jpath)
            break
    hybrid_block = None
    if first_fusion_scores:
        hybrid_map = hybrid_scores(manifest, first_fusion_scores, fullframe, gated_pids)
        y_h, s_h, _ = build_vector(manifest, hybrid_map)
        thr_h = thr_ff
        hybrid_block = metrics_at_thr(y_h, s_h, thr_h)

    out = {
        "manifest_test_n": len(manifest),
        "manifest_pos": sum(manifest.values()),
        "fullframe": {
            "manifest_complete_impute0": m_manifest_ff,
            "embed_present": m_embed_ff,
            "threshold": thr_ff,
        },
        "fusion_unified_runs": fusion_runs,
        "hybrid_fusion_or_fullframe": hybrid_block,
    }
    (args.out_dir / "unified_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Unified test-set metrics",
        "",
        f"Manifest test: **{len(manifest)}** patients "
        f"({sum(manifest.values())} CHD + {len(manifest) - sum(manifest.values())} normal)",
        "",
        "## Full-frame patient-mean",
        "",
        "| protocol | n_pos/n_neg | AUC | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|",
        f"| manifest-complete (unscored→0) | "
        f"{m_manifest_ff['n_pos']}/{m_manifest_ff['n_neg']} | "
        f"{m_manifest_ff['auc']:.4f} | {m_manifest_ff['f1']:.4f} | "
        f"{m_manifest_ff['sensitivity']:.4f} | {m_manifest_ff['specificity']:.4f} |",
        f"| embed-present only | "
        f"{m_embed_ff['n_pos']}/{m_embed_ff['n_neg']} | "
        f"{m_embed_ff['auc']:.4f} | {m_embed_ff['f1']:.4f} | "
        f"{m_embed_ff['sensitivity']:.4f} | {m_embed_ff['specificity']:.4f} |",
        "",
        "## Fusion (unified manifest, --include-empty-view-patients)",
        "",
        "| run | n_pos/n_neg | AUC | F1 | Sens | Spec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fusion_runs:
        m = row["manifest_unified"]
        lines.append(
            f"| `{row['run']}` | {m['n_pos']}/{m['n_neg']} | "
            f"{m['auc']:.4f} | {m['f1']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |"
        )
    if hybrid_block:
        lines.extend([
            "",
            "## Hybrid (gated fusion else full-frame)",
            "",
            f"- n_pos/n_neg: {hybrid_block['n_pos']}/{hybrid_block['n_neg']}",
            f"- AUC/F1/Sens/Spec: {hybrid_block['auc']:.4f} / {hybrid_block['f1']:.4f} / "
            f"{hybrid_block['sensitivity']:.4f} / {hybrid_block['specificity']:.4f}",
        ])
    (args.out_dir / "AGGREGATE_UNIFIED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(out["fullframe"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.out_dir / 'AGGREGATE_UNIFIED.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
