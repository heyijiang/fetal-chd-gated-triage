#!/usr/bin/env python3
"""C2-b + coverage-aware multi-view decision (P0).

Clinical idea (your point):
  - Some findings: one clear view is enough → call abnormal
  - Others: need several views for confidence → else indeterminate

Does NOT change C2-b frame scoring (gate + M0‖plane-crop CLIP + shared LR).
Only changes *patient decision / reporting* using n_views + score.

Policies
--------
always          current: always call at F1-thr (baseline)
indeterminate_lt2   if n_views < 2 → IND (excluded from called metrics)
indeterminate_lt3   if n_views < 3 → IND
dual_gate       if score >= thr_hi → call even with 1 view (strong single-view)
                elif score >= thr_lo and n_views >= min_views → call
                elif n_views < min_views → IND
                else → negative
coverage_score  ranking score = patient_max * (n_views / 4)  (AUC sanity)

Also reports AUC/Sens/Spec stratified by n_views ∈ {1,2,3,4}.

Usage:
  cd experiments
  CUDA_VISIBLE_DEVICES=0 bash run_cardium_c2b_coverage.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from agcd.fetalclip_embed import CARDIUM_CROP_CACHE, build_or_load_embeddings  # noqa: E402
from cardium_dataset import load_fold_split  # noqa: E402
from chd_baseline.metrics import find_best_binary_threshold  # noqa: E402
from masvf_m0_screening import (  # noqa: E402
    CARDIUM_FEATURE_CACHE,
    apply_frame_selection,
    build_cardium_frame_rows,
    collect_crop_boxes,
    collect_embed_items,
    filter_view_mode,
    fit_lr,
    load_feature_cache,
    load_tags,
    predict_lr,
    rows_to_X,
    split_val_patients,
)

VIEWS = ("four_chamber", "lvot", "rvot", "vvt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2-b coverage-aware fusion")
    p.add_argument("--device", default="0")
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--cardium-processed", type=Path,
                   default=PROJECT_ROOT / "CARDIUM dataset" / "processed")
    p.add_argument("--image-tags", type=Path, default=None)
    p.add_argument("--feature-cache", type=Path, default=None)
    p.add_argument("--fetalclip-cache", type=Path, default=CARDIUM_CROP_CACHE)
    p.add_argument("--crop-pad", type=float, default=0.12)
    p.add_argument("--val-patient-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thr-hi-quantile", type=float, default=0.85,
                   help="LOCKED default 0.85: high thr = quantile of val abnormal scores")
    p.add_argument("--min-views-dual", type=int, default=3,
                   help="LOCKED default 3")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "cardium_c2b_coverage")
    return p.parse_args()


def resolve_tags(path: Path | None) -> Path:
    screen = PROJECT_ROOT / "data" / "study_screening"
    if path and path.is_file():
        return path
    for cand in (
        screen / "yolo_image_tags_cardium_anatomy.jsonl",
        screen / "yolo_image_tags_cardium_xyxy.jsonl",
        screen / "yolo_image_tags_cardium.jsonl",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("missing tags")


def _auc(y, s) -> float:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _clf_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(1 - y, 1 - pred, zero_division=0)),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }


def build_patients(rows, probs) -> list[dict]:
    by: dict[str, list] = defaultdict(list)
    for r, p in zip(rows, probs):
        by[str(r.patient_id)].append((r, float(p)))
    out = []
    for pid, items in by.items():
        best_r, best_p = max(items, key=lambda x: x[1])
        views = sorted({r.cardium_view for r, _ in items if r.cardium_view in VIEWS})
        out.append({
            "patient_id": pid,
            "label": int(best_r.label),
            "label_name": best_r.label_name,
            "fold": int(best_r.fold),
            "split": best_r.split,
            "score": best_p,
            "n_frames": len(items),
            "n_views": len(views),
            "views": views,
            "decisive_view": best_r.cardium_view,
        })
    return out


def strata_by_nviews(patients: list[dict]) -> dict:
    out = {}
    for nv in (1, 2, 3, 4):
        sub = [p for p in patients if p["n_views"] == nv]
        if not sub:
            out[str(nv)] = {"n": 0}
            continue
        y = [p["label"] for p in sub]
        s = [p["score"] for p in sub]
        out[str(nv)] = {
            "n": len(sub),
            "n_pos": int(sum(y)),
            "n_neg": int(len(y) - sum(y)),
            "auc": _auc(y, s),
            "mean_score_pos": float(np.mean([p["score"] for p in sub if p["label"] == 1])) if any(p["label"] == 1 for p in sub) else float("nan"),
            "mean_score_neg": float(np.mean([p["score"] for p in sub if p["label"] == 0])) if any(p["label"] == 0 for p in sub) else float("nan"),
        }
    return out


def decide_always(p: dict, thr: float) -> str:
    return "POS" if p["score"] >= thr else "NEG"


def decide_indeterminate_lt(p: dict, thr: float, min_views: int) -> str:
    if p["n_views"] < min_views:
        return "IND"
    return "POS" if p["score"] >= thr else "NEG"


def decide_dual_gate_legacy(p: dict, thr_lo: float, thr_hi: float, min_views: int) -> str:
    """v1: any score with n_views < min → IND (bloated IND on clear NEG)."""
    s, nv = p["score"], p["n_views"]
    if s >= thr_hi:
        return "POS"
    if nv < min_views:
        return "IND"
    if s >= thr_lo:
        return "POS"
    return "NEG"


def decide_dual_gate(p: dict, thr_lo: float, thr_hi: float, min_views: int) -> str:
    """v2 LOCKED: IND only in gray zone (thr_lo ≤ score < thr_hi).

    Clinical:
      - score ≥ thr_hi → POS (single-view enough)
      - score < thr_lo → NEG (clear negative; coverage irrelevant)
      - gray + n_views < min → IND (need more planes to commit)
      - gray + enough views → POS
    """
    s, nv = p["score"], p["n_views"]
    if s >= thr_hi:
        return "POS"
    if s < thr_lo:
        return "NEG"
    if nv < min_views:
        return "IND"
    return "POS"


def decide_dual_gate_soft_neg(p: dict, thr_lo: float, thr_hi: float, min_views: int) -> str:
    """v2.1 candidate: under-covered hard band → IND (not hard NEG).

    Same as v2 except:
      score < thr_lo AND n_views < min → IND
    Goal: convert coverage-linked misses into deferred cases without
    inventing positives.
    """
    s, nv = p["score"], p["n_views"]
    if s >= thr_hi:
        return "POS"
    if s < thr_lo:
        return "IND" if nv < min_views else "NEG"
    if nv < min_views:
        return "IND"
    return "POS"


def eval_policy(patients: list[dict], decisions: list[str]) -> dict:
    """Metrics on called patients; plus coverage / IND rates."""
    assert len(patients) == len(decisions)
    n = len(patients)
    n_ind = sum(1 for d in decisions if d == "IND")
    called_y, called_pred = [], []
    for p, d in zip(patients, decisions):
        if d == "IND":
            continue
        called_y.append(p["label"])
        called_pred.append(1 if d == "POS" else 0)

    # Among abnormals: how many caught / missed / deferred
    abn = [p for p in patients if p["label"] == 1]
    abn_pos = sum(1 for p, d in zip(patients, decisions) if p["label"] == 1 and d == "POS")
    abn_neg = sum(1 for p, d in zip(patients, decisions) if p["label"] == 1 and d == "NEG")
    abn_ind = sum(1 for p, d in zip(patients, decisions) if p["label"] == 1 and d == "IND")
    nor_pos = sum(1 for p, d in zip(patients, decisions) if p["label"] == 0 and d == "POS")

    m = _clf_metrics(called_y, called_pred) if called_y else {
        "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"),
        "sensitivity": float("nan"), "specificity": float("nan"), "n": 0,
        "n_pos": 0, "n_neg": 0,
    }
    # ranking on all (for dual uses same score); IND does not change AUC of raw score
    auc_all = _auc([p["label"] for p in patients], [p["score"] for p in patients])
    return {
        **m,
        "auc_score_all": auc_all,
        "n_total": n,
        "n_indeterminate": n_ind,
        "indeterminate_rate": n_ind / max(n, 1),
        "coverage_call_rate": (n - n_ind) / max(n, 1),
        "abnormal_POS": abn_pos,
        "abnormal_NEG": abn_neg,
        "abnormal_IND": abn_ind,
        "abnormal_recall_called_only": abn_pos / max(len(abn) - abn_ind, 1),
        "abnormal_capture_including_defer": (abn_pos) / max(len(abn), 1),
        "abnormal_missed_hard_neg": abn_neg / max(len(abn), 1),
        "normal_FP": nor_pos,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    tags_path = resolve_tags(args.image_tags)
    feat = args.feature_cache or CARDIUM_FEATURE_CACHE
    if not Path(feat).is_file():
        legacy = PROJECT_ROOT / "data" / "study_screening" / "cardium_det_feature_cache.jsonl"
        if legacy.is_file():
            feat = legacy

    tags = load_tags(tags_path)
    cache = load_feature_cache(feat)
    print(f"tags={tags_path}")
    print("Policy note: single-view strong score → call; weak coverage mid-score → IND")

    ns = argparse.Namespace(
        device=args.device, fetalclip_device=args.device, batch_size=32,
        yolo_conf=0.25, yolo_imgsz=640, min_plane_conf=0.15,
        frame_select="fetalclip_diverse_gate", k_per_view=0, view_mode="4view",
        model="m1", clip_crop="plane", crop_pad=args.crop_pad,
        rebuild_cache=False, rebuild_fetalclip_cache=False, cohort="cardium",
        folds=args.folds, cardium_processed=args.cardium_processed,
        feature_cache=feat, fetalclip_cache=args.fetalclip_cache,
        image_tags=tags_path, val_patient_ratio=args.val_patient_ratio,
        seed=args.seed, yolo_weights=None, max_studies=0,
    )

    print("=== Build gated C2-b frames ===")
    all_records = []
    for fold in folds:
        for sp in ("train", "test"):
            all_records.extend(load_fold_split(str(fold), sp, args.cardium_processed))
    all_rows = build_cardium_frame_rows(
        all_records, tags=tags, yolo_model=None, args=ns, cache=cache,
    )
    all_rows = filter_view_mode(all_rows, "4view")
    items = collect_embed_items(ns)
    crop_boxes = collect_crop_boxes(tags, items)
    embed_cache = build_or_load_embeddings(
        items, cache_path=args.fetalclip_cache, device=str(args.device),
        batch_size=32, crop_boxes=crop_boxes, crop_pad=args.crop_pad,
    )
    all_rows = apply_frame_selection(all_rows, tags, ns, embed_cache)
    print(f"  gated={len(all_rows)}")

    fold_out = {}
    pooled_test: list[dict] = []
    # accumulate policy metrics across folds (micro on called + rates)
    policy_names = (
        "always",
        "indeterminate_lt2",
        "indeterminate_lt3",
        "dual_gate",
    )
    pooled_decisions: dict[str, list[tuple[dict, str]]] = {k: [] for k in policy_names}

    for fold in folds:
        print(f"\n=== Fold {fold} ===")
        fold_rows = [r for r in all_rows if r.fold == fold]
        train_rows = [r for r in fold_rows if r.split == "train"]
        test_rows = [r for r in fold_rows if r.split == "test"]
        fit_rows, val_rows = split_val_patients(
            train_rows, args.val_patient_ratio, args.seed + fold, lambda r: r.patient_id,
        )
        clf, fill = fit_lr(
            rows_to_X(fit_rows, model="m1", embed_cache=embed_cache),
            np.array([r.label for r in fit_rows], dtype=np.int64),
        )
        score_rows = fit_rows + val_rows + test_rows
        probs = predict_lr(
            clf, fill, rows_to_X(score_rows, model="m1", embed_cache=embed_cache),
        )
        pmap = {r.sample_id: float(p) for r, p in zip(score_rows, probs)}

        def patients_of(rows):
            pp = np.array([pmap[r.sample_id] for r in rows])
            return build_patients(rows, pp)

        va = patients_of(val_rows)
        te = patients_of(test_rows)
        thr_lo, _ = find_best_binary_threshold(
            np.array([p["label"] for p in va]),
            np.array([p["score"] for p in va]),
        )
        # thr_hi: quantile of val *abnormal* scores (strong single-view bar)
        abn_va = [p["score"] for p in va if p["label"] == 1]
        if abn_va:
            thr_hi = float(np.quantile(abn_va, args.thr_hi_quantile))
        else:
            thr_hi = max(thr_lo, 0.95)
        thr_hi = max(thr_hi, thr_lo)  # ensure hi >= lo

        strata = strata_by_nviews(te)
        print("  strata n_views → auc (test):")
        for nv, st in strata.items():
            if st.get("n", 0) == 0:
                continue
            print(f"    views={nv}: n={st['n']} pos={st['n_pos']} AUC={st['auc']:.3f}")

        policies = {}
        for name in policy_names:
            decs = []
            for p in te:
                if name == "always":
                    d = decide_always(p, thr_lo)
                elif name == "indeterminate_lt2":
                    d = decide_indeterminate_lt(p, thr_lo, 2)
                elif name == "indeterminate_lt3":
                    d = decide_indeterminate_lt(p, thr_lo, 3)
                else:
                    d = decide_dual_gate(p, thr_lo, thr_hi, args.min_views_dual)
                decs.append(d)
                pooled_decisions[name].append((p, d))
            metrics = eval_policy(te, decs)
            policies[name] = {
                "metrics": metrics,
                "thr_lo": thr_lo,
                "thr_hi": thr_hi if name == "dual_gate" else None,
            }
            n_abn = metrics["abnormal_POS"] + metrics["abnormal_NEG"] + metrics["abnormal_IND"]
            print(
                f"  {name:20s} called_F1={metrics['f1']:.3f} "
                f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f} "
                f"IND={metrics['indeterminate_rate']:.2%} "
                f"abn_miss={metrics['abnormal_missed_hard_neg']:.2%} "
                f"abn_IND={metrics['abnormal_IND']}/{n_abn}"
            )

        # coverage-weighted score AUC (ranking probe)
        y = [p["label"] for p in te]
        s_cov = [p["score"] * (p["n_views"] / 4.0) for p in te]
        cov_auc = _auc(y, s_cov)

        fold_out[str(fold)] = {
            "thr_lo": thr_lo,
            "thr_hi": thr_hi,
            "strata_by_n_views": strata,
            "coverage_weighted_auc": cov_auc,
            "policies": policies,
        }
        pooled_test.extend(te)

    # pooled strata + policies
    pooled_strata = strata_by_nviews(pooled_test)
    pooled_policies = {}
    for name, pairs in pooled_decisions.items():
        pats = [p for p, _ in pairs]
        decs = [d for _, d in pairs]
        pooled_policies[name] = eval_policy(pats, decs)

    print("\n========== POOLED strata (n_views) ==========")
    for nv, st in pooled_strata.items():
        if st.get("n", 0) == 0:
            continue
        print(f"  views={nv}: n={st['n']:4d} pos={st['n_pos']:3d} AUC={st['auc']:.3f}")

    print("\n========== POOLED policies ==========")
    for name in policy_names:
        m = pooled_policies[name]
        print(
            f"{name:20s} F1={m['f1']:.3f} Sens={m['sensitivity']:.3f} Spec={m['specificity']:.3f} "
            f"IND%={m['indeterminate_rate']:.1%} call%={m['coverage_call_rate']:.1%} "
            f"abn_miss%={m['abnormal_missed_hard_neg']:.1%} abn_IND={m['abnormal_IND']} FP={m['normal_FP']}"
        )

    out = {
        "idea": {
            "single_view_enough": "score >= thr_hi → POS even if n_views==1",
            "need_multi_view": "thr_lo <= score < thr_hi and n_views < min → IND (defer)",
            "always_max": "C2-b patient score unchanged; policy only affects call/IND",
        },
        "folds": fold_out,
        "pooled_strata_by_n_views": pooled_strata,
        "pooled_policies": pooled_policies,
        "pooled_auc_raw": _auc([p["label"] for p in pooled_test], [p["score"] for p in pooled_test]),
        "pooled_auc_coverage_weighted": _auc(
            [p["label"] for p in pooled_test],
            [p["score"] * (p["n_views"] / 4.0) for p in pooled_test],
        ),
    }
    path = args.output_dir / "coverage_fusion_results.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    print("Read: dual_gate should cut hard FN (abn_miss) by deferring low-coverage mid scores,")
    print("      while keeping strong single-view POS; Spec/FP trade via IND rate.")


if __name__ == "__main__":
    main()
