#!/usr/bin/env python3
"""Full-frame FetalCLIP linear probe (NO YOLO / NO plane crop).

CARDIUM-style baseline on any patient-level manifest:
  - EVERY frame inherits the patient label (abnormal patient → all frames=1)
  - WHOLE image (no YOLO gate, no plane crop)
  - frozen FetalCLIP (+ optional homologous LoRA) → frame LR
  - patient score = mean (and max) over all frames

Usage:
  # private
  CUDA_VISIBLE_DEVICES=0 python -u experiments/private_fetalclip_fullframe_linear.py --device 0

  # homologous_real (recommended wrapper)
  bash experiments/run_homologous_real_fullframe_cardium_mean.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agcd.fetalclip_embed import (  # noqa: E402
    FETALCLIP_EMBED_DIM,
    PRIVATE_CACHE,
    build_or_load_embeddings,
    lookup_embedding,
)
from chd_baseline.metrics import (  # noqa: E402
    binary_metrics_at_threshold,
    find_best_binary_threshold,
)

DEFAULT_LORA = PROJECT_ROOT / "outputs" / "homologous_midlate_fetalclip_lora" / "best_adapter.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-frame FetalCLIP linear probe (no filtering)")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/study_screening/manifest.jsonl")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--cache", type=Path, default=PRIVATE_CACHE,
                   help="Full-image FetalCLIP embedding cache")
    p.add_argument("--lora-adapter", type=Path, default=None,
                   help="Optional homologous LoRA adapter (CARDIUM-style domain adapt)")
    p.add_argument("--device", default="0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--C", type=float, default=1.0, help="LR inverse regularization")
    p.add_argument(
        "--fit-level", choices=("frame", "patient_mean"), default="frame",
        help="Fit LR on frames or one mean embedding per patient.",
    )
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--canon-size", type=int, default=0)
    p.add_argument("--canon-jpeg-quality", type=int, default=0)
    p.add_argument("--intensity-normalize", action="store_true")
    p.add_argument("--fixed-crop-frac", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-studies", type=int, default=0, help="Debug cap")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--embed-load-only", action="store_true",
                   help="Require existing --cache; never call FetalCLIP")
    p.add_argument(
        "--embed-only",
        action="store_true",
        help="Encode embeddings into --cache and exit without training LR",
    )
    p.add_argument(
        "--verify-exists",
        action="store_true",
        help="Per-frame is_file() during manifest load (slow on large NFS corpora)",
    )
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/private_fetalclip_fullframe_linear")
    return p.parse_args()


def resolve_frame_path(data_root: Path, rel: str) -> Path:
    """Prefer absolute paths as-is; avoid resolve()/is_file on every frame."""
    raw = Path(rel)
    if raw.is_absolute():
        return raw
    return data_root / raw


def load_frames(
    manifest: Path,
    data_root: Path,
    max_studies: int,
    *,
    verify_exists: bool = False,
) -> list[dict]:
    """Load frame rows from a patient manifest.

    Existence checks are off by default: on large tertiary corpora (~5e5 frames)
    per-frame is_file()/resolve() can stall for tens of minutes on NFS.
    """
    rows: list[dict] = []
    n_study = 0
    n_skip = 0
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            n_study += 1
            if max_studies and n_study > max_studies:
                break
            if n_study == 1 or n_study % 200 == 0:
                print(
                    f"  loading patients={n_study} frames={len(rows)} "
                    f"(skip_missing={n_skip})",
                    flush=True,
                )
            pid = str(s["patient_id"])
            label = int(s.get("label_binary", 0))
            split = s.get("split", "")
            disease = s.get("disease_folder") or s.get("label_disease_cn") or (
                "正常" if label == 0 else "未知"
            )
            for rel in s.get("frame_paths") or []:
                p = resolve_frame_path(data_root, str(rel))
                if verify_exists and not p.is_file():
                    n_skip += 1
                    continue
                rows.append({
                    "sample_id": f"{s['study_id']}|{rel}",
                    "path": p,
                    "patient_id": pid,
                    "label": label,
                    "split": split,
                    "disease": disease,
                })
    print(
        f"  loaded patients={n_study} frames={len(rows)} skip_missing={n_skip}",
        flush=True,
    )
    return rows


def to_matrix(rows: list[dict], emb: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X, y, pids, diseases = [], [], [], []
    miss = 0
    for r in rows:
        e = lookup_embedding(emb, sample_id=r["sample_id"], image_path=str(r["path"]))
        if e is None:
            miss += 1
            continue
        X.append(np.asarray(e, dtype=np.float64))
        y.append(r["label"])
        pids.append(r["patient_id"])
        diseases.append(r["disease"])
    if miss:
        print(f"  WARN: {miss}/{len(rows)} frames missing embedding (skipped)")
    if not X:
        return np.empty((0, FETALCLIP_EMBED_DIM)), np.array([]), [], []
    return np.stack(X), np.array(y, dtype=np.int64), pids, diseases


def patient_scores(y, prob, pids, agg: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for yt, pr, pid in zip(y, prob, pids):
        groups[str(pid)].append((int(yt), float(pr)))
    yy, pp, ids = [], [], []
    for pid in sorted(groups):
        vals = groups[pid]
        yy.append(vals[0][0])
        probs = [v[1] for v in vals]
        pp.append(max(probs) if agg == "max" else float(np.mean(probs)))
        ids.append(pid)
    return np.asarray(yy, dtype=np.int64), np.asarray(pp, dtype=np.float64), ids


def patient_auc(y, prob, pids, agg: str) -> tuple[float | None, int]:
    yy, pp, _ = patient_scores(y, prob, pids, agg)
    if len(set(yy.tolist())) < 2:
        return None, int(len(yy))
    return float(roc_auc_score(yy, pp)), int(len(yy))


def patient_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    pids: list[str],
    diseases: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pid in enumerate(pids):
        groups[str(pid)].append(index)
    features, labels, ids, patient_diseases = [], [], [], []
    for pid in sorted(groups):
        indices = groups[pid]
        patient_labels = {int(y[index]) for index in indices}
        if len(patient_labels) != 1:
            raise ValueError(f"patient {pid} has inconsistent labels: {patient_labels}")
        features.append(X[indices].mean(axis=0))
        labels.append(patient_labels.pop())
        ids.append(pid)
        patient_diseases.append(diseases[indices[0]])
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        ids,
        patient_diseases,
    )


def _metrics_block(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    m = binary_metrics_at_threshold(y, s, thr)
    auc = float(roc_auc_score(y, s)) if len(set(y.tolist())) >= 2 else None
    return {
        "auc": auc,
        "threshold": float(thr),
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()},
    }


def per_disease_patient_auc(y, prob, pids, diseases, agg="max") -> dict:
    # normal patient scores
    norm_scores: dict[str, list[float]] = defaultdict(list)
    dis_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for yt, pr, pid, dis in zip(y, prob, pids, diseases):
        if yt == 0:
            norm_scores[pid].append(float(pr))
        else:
            dis_scores[dis][pid].append(float(pr))

    def agg_scores(d: dict[str, list[float]]) -> list[float]:
        out = []
        for pid, vals in d.items():
            out.append(max(vals) if agg == "max" else float(np.mean(vals)))
        return out

    norm_vec = agg_scores(norm_scores)
    out = {}
    for dis, pat in sorted(dis_scores.items()):
        dis_vec = agg_scores(pat)
        yy = [0] * len(norm_vec) + [1] * len(dis_vec)
        ss = norm_vec + dis_vec
        if len(set(yy)) < 2 or len(dis_vec) < 3:
            out[dis] = {"patient_auc": None, "n_disease_patients": len(dis_vec)}
            continue
        out[dis] = {
            "patient_auc": float(roc_auc_score(yy, ss)),
            "n_disease_patients": len(dis_vec),
        }
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora = args.lora_adapter
    if lora is not None and not Path(lora).is_file():
        print(f"ERROR: LoRA adapter not found: {lora}")
        return 1

    print(f"Loading manifest: {args.manifest}", flush=True)
    rows = load_frames(
        args.manifest,
        args.data_root,
        args.max_studies,
        verify_exists=args.verify_exists,
    )
    n_by_split = defaultdict(int)
    for r in rows:
        n_by_split[r["split"]] += 1
    print(f"frames (no filtering): {len(rows)}  by split={dict(n_by_split)}")
    print("  label rule: every frame inherits patient label (skip YOLO)")

    items_all = [(r["sample_id"], r["path"]) for r in rows]
    print(f"Building/loading FULL-IMAGE FetalCLIP embeddings (cache={args.cache})")
    if lora:
        print(f"  LoRA adapter: {lora}")
    if args.embed_load_only and not args.cache.is_file():
        print(f"ERROR: --embed-load-only but missing cache {args.cache}")
        return 1
    if args.embed_load_only:
        from agcd.fetalclip_embed import load_embed_cache
        emb = load_embed_cache(args.cache)
        print(f"  loaded cache only: {len(emb)} keys")
    else:
        emb = build_or_load_embeddings(
            items_all,
            cache_path=args.cache,
            device=args.device,
            batch_size=args.batch_size,
            rebuild=args.rebuild_cache,
            crop_boxes=None,  # whole image, no plane crop
            grayscale=args.grayscale,
            canon_size=args.canon_size,
            canon_jpeg_quality=args.canon_jpeg_quality,
            intensity_normalize=args.intensity_normalize,
            fixed_crop_frac=args.fixed_crop_frac,
            lora_adapter=lora,
        )

    if args.embed_only:
        have = sum(
            1
            for r in rows
            if lookup_embedding(emb, sample_id=r["sample_id"], image_path=str(r["path"])) is not None
        )
        print(f"embed-only done: {have}/{len(rows)} frames in {args.cache}")
        return 0 if have == len(rows) else 3

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]

    Xtr, ytr, pid_tr, dis_tr = to_matrix(train, emb)
    Xva, yva, pid_va, dis_va = to_matrix(val, emb)
    Xte, yte, pid_te, dis_te = to_matrix(test, emb)
    print(f"train={len(ytr)} val={len(yva)} test={len(yte)}")

    if not len(ytr) or not len(yva) or not len(yte):
        print("ERROR: empty split; check cache/paths")
        return 1

    probe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", C=args.C, random_state=args.seed)),
    ])
    if args.fit_level == "patient_mean":
        Xtr_fit, ytr_fit, _, _ = patient_feature_matrix(
            Xtr, ytr, pid_tr, dis_tr
        )
    else:
        Xtr_fit, ytr_fit = Xtr, ytr
    probe.fit(Xtr_fit, ytr_fit)

    # Image-level threshold on val
    prob_va = probe.predict_proba(Xva)[:, 1]
    thr_img, _ = find_best_binary_threshold(yva, prob_va)
    # Patient-mean threshold on val (primary CARDIUM-style readout)
    yva_p, sva_p, _ = patient_scores(yva, prob_va, pid_va, "mean")
    if args.fit_level == "patient_mean":
        Xva_p, yva_p, _, _ = patient_feature_matrix(
            Xva, yva, pid_va, dis_va
        )
        sva_p = probe.predict_proba(Xva_p)[:, 1]
    thr_pat, _ = find_best_binary_threshold(yva_p, sva_p)

    prob_te = probe.predict_proba(Xte)[:, 1]
    image_block = _metrics_block(yte, prob_te, thr_img)

    y_max, s_max, _ = patient_scores(yte, prob_te, pid_te, "max")
    y_mean, s_mean, pids_mean = patient_scores(yte, prob_te, pid_te, "mean")
    if args.fit_level == "patient_mean":
        Xte_p, y_mean, pids_mean, _ = patient_feature_matrix(
            Xte, yte, pid_te, dis_te
        )
        s_mean = probe.predict_proba(Xte_p)[:, 1]
    pmax_block = _metrics_block(y_max, s_max, thr_pat)
    pmean_block = _metrics_block(y_mean, s_mean, thr_pat)
    by_disease = per_disease_patient_auc(yte, prob_te, pid_te, dis_te, agg="mean")

    report = {
        "protocol": {
            "filtering": "NONE (all frames, no YOLO gate, no frame selection)",
            "image": "whole image (no plane crop)",
            "label": "every frame inherits patient label",
            "model": "frozen FetalCLIP"
            + ("+homologous-LoRA" if lora else "")
            + " + logistic regression",
            "aggregation": "patient-mean (primary), patient-max",
            "fit": f"manifest train at {args.fit_level} level",
            "input_transform": {
                "grayscale": args.grayscale,
                "canon_size": args.canon_size,
                "canon_jpeg_quality": args.canon_jpeg_quality,
                "intensity_normalize": args.intensity_normalize,
                "fixed_crop_frac": args.fixed_crop_frac,
            },
            "threshold": "val-tuned on patient-mean scores",
            "eval": "manifest test",
            "C": args.C,
            "lora_adapter": str(lora) if lora else None,
            "manifest": str(args.manifest),
            "cache": str(args.cache),
        },
        "counts": {
            "frames_total": len(rows),
            "frames_by_split": dict(n_by_split),
            "train_frames_with_emb": int(len(ytr)),
            "val_frames_with_emb": int(len(yva)),
            "test_frames_with_emb": int(len(yte)),
            "test_patients": int(len(y_mean)),
        },
        "threshold_image_val_tuned": float(thr_img),
        "threshold_patient_mean_val_tuned": float(thr_pat),
        "test_image": image_block,
        "test_patient_max": pmax_block,
        "test_patient_mean": pmean_block,
        # legacy keys
        "test_image_auc": image_block.get("auc"),
        "test_patient_max_auc": pmax_block.get("auc"),
        "test_patient_mean_auc": pmean_block.get("auc"),
        "test_per_disease_patient_mean_auc": by_disease,
        "test_patient_mean_scores": [
            {"patient_id": pid, "label": int(y), "score": float(s)}
            for pid, y, s in zip(pids_mean, y_mean.tolist(), s_mean.tolist())
        ],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def _row(name: str, b: dict) -> str:
        auc = b.get("auc")
        f1 = b.get("f1")
        sens = b.get("sensitivity") or b.get("recall")
        spec = b.get("specificity")
        a = f"{auc:.4f}" if auc is not None else "n/a"
        f = f"{f1:.4f}" if f1 is not None else "n/a"
        se = f"{sens:.4f}" if sens is not None else "n/a"
        sp = f"{spec:.4f}" if spec is not None else "n/a"
        return f"| {name} | {a} | {f} | {se} | {sp} | {b.get('n_pos')}/{b.get('n_neg')} |"

    lines = [
        "# Full-frame FetalCLIP linear (CARDIUM-style, no YOLO)",
        "",
        "- filtering: **none** (all frames), image: **whole** (no crop)",
        f"- model: FetalCLIP{'+LoRA' if lora else ''} + LR → **patient-mean**",
        f"- manifest: `{args.manifest.name}`",
        f"- frames: {len(rows)}  test patients: {len(y_mean)}",
        "",
        "| level | AUC | F1 | Sens | Spec | n_pos/n_neg |",
        "|---|---:|---:|---:|---:|---:|",
        _row("test image", image_block),
        _row("test patient-max", pmax_block),
        _row("test patient-mean ★", pmean_block),
        "",
        "## Per-disease patient-mean AUC (disease vs normal, test)",
        "",
        "| disease | n_patients | patient-mean AUC |",
        "|---|---:|---:|",
    ]
    for dis, r in sorted(by_disease.items(), key=lambda kv: -(kv[1]["patient_auc"] or 0)):
        a = r["patient_auc"]
        lines.append(
            f"| {dis} | {r['n_disease_patients']} | {a:.4f} |"
            if a is not None
            else f"| {dis} | {r['n_disease_patients']} | n/a |"
        )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== Full-frame FetalCLIP (NO YOLO) · CARDIUM-style patient-mean ===")
    print(f"  test image AUC          : {image_block.get('auc')}")
    print(f"  test patient-max AUC    : {pmax_block.get('auc')}  F1={pmax_block.get('f1')}")
    print(f"  test patient-mean AUC ★ : {pmean_block.get('auc')}  F1={pmean_block.get('f1')}")
    print(f"  thr_patient_mean (val)  : {thr_pat:.4f}")
    print("\n  per-disease patient-mean AUC:")
    for dis, r in sorted(by_disease.items(), key=lambda kv: -(kv[1]["patient_auc"] or 0)):
        a = r["patient_auc"]
        print(
            f"    {dis:<16} n={r['n_disease_patients']:4d}  auc={a:.4f}"
            if a is not None
            else f"    {dis:<16} n={r['n_disease_patients']:4d}  auc=n/a"
        )
    print(f"\nWrote {args.output_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
