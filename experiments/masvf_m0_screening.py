#!/usr/bin/env python3
"""MASVF M0/M1 screening: plane stats + optional FetalCLIP concat + LR.

M0: 24-d plane-conditioned stats/rule features only.
M1: M0 + frozen FetalCLIP 768-d embedding (early concat at frame level).

Default:
  model        = m0
  frame_select = fetalclip_diverse_gate  (usable frames only, no top-K cap)
  patient_agg  = max

Usage:
  cd experiments
  bash run_masvf_m0_private.sh
  MODEL=m1 bash run_masvf_m0_private.sh
  bash run_masvf.sh   # M0 + M1 together
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from agcd.cardium_views import CARDIUM_STANDARD_VIEWS, dominant_plane_from_dets, map_to_cardium_view  # noqa: E402
from agcd.fetalclip_embed import DEFAULT_CACHE as FETALCLIP_CACHE  # noqa: E402
from agcd.fetalclip_embed import PRIVATE_CACHE as PRIVATE_FETALCLIP_CACHE  # noqa: E402
from agcd.fetalclip_embed import CARDIUM_CROP_CACHE, PRIVATE_CROP_CACHE  # noqa: E402
from agcd.fetalclip_embed import DEFAULT_CROP_PAD, FETALCLIP_EMBED_DIM  # noqa: E402
from agcd.fetalclip_embed import build_or_load_embeddings, lookup_embedding  # noqa: E402
from agcd.frame_select import select_frames  # noqa: E402
from agcd.plane_features import (  # noqa: E402
    M0_FEATURE_NAMES,
    M0_FEATURE_SUBSETS,
    detected_ids_from_tag,
    m0_feat_vector,
    resolve_m0_feature_names,
    select_m0_columns,
)
from cardium_dataset import CardiumRecord, load_fold_split  # noqa: E402
from chd_baseline.chamber_ratios import extract_chamber_features, impute_features  # noqa: E402
from chd_baseline.metrics import (  # noqa: E402
    aggregate_patient_max_confidence,
    binary_metrics_at_threshold,
    find_best_binary_threshold,
)
from detect_rule_binary_screening import yolo_to_anatomy_boxes  # noqa: E402
from yolo_io import _make_logistic_regression, load_yolo_model, resolve_weights  # noqa: E402

T = TypeVar("T")

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "study_screening" / "manifest.jsonl"
DEFAULT_PRIVATE_TAGS = PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_private.jsonl"
DEFAULT_CARDIUM_TAGS = PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_cardium.jsonl"
PRIVATE_FEATURE_CACHE = PROJECT_ROOT / "data" / "study_screening" / "masvf_m0_private_feature_cache.jsonl"
CARDIUM_FEATURE_CACHE = PROJECT_ROOT / "data" / "study_screening" / "masvf_m0_cardium_feature_cache.jsonl"


@dataclass
class FrameRow:
    sample_id: str
    study_id: str
    patient_id: str
    label: int
    label_name: str
    split: str
    fold: int
    image_path: str
    cardium_view: str | None
    plane_conf: float
    feat: dict[str, float]
    meta: dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MASVF M0/M1 screening LR")
    p.add_argument("--model", choices=("m0", "m1"), default="m0",
                   help="m0=stats only; m1=stats+FetalCLIP concat")
    p.add_argument("--cohort", choices=("private", "cardium"), default="private")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--image-tags", type=Path, default=None)
    p.add_argument("--feature-cache", type=Path, default=None)
    p.add_argument("--cardium-processed", type=Path, default=PROJECT_ROOT / "CARDIUM dataset" / "processed")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--fetalclip-device", default="0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--min-plane-conf", type=float, default=0.15)
    p.add_argument("--folds", default="1,2,3", help="cardium only")
    p.add_argument("--split", default="", help="private: train|val|test; empty=all for train+eval")
    p.add_argument("--frame-select", default="fetalclip_diverse_gate",
                   choices=("none", "legacy", "paradigm_a", "fetalclip_diverse",
                            "fetalclip_diverse_gate", "paradigm_a_gate"))
    p.add_argument("--k-per-view", type=int, default=0, help="0 = no cap for gate modes")
    p.add_argument("--view-mode", choices=("4c", "4view", "all"), default="4view",
                   help="Filter frames by view after tagging")
    p.add_argument("--patient-agg", choices=("max", "mean"), default="max")
    p.add_argument("--clip-crop", choices=("none", "plane"), default="none",
                   help="M1 FetalCLIP input: full image or YOLO plane crop")
    p.add_argument("--crop-pad", type=float, default=DEFAULT_CROP_PAD,
                   help="Relative padding around plane box for --clip-crop plane")
    p.add_argument("--val-patient-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--features-only", action="store_true")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--rebuild-fetalclip-cache", action="store_true")
    p.add_argument("--max-studies", type=int, default=0)
    p.add_argument("--fetalclip-cache", type=Path, default=None)
    p.add_argument(
        "--m0-features",
        default="all",
        choices=tuple(M0_FEATURE_SUBSETS.keys()),
        help="M0 subset: all=24-d; true_ratios=3 cross-domain anatomical ratios only",
    )
    return p.parse_args()


def load_tags(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by_rel: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["sample_id"]] = row
                if row.get("rel_path"):
                    by_rel[row["rel_path"]] = row
    out["_by_rel"] = by_rel  # type: ignore[assignment]
    return out


def load_feature_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["sample_id"]] = row
    return out


def append_feature_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tag_for_private(study_id: str, rel: str, tags: dict[str, dict]) -> dict:
    sid = f"{study_id}|{rel}"
    if sid in tags:
        return tags[sid]
    return (tags.get("_by_rel") or {}).get(rel, {})


def load_private_studies(manifest: Path, split: str) -> list[dict]:
    studies = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                if split and s.get("split") != split:
                    continue
                studies.append(s)
    return studies


def assign_view_from_yolo(res, min_conf: float) -> tuple[str | None, float]:
    dets = []
    if res.boxes is not None and len(res.boxes):
        for b in res.boxes:
            dets.append((int(b.cls.item()), float(b.conf.item())))
    pa = dominant_plane_from_dets(dets, min_conf=min_conf)
    if pa is None:
        return None, 0.0
    return map_to_cardium_view(pa.plane_code), pa.conf


def extract_feat_from_yolo(res) -> dict[str, float]:
    try:
        return extract_chamber_features(yolo_to_anatomy_boxes(res)).values
    except Exception:
        return {}


def is_readable_image(path: Path) -> bool:
    """Deprecated: do not pre-check images on NFS. Kept for callers; always True."""
    return True


def yolo_predict_paths(yolo_model, paths: list[str], args: argparse.Namespace) -> dict[str, object]:
    """Predict path→result. Skip pre-open/verify; let YOLO fail fast on bad files.

    Previous code called PIL.verify() on every path before GPU work, which on NFS
    left VRAM allocated while util≈0 for minutes per batch.
    """
    out: dict[str, object] = {}
    if not paths:
        return out
    try:
        results = yolo_model.predict(
            source=paths, conf=args.yolo_conf, imgsz=args.yolo_imgsz,
            device=args.device, verbose=False, stream=False,
        )
        for p, res in zip(paths, results):
            out[p] = res
        return out
    except Exception as exc:
        print(
            f"  WARN: YOLO batch failed ({type(exc).__name__}: {exc}); per-image fallback",
            file=sys.stderr, flush=True,
        )
    skipped = 0
    for p in paths:
        try:
            res = yolo_model.predict(
                source=[p], conf=args.yolo_conf, imgsz=args.yolo_imgsz,
                device=args.device, verbose=False, stream=False,
            )
            out[p] = res[0]
        except Exception:
            skipped += 1
    if skipped:
        acc = getattr(args, "_unreadable_skip_total", None)
        if acc is not None:
            args._unreadable_skip_total = int(acc) + skipped
        else:
            print(f"  WARN: skipped {skipped}/{len(paths)} unreadable frames", flush=True)
    return out


def build_private_frame_rows(
    studies: list[dict],
    *,
    data_root: Path,
    tags: dict[str, dict],
    yolo_model,
    args: argparse.Namespace,
    cache: dict[str, dict],
) -> list[FrameRow]:
    rows: list[FrameRow] = []
    to_run: list[tuple[dict, str, Path, dict]] = []
    n_hit = 0
    n_scan = 0

    print(f"M0 scan: {len(studies)} studies against feature cache…", flush=True)
    for study in studies:
        study_id = study["study_id"]
        for rel in study.get("frame_paths") or []:
            n_scan += 1
            # Absolute tertiary/CHD paths: use as-is; never Path.resolve() (NFS stall).
            p = Path(rel)
            img_path = p if p.is_absolute() else (data_root / rel)
            sample_id = f"{study_id}|{rel}"
            tag = tag_for_private(study_id, rel, tags)
            if sample_id in cache and not args.rebuild_cache:
                n_hit += 1
                c = cache[sample_id]
                meta = dict(c.get("meta") or {})
                meta.setdefault("source_cohort", study.get("source_cohort", ""))
                meta.setdefault("source_mode", study.get("source_mode", ""))
                rows.append(FrameRow(
                    sample_id=sample_id,
                    study_id=study_id,
                    patient_id=study["patient_id"],
                    label=int(study["label_binary"]),
                    label_name=study.get("label_disease", ""),
                    split=study.get("split", ""),
                    fold=0,
                    image_path=str(img_path),
                    cardium_view=c.get("cardium_view"),
                    plane_conf=float(c.get("plane_conf", 0.0)),
                    feat=c["feat"],
                    meta=meta,
                ))
            else:
                # Skip existence probe here; missing files fail later in YOLO batch.
                to_run.append((study, rel, img_path, tag))
            if n_scan % 50000 == 0:
                print(
                    f"  … scanned {n_scan} frames (cache_hit={n_hit} miss={len(to_run)})",
                    flush=True,
                )
    print(
        f"M0 scan done: frames={n_scan} hit={n_hit} miss={len(to_run)}",
        flush=True,
    )

    if to_run and yolo_model is None:
        if getattr(args, "allow_missing_cache", False):
            print(f"WARN: skip {len(to_run)} private frames missing feature cache")
            to_run = []
        else:
            raise RuntimeError(
                f"{len(to_run)} private frames missing cache; run without --features-only"
            )

    new_cache: list[dict] = []
    n_skip = 0
    n_done = 0
    flush_every = max(1, int(getattr(args, "cache_flush_batches", 20)))
    n_batches = (len(to_run) + args.batch_size - 1) // args.batch_size if to_run else 0
    if to_run:
        print(
            f"M0 YOLO backfill: {len(to_run)} frames / {n_batches} batches "
            f"(batch_size={args.batch_size})",
            flush=True,
        )
    for i in range(0, len(to_run), args.batch_size):
        chunk = to_run[i : i + args.batch_size]
        paths = [str(x[2]) for x in chunk]
        pred_map = yolo_predict_paths(yolo_model, paths, args)
        for study, rel, img_path, tag in chunk:
            res = pred_map.get(str(img_path))
            if res is None:
                n_skip += 1
                continue
            study_id = study["study_id"]
            sample_id = f"{study_id}|{rel}"
            if tag.get("cardium_view") is not None:
                view, conf = tag["cardium_view"], float(tag.get("plane_conf", 0.0))
            else:
                view, conf = assign_view_from_yolo(res, args.min_plane_conf)
            feat = extract_feat_from_yolo(res)
            meta = {
                "cardium_view": view,
                "plane_conf": conf,
                "plane_area_norm": tag.get("plane_area_norm", 0.0),
                "plane_xyxy": tag.get("plane_xyxy"),
                "detected_class_ids": list(detected_ids_from_tag(tag)) or None,
                "source_cohort": study.get("source_cohort", ""),
                "source_mode": study.get("source_mode", ""),
            }
            if meta["detected_class_ids"] is None:
                meta["detected_class_ids"] = list(detected_ids_from_tag({"present": {
                    k: bool(feat.get(f"{k}_present", 0)) for k in ("lv", "rv", "la", "ra", "ivs", "crux")
                }}))
            row = FrameRow(
                sample_id=sample_id,
                study_id=study_id,
                patient_id=study["patient_id"],
                label=int(study["label_binary"]),
                label_name=study.get("label_disease", ""),
                split=study.get("split", ""),
                fold=0,
                image_path=str(img_path),
                cardium_view=view,
                plane_conf=conf,
                feat=feat,
                meta=meta,
            )
            rows.append(row)
            cache_entry = {
                "sample_id": sample_id,
                "cardium_view": view,
                "plane_conf": conf,
                "feat": feat,
                "meta": meta,
            }
            cache[sample_id] = cache_entry
            new_cache.append(cache_entry)
            n_done += 1

        bi = i // args.batch_size + 1
        if bi % 5 == 0 or bi == n_batches:
            print(
                f"  M0 backfill [{bi}/{n_batches}] done={n_done} skip={n_skip}",
                flush=True,
            )
        # Periodic flush so NFS kill / restart does not lose hours of work
        if new_cache and not args.rebuild_cache and bi % flush_every == 0:
            append_feature_cache(args.feature_cache, new_cache)
            print(f"  flushed {len(new_cache)} rows → {args.feature_cache}", flush=True)
            new_cache = []

    if new_cache and not args.rebuild_cache:
        append_feature_cache(args.feature_cache, new_cache)
        print(f"  flushed final {len(new_cache)} rows → {args.feature_cache}", flush=True)
    if to_run:
        print(f"M0 YOLO backfill finished: wrote={n_done} skip={n_skip}", flush=True)
    return rows


# Midlate Chinese view folder → CARDIUM 4-view slot (vv collapses to vvt)
MIDLATE_CN_TO_CARDIUM: dict[str, str] = {
    "四腔心切面心脏": "four_chamber",
    "左室流出道切面心脏": "lvot",
    "右室流出道切面心脏": "rvot",
    "三血管气管切面心脏": "vvt",
    "三血管切面心脏": "vvt",
}


def build_midlate_frame_rows(
    *,
    norm_corpus: Path,
    frames_per_view: int,
    seed: int,
    yolo_model,
    args: argparse.Namespace,
    cache: dict[str, dict],
    max_norm: int = 0,
    synth_mode: str = "multiview",
    n_synth_patients: int = 0,
) -> list[FrameRow]:
    """Homologous midlate normals as FrameRows (synth patients; GT view + box).

    synth_mode:
      - multiview (fusion default): ``n_synth_patients`` bags, each with
        ``frames_per_view`` frames on each of 4CH/LVOT/RVOT/VVT.
      - frame (legacy / image LoRA): each sampled plane frame = 1 patient;
        ``frames_per_view`` = samples per Chinese view folder.
    """
    from midlate_crop_fetalclip_linear import (  # local import avoids cycles
        collect_normals_midlate,
        sample_midlate_as_multiview_synth_patients,
        sample_midlate_as_synth_patients,
    )

    raw = collect_normals_midlate(Path(norm_corpus), max_n=max_norm)
    mode = (synth_mode or "multiview").lower()
    if mode == "multiview":
        n_pat = int(n_synth_patients) if int(n_synth_patients) > 0 else max(1, int(frames_per_view))
        fpv = max(1, int(frames_per_view)) if int(frames_per_view) > 0 else 1
        # When auto-balance sets n_synth_patients, keep 1 frame/view (fusion pools to 1 token/view)
        if int(n_synth_patients) > 0:
            fpv = 1
        dicts = sample_midlate_as_multiview_synth_patients(
            raw, n_pat, seed=seed, frames_per_view=fpv,
        )
    else:
        dicts = sample_midlate_as_synth_patients(raw, frames_per_view, seed=seed)
    rows: list[FrameRow] = []
    to_run: list[dict] = []

    for d in dicts:
        path = Path(d["path"])
        # Do not path.is_file()/resolve() — NFS stall; YOLO try/except skips missing.
        view_cn = str(d.get("view") or "")
        cardium_view = str(d.get("cardium_view") or "") or MIDLATE_CN_TO_CARDIUM.get(view_cn)
        if cardium_view is None:
            continue
        stem = path.stem
        patient_id = str(d["patient_id"])
        # Cache key stays image-stable; patient_id carries the synth bag identity.
        sample_id = f"midlate|{view_cn}|{stem}"
        box = d.get("box")
        plane_xyxy = list(box) if box is not None else None

        if sample_id in cache and not getattr(args, "rebuild_cache", False):
            c = cache[sample_id]
            meta = dict(c.get("meta") or {})
            meta["cardium_view"] = cardium_view
            meta["plane_xyxy"] = plane_xyxy or meta.get("plane_xyxy")
            meta["homologous_midlate"] = True
            meta["source_cohort"] = "midlate"
            rows.append(FrameRow(
                sample_id=sample_id,
                study_id=patient_id,
                patient_id=patient_id,
                label=0,
                label_name="midlate_normal",
                split="train",
                fold=0,
                image_path=str(path),
                cardium_view=cardium_view,
                plane_conf=float(c.get("plane_conf", 1.0)),
                feat=c.get("feat") or {},
                meta=meta,
            ))
        else:
            to_run.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "path": path,
                "cardium_view": cardium_view,
                "plane_xyxy": plane_xyxy,
            })

    if to_run and yolo_model is None:
        if getattr(args, "allow_missing_cache", False) or getattr(args, "features_only", False):
            # zeros M0 fallback so homologous smoke can proceed without YOLO
            print(f"WARN: midlate M0 cache miss n={len(to_run)}; using GT-view + empty feat")
            for item in to_run:
                meta = {
                    "cardium_view": item["cardium_view"],
                    "plane_conf": 1.0,
                    "plane_area_norm": 0.0,
                    "plane_xyxy": item["plane_xyxy"],
                    "detected_class_ids": None,
                    "homologous_midlate": True,
                }
                rows.append(FrameRow(
                    sample_id=item["sample_id"],
                    study_id=item["patient_id"],
                    patient_id=item["patient_id"],
                    label=0,
                    label_name="midlate_normal",
                    split="train",
                    fold=0,
                    image_path=str(item["path"]),
                    cardium_view=item["cardium_view"],
                    plane_conf=1.0,
                    feat={},
                    meta=meta,
                ))
            to_run = []
        else:
            raise RuntimeError(
                f"{len(to_run)} midlate frames missing M0 cache; run without --features-only"
            )

    new_cache: list[dict] = []
    for i in range(0, len(to_run), getattr(args, "batch_size", 32)):
        chunk = to_run[i : i + getattr(args, "batch_size", 32)]
        paths = [str(x["path"]) for x in chunk]
        pred_map = yolo_predict_paths(yolo_model, paths, args)
        for item in chunk:
            res = pred_map.get(str(item["path"]))
            feat = extract_feat_from_yolo(res) if res is not None else {}
            conf = 1.0
            if res is not None:
                _v, conf = assign_view_from_yolo(res, getattr(args, "min_plane_conf", 0.15))
                if conf <= 0:
                    conf = 1.0
            meta = {
                "cardium_view": item["cardium_view"],
                "plane_conf": float(conf),
                "plane_area_norm": 0.0,
                "plane_xyxy": item["plane_xyxy"],
                "detected_class_ids": None,
                "homologous_midlate": True,
            }
            row = FrameRow(
                sample_id=item["sample_id"],
                study_id=item["patient_id"],
                patient_id=item["patient_id"],
                label=0,
                label_name="midlate_normal",
                split="train",
                fold=0,
                image_path=str(item["path"]),
                cardium_view=item["cardium_view"],
                plane_conf=float(conf),
                feat=feat,
                meta=meta,
            )
            rows.append(row)
            entry = {
                "sample_id": item["sample_id"],
                "cardium_view": item["cardium_view"],
                "plane_conf": float(conf),
                "feat": feat,
                "meta": meta,
            }
            cache[item["sample_id"]] = entry
            new_cache.append(entry)

    if new_cache and not getattr(args, "rebuild_cache", False):
        append_feature_cache(args.feature_cache, new_cache)
    print(
        f"[midlate-FrameRow] n={len(rows)} "
        f"patients={len({r.patient_id for r in rows})} "
        f"views={ {v: sum(1 for r in rows if r.cardium_view == v) for v in sorted({r.cardium_view for r in rows if r.cardium_view})} }"
    )
    return rows


def build_cardium_frame_rows(
    records: list[CardiumRecord],
    *,
    tags: dict[str, dict],
    yolo_model,
    args: argparse.Namespace,
    cache: dict[str, dict],
) -> list[FrameRow]:
    rows: list[FrameRow] = []
    to_run: list[CardiumRecord] = []
    for rec in records:
        # No is_file() probe — missing/corrupt handled in yolo_predict_paths try/except.
        if rec.sample_id in cache and not args.rebuild_cache:
            c = cache[rec.sample_id]
            rows.append(FrameRow(
                sample_id=rec.sample_id,
                patient_id=rec.patient_id,
                study_id=rec.patient_id,
                label=rec.label_binary,
                label_name=rec.label_name,
                split=rec.split,
                fold=rec.fold,
                image_path=str(rec.image_path),
                cardium_view=c.get("cardium_view"),
                plane_conf=float(c.get("plane_conf", 0.0)),
                feat=c["feat"],
                meta=c.get("meta", {}),
            ))
        else:
            to_run.append(rec)

    if to_run and yolo_model is None:
        if getattr(args, "allow_missing_cache", False):
            print(f"WARN: skip {len(to_run)} cardium frames missing feature cache")
            to_run = []
        else:
            raise RuntimeError(f"{len(to_run)} cardium frames missing cache")

    new_cache: list[dict] = []
    n_skip = 0
    for i in range(0, len(to_run), args.batch_size):
        chunk = to_run[i : i + args.batch_size]
        paths = [str(r.image_path) for r in chunk]
        pred_map = yolo_predict_paths(yolo_model, paths, args)
        for rec in chunk:
            res = pred_map.get(str(rec.image_path))
            if res is None:
                n_skip += 1
                continue
            tag = tags.get(rec.sample_id, {})
            if tag.get("cardium_view") is not None:
                view, conf = tag["cardium_view"], float(tag.get("plane_conf", 0.0))
            else:
                view, conf = assign_view_from_yolo(res, args.min_plane_conf)
            feat = extract_feat_from_yolo(res)
            meta = {
                "cardium_view": view,
                "plane_conf": conf,
                "plane_area_norm": float(tag.get("plane_area_norm", 0.0) or 0.0),
                "plane_xyxy": tag.get("plane_xyxy"),
                "detected_class_ids": list(detected_ids_from_tag(tag)),
            }
            rows.append(FrameRow(
                sample_id=rec.sample_id,
                study_id=rec.patient_id,
                patient_id=rec.patient_id,
                label=rec.label_binary,
                label_name=rec.label_name,
                split=rec.split,
                fold=rec.fold,
                image_path=str(rec.image_path),
                cardium_view=view,
                plane_conf=conf,
                feat=feat,
                meta=meta,
            ))
            cache_entry = {
                "sample_id": rec.sample_id,
                "cardium_view": view,
                "plane_conf": conf,
                "feat": feat,
                "meta": meta,
            }
            cache[rec.sample_id] = cache_entry
            new_cache.append(cache_entry)

    if new_cache and not args.rebuild_cache:
        append_feature_cache(args.feature_cache, new_cache)
    return rows


def filter_view_mode(rows: list[FrameRow], mode: str) -> list[FrameRow]:
    if mode == "4c":
        return [r for r in rows if r.cardium_view == "four_chamber"]
    if mode == "4view":
        return [r for r in rows if r.cardium_view in CARDIUM_STANDARD_VIEWS]
    return rows


def row_to_select_dict(r: FrameRow, tag: dict) -> dict:
    meta = r.meta or {}
    return {
        "path": r.image_path,
        "sample_id": r.sample_id,
        "cardium_view": r.cardium_view,
        "plane_conf": r.plane_conf,
        "score": float(tag.get("score", tag.get("plane_conf", 0.05))),
        "present": tag.get("present", {}),
        "p4c": float(tag.get("p4c", 0.0)),
        "plane_area_norm": meta.get("plane_area_norm", tag.get("plane_area_norm")),
        "detected_class_ids": meta.get("detected_class_ids", tag.get("detected_class_ids")),
    }


def apply_frame_selection(
    rows: list[FrameRow],
    tags: dict[str, dict],
    args: argparse.Namespace,
    embed_cache: dict[str, np.ndarray] | None,
) -> list[FrameRow]:
    if args.frame_select == "none":
        return rows

    k = args.k_per_view
    if args.frame_select.endswith("_gate") and k <= 0:
        k = 0

    if args.cohort == "private":
        group_key = lambda r: (r.study_id, r.cardium_view or "")
    else:
        group_key = lambda r: (r.fold, r.split, r.patient_id, r.cardium_view or "")

    groups: dict[tuple, list[FrameRow]] = defaultdict(list)
    for r in rows:
        if r.cardium_view:
            groups[group_key(r)].append(r)

    selected: list[FrameRow] = []
    for grp in groups.values():
        frame_dicts = []
        for im in grp:
            tag = tags.get(im.sample_id, {})
            if not tag.get("cardium_view") and "|" in im.sample_id:
                _sid, rel = im.sample_id.split("|", 1)
                tag = tag_for_private(_sid, rel, tags)
            frame_dicts.append(row_to_select_dict(im, tag))
        paths = select_frames(
            frame_dicts, k, method=args.frame_select, path_key="path",
            embeddings=embed_cache,
        )
        pset = set(paths)
        selected.extend(im for im in grp if im.image_path in pset)
    return selected


def split_val_patients(items: list[T], ratio: float, seed: int, pid_fn) -> tuple[list[T], list[T]]:
    pids = sorted(set(pid_fn(x) for x in items))
    rng = random.Random(seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * ratio))
    val_set = set(pids[:n_val])
    fit = [x for x in items if pid_fn(x) not in val_set]
    val = [x for x in items if pid_fn(x) in val_set]
    return fit, val


def rows_to_X(
    rows: list[FrameRow],
    *,
    model: str,
    embed_cache: dict[str, np.ndarray] | None,
    m0_features: str | list[str] = "all",
    appearance_dim: int | None = None,
) -> np.ndarray:
    m0_names = resolve_m0_feature_names(m0_features)
    m0_vecs = []
    for r in rows:
        meta = dict(r.meta or {})
        meta.setdefault("cardium_view", r.cardium_view)
        meta.setdefault("plane_conf", r.plane_conf)
        full = m0_feat_vector(r.feat, meta)
        m0_vecs.append(select_m0_columns(full.reshape(1, -1), m0_names).ravel())
    X_m0 = np.stack(m0_vecs)
    if model == "m0":
        return X_m0
    if embed_cache is None:
        raise RuntimeError("M1 requires appearance embedding cache")

    dim = int(appearance_dim) if appearance_dim is not None else FETALCLIP_EMBED_DIM
    if appearance_dim is None:
        for vec in embed_cache.values():
            dim = int(np.asarray(vec).ravel().shape[0])
            break

    clip_vecs = []
    missing = 0
    for r in rows:
        emb = lookup_embedding(embed_cache, sample_id=r.sample_id, image_path=r.image_path)
        if emb is None:
            missing += 1
            clip_vecs.append(np.full(dim, np.nan, dtype=np.float64))
        else:
            arr = np.asarray(emb, dtype=np.float64).ravel()
            if arr.shape[0] != dim:
                raise RuntimeError(
                    f"appearance dim mismatch: cache vec {arr.shape[0]} vs expected {dim} "
                    f"(sample_id={r.sample_id!r}). Cache file likely mixed "
                    f"(FetalCLIP 768 appended into ImageNet cache). Rebuild with "
                    f"build_imagenet_backbone_embed_cache.py --rebuild"
                )
            clip_vecs.append(arr)
    if missing:
        print(f"  WARN: {missing}/{len(rows)} frames missing appearance embedding (median impute)")
    return np.hstack([X_m0, np.stack(clip_vecs)])


def _l2_normalize_rows(X: np.ndarray, n_m0: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    if n_m0 <= 0 or X.shape[1] <= n_m0:
        return X
    clip = X[:, n_m0:]
    norms = np.linalg.norm(clip, axis=1, keepdims=True)
    norms = np.where(np.isnan(norms) | (norms < 1e-8), 1.0, norms)
    nan_mask = np.isnan(clip)
    clip = clip / norms
    clip[nan_mask] = np.nan
    X[:, n_m0:] = clip
    return X


class HybridM0ClipTransformer:
    """sklearn-compatible: scale M0 columns; L2-normalize CLIP block per row."""

    def __init__(self, n_m0: int = 0):
        self.n_m0 = int(n_m0)
        self.m0_scaler = StandardScaler()

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.m0_scaler.fit(X[:, : self.n_m0])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = self.n_m0
        X = _l2_normalize_rows(X, n)
        m0 = self.m0_scaler.transform(X[:, :n])
        return np.hstack([m0, X[:, n:]])

    def get_params(self, deep=True):
        return {"n_m0": self.n_m0}

    def set_params(self, **params):
        if "n_m0" in params:
            self.n_m0 = int(params["n_m0"])
        return self


def fit_lr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_m0: int = 0,
    clip_norm: str = "joint",
) -> tuple[Pipeline, np.ndarray]:
    """Fit frame-level LR.

    clip_norm:
      - joint: StandardScaler on full [M0‖CLIP] (legacy)
      - hybrid: StandardScaler on M0 only; CLIP rows L2-normalized (no per-dim scale)
    """
    X_imp, fill = impute_features(X)
    mode = str(clip_norm or "joint").lower()
    if mode == "hybrid" and int(n_m0) > 0 and X_imp.shape[1] > int(n_m0):
        clf = Pipeline([
            ("hybrid", HybridM0ClipTransformer(n_m0=int(n_m0))),
            ("lr", _make_logistic_regression()),
        ])
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", _make_logistic_regression()),
        ])
    clf.fit(X_imp, y)
    return clf, fill


# Groups for LR coef attribution (protocol / FOV vs true ratios).
_LR_COEF_GROUPS: dict[str, tuple[str, ...]] = {
    "protocol": (
        "chamber_count", "chamber_lt4", "chamber_le2",
        "ivs_present", "ivs_missing", "crux_present", "crux_missing",
        "lv_present", "rv_present", "la_present", "ra_present",
        "anomaly_vote", "key_anatomy_completeness",
    ),
    "fov": ("plane_area_norm", "vent_area_norm", "vent_area_total"),
    "true_ratios": (
        "lv_rv_area_ratio", "la_ra_area_ratio", "left_right_area_ratio",
        "lv_rv_w_ratio", "la_ra_w_ratio", "ivs_vent_area_ratio",
        "crux_vent_area_ratio", "atrium_vent_area_ratio",
    ),
    "view": ("view_four_chamber", "view_lvot", "view_rvot", "view_vvt"),
    "conf": ("plane_conf",),
}


def summarize_frame_lr_coefs(
    clf: Pipeline,
    *,
    m0_names: list[str],
    n_m0: int,
    top_k: int = 15,
) -> dict:
    """Absolute LR coefs on M0 dims (+ CLIP L2 mass). Coefs are in post-scaler space."""
    lr = clf.named_steps.get("lr")
    if lr is None or not hasattr(lr, "coef_"):
        return {"error": "no lr step"}
    coef = np.asarray(lr.coef_, dtype=np.float64).ravel()
    n_m0 = int(min(max(n_m0, 0), len(coef), len(m0_names)))
    m0_coef = coef[:n_m0] if n_m0 > 0 else np.zeros(0, dtype=np.float64)
    clip_coef = coef[n_m0:] if len(coef) > n_m0 else np.zeros(0, dtype=np.float64)
    names = list(m0_names[:n_m0])
    ranked = sorted(
        (
            {
                "name": names[i],
                "coef": float(m0_coef[i]),
                "abs_coef": float(abs(m0_coef[i])),
            }
            for i in range(n_m0)
        ),
        key=lambda r: r["abs_coef"],
        reverse=True,
    )
    group_l1: dict[str, float] = {}
    name_to_abs = {r["name"]: r["abs_coef"] for r in ranked}
    assigned = set()
    for g, members in _LR_COEF_GROUPS.items():
        s = 0.0
        for n in members:
            if n in name_to_abs:
                s += name_to_abs[n]
                assigned.add(n)
        group_l1[g] = float(s)
    group_l1["other_m0"] = float(sum(v for k, v in name_to_abs.items() if k not in assigned))
    m0_l1 = float(np.abs(m0_coef).sum()) if n_m0 else 0.0
    clip_l2 = float(np.linalg.norm(clip_coef)) if clip_coef.size else 0.0
    return {
        "n_m0": n_m0,
        "n_clip": int(clip_coef.size),
        "m0_l1": m0_l1,
        "clip_coef_l2": clip_l2,
        "m0_share_of_l1": float(m0_l1 / (m0_l1 + float(np.abs(clip_coef).sum()) + 1e-12)),
        "group_l1": group_l1,
        "top_m0": ranked[: int(top_k)],
        "all_m0": ranked,
        "preprocess": "hybrid" if "hybrid" in clf.named_steps else "joint_scaler",
    }


def print_frame_lr_coef_summary(summary: dict, *, top_k: int = 10) -> None:
    if summary.get("error"):
        print(f"  LR coefs: {summary['error']}")
        return
    gl = summary.get("group_l1") or {}
    print(
        f"  LR coefs [{summary.get('preprocess')}]: "
        f"m0_l1={summary.get('m0_l1', 0):.3f} "
        f"clip_l2={summary.get('clip_coef_l2', 0):.3f} "
        f"m0_share≈{100 * summary.get('m0_share_of_l1', 0):.1f}%"
    )
    if gl:
        parts = " ".join(f"{k}={v:.3f}" for k, v in gl.items() if v > 0)
        print(f"  LR M0 group |coef| L1: {parts}")
    top = (summary.get("top_m0") or [])[:top_k]
    if top:
        bits = ", ".join(f"{r['name']}={r['coef']:+.3f}" for r in top)
        print(f"  LR M0 top{len(top)}: {bits}")


def predict_lr(clf: Pipeline, fill: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_imp, _ = impute_features(X, fill)
    return clf.predict_proba(X_imp)[:, 1]


def standardize_feature_blocks(
    X_fit: np.ndarray,
    *X_others: np.ndarray,
    n_m0: int,
    clip_norm: str = "joint",
) -> tuple[np.ndarray, ...]:
    """Align fusion feature scaling with LR: hybrid = M0 z-score, CLIP L2 only."""
    mode = str(clip_norm or "joint").lower()
    Xs = [np.asarray(X_fit, dtype=np.float64)] + [np.asarray(x, dtype=np.float64) for x in X_others]
    if mode == "hybrid" and int(n_m0) > 0 and Xs[0].shape[1] > int(n_m0):
        n = int(n_m0)
        out: list[np.ndarray] = []
        mu = sig = None
        for i, X in enumerate(Xs):
            X = _l2_normalize_rows(X, n)
            if i == 0:
                mu = X[:, :n].mean(axis=0)
                sig = X[:, :n].std(axis=0)
                sig = np.where(sig < 1e-6, 1.0, sig)
            m0 = (X[:, :n] - mu) / sig
            out.append(np.hstack([m0, X[:, n:]]).astype(np.float32))
        return tuple(out)
    out = []
    mu = sig = None
    for i, X in enumerate(Xs):
        if i == 0:
            mu = X.mean(axis=0)
            sig = X.std(axis=0)
            sig = np.where(sig < 1e-6, 1.0, sig)
        out.append(((X - mu) / sig).astype(np.float32))
    return tuple(out)


def aggregate_patient_mean(pids, y_true, y_prob, threshold):
    from collections import defaultdict
    groups = defaultdict(list)
    for pid, yt, yp in zip(pids, y_true, y_prob):
        groups[str(pid)].append((int(yt), float(yp)))
    ordered = sorted(groups)
    y_true_p = np.array([groups[p][0][0] for p in ordered], dtype=np.int64)
    y_prob_p = np.array([float(np.mean([x[1] for x in groups[p]])) for p in ordered])
    y_pred_p = (y_prob_p >= threshold).astype(np.int64)
    meta = {
        "aggregation": "mean",
        "num_patients": len(ordered),
        "avg_images_per_patient": float(np.mean([len(groups[p]) for p in ordered])),
    }
    return y_true_p, y_pred_p, y_prob_p, meta


def eval_rows(
    name: str,
    rows: list[FrameRow],
    y_prob: np.ndarray,
    threshold: float,
    agg: str,
    *,
    patient_threshold: float | None = None,
) -> dict:
    """Evaluate image + patient metrics.

    ``threshold`` is the image operating point (val-tuned on frames).
    ``patient_threshold`` defaults to the same value for backward compat; prefer
    a patient-agg val-tuned threshold so F1 is not inflated/deflated by AUC.
    """
    y_true = np.array([r.label for r in rows], dtype=np.int64)
    img_m = binary_metrics_at_threshold(y_true, y_prob, threshold)
    pids = np.array([r.patient_id for r in rows])
    if agg == "max":
        y_true_p, _, y_prob_p, pmeta = aggregate_patient_max_confidence(
            pids, y_true, y_prob, threshold=threshold
        )
    else:
        y_true_p, _, y_prob_p, pmeta = aggregate_patient_mean(pids, y_true, y_prob, threshold)
    thr_p = float(threshold if patient_threshold is None else patient_threshold)
    pat_m = binary_metrics_at_threshold(y_true_p, y_prob_p, thr_p)

    print(f"\n=== {name} ===")
    print(f"  frames={len(rows)} patients={pmeta['num_patients']} avg_f/p={pmeta['avg_images_per_patient']:.1f}")
    print(
        f"  IMAGE   F1={img_m['f1']:.3f} AUC={img_m['auc']:.3f} "
        f"Sens={img_m['sensitivity']:.3f} Spec={img_m['specificity']:.3f} thr={threshold:.2f}"
    )
    print(
        f"  PATIENT F1={pat_m['f1']:.3f} AUC={pat_m['auc']:.3f} "
        f"Sens={pat_m['sensitivity']:.3f} Spec={pat_m['specificity']:.3f} thr={thr_p:.2f} agg={agg}"
    )

    return {
        "n_frames": len(rows),
        "n_patients": int(pmeta["num_patients"]),
        "avg_frames_per_patient": pmeta["avg_images_per_patient"],
        "threshold": threshold,
        "patient_threshold": thr_p,
        "patient_agg": agg,
        "image": {k: v for k, v in img_m.items() if k != "confusion_matrix"},
        "patient": {k: v for k, v in pat_m.items() if k != "confusion_matrix"},
        "view_counts": dict(Counter(r.cardium_view for r in rows)),
    }


def thresholds_from_val_rows(
    rows: list[FrameRow],
    y_prob: np.ndarray,
    agg: str,
) -> tuple[float, float]:
    """Val-tune image thr and patient-agg thr for max F1."""
    y_true = np.array([r.label for r in rows], dtype=np.int64)
    thr_img, _ = find_best_binary_threshold(y_true, y_prob)
    pids = np.array([r.patient_id for r in rows])
    if agg == "max":
        y_p, _, p_p, _ = aggregate_patient_max_confidence(pids, y_true, y_prob, threshold=0.5)
    else:
        y_p, _, p_p, _ = aggregate_patient_mean(pids, y_true, y_prob, 0.5)
    thr_pat, _ = find_best_binary_threshold(y_p, p_p)
    return float(thr_img), float(thr_pat)


def run_private(args: argparse.Namespace, tags: dict, cache: dict, yolo_model, embed_cache) -> dict:
    studies = load_private_studies(args.manifest, "")
    if args.max_studies > 0:
        studies = studies[: args.max_studies]
    all_rows = build_private_frame_rows(
        studies, data_root=args.data_root, tags=tags, yolo_model=yolo_model, args=args, cache=cache,
    )
    all_rows = filter_view_mode(all_rows, args.view_mode)
    all_rows = apply_frame_selection(all_rows, tags, args, embed_cache)

    train_rows = [r for r in all_rows if r.split == "train"]
    val_rows = [r for r in all_rows if r.split == "val"]
    test_rows = [r for r in all_rows if r.split == "test"]

    if len(train_rows) < 20:
        return {"error": "insufficient train frames"}

    fit_rows, holdout = split_val_patients(train_rows, args.val_patient_ratio, args.seed, lambda r: r.patient_id)
    if not val_rows:
        val_rows = holdout

    clf, fill = fit_lr(
        rows_to_X(fit_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features),
        np.array([r.label for r in fit_rows]),
    )
    val_prob = predict_lr(clf, fill, rows_to_X(val_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features))
    thr, thr_pat = thresholds_from_val_rows(val_rows, val_prob, args.patient_agg)

    results = {
        "cohort": "private",
        "model": args.model,
        "protocol": {
            "clip_crop": args.clip_crop,
            "crop_pad": args.crop_pad,
            "frame_select": args.frame_select,
            "patient_agg": args.patient_agg,
            "view_mode": args.view_mode,
            "note": "C2-b iff model=m1 and clip_crop=plane and patient_agg=max; primary metric=patient F1",
        },
        "val_thresholds": {"image": thr, "patient": thr_pat},
        "splits": {},
    }
    for split_name, split_rows in [("val", val_rows), ("test", test_rows)]:
        if not split_rows:
            continue
        prob = predict_lr(clf, fill, rows_to_X(split_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features))
        results["splits"][split_name] = eval_rows(
            f"private {split_name}", split_rows, prob, thr, args.patient_agg, patient_threshold=thr_pat
        )

    # study- and patient-level view coverage (for stratified reporting)
    study_views: dict[str, set[str]] = defaultdict(set)
    patient_views: dict[str, set[str]] = defaultdict(set)
    for r in all_rows:
        if r.cardium_view:
            study_views[r.study_id].add(r.cardium_view)
            patient_views[str(r.patient_id)].add(r.cardium_view)
    results["coverage"] = {
        "studies": len(study_views),
        "full_4view": sum(1 for v in study_views.values() if len(v) >= 4),
        "only_4c": sum(1 for v in study_views.values() if v == {"four_chamber"}),
        "views_present_hist": dict(Counter(len(v) for v in study_views.values())),
        "patient_views_present_hist": dict(Counter(len(v) for v in patient_views.values())),
        "n_patients_with_views": len(patient_views),
    }

    bundle = args.output_dir / f"private_{args.model}_lr_bundle.pkl"
    bundle.write_bytes(pickle.dumps({
        "clf": clf, "fill": fill, "threshold": thr, "patient_threshold": thr_pat,
    }))
    results["bundle"] = str(bundle)
    return results


def run_cardium(args: argparse.Namespace, tags: dict, cache: dict, yolo_model, embed_cache) -> dict:
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    all_records = []
    for fold in folds:
        for split in ("train", "test"):
            all_records.extend(load_fold_split(fold, split, args.cardium_processed))  # type: ignore[arg-type]

    all_rows = build_cardium_frame_rows(all_records, tags=tags, yolo_model=yolo_model, args=args, cache=cache)
    all_rows = filter_view_mode(all_rows, args.view_mode)
    all_rows = apply_frame_selection(all_rows, tags, args, embed_cache)

    results = {"cohort": "cardium", "model": args.model, "folds": {}, "comparison": {}}
    pooled = []

    for fold in folds:
        fold_int = int(fold)
        train_rows = [r for r in all_rows if r.fold == fold_int and r.split == "train"]
        test_rows = [r for r in all_rows if r.fold == fold_int and r.split == "test"]
        if len(train_rows) < 20 or len(test_rows) < 10:
            results["folds"][fold] = {"error": "insufficient"}
            continue

        fit_rows, val_rows = split_val_patients(
            train_rows, args.val_patient_ratio, args.seed + fold_int, lambda r: r.patient_id,
        )
        clf, fill = fit_lr(
            rows_to_X(fit_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features),
            np.array([r.label for r in fit_rows]),
        )
        val_prob = predict_lr(clf, fill, rows_to_X(val_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features))
        thr, thr_pat = thresholds_from_val_rows(val_rows, val_prob, args.patient_agg)
        test_prob = predict_lr(clf, fill, rows_to_X(test_rows, model=args.model, embed_cache=embed_cache, m0_features=args.m0_features))
        metrics = eval_rows(
            f"cardium fold{fold} test",
            test_rows,
            test_prob,
            thr,
            args.patient_agg,
            patient_threshold=thr_pat,
        )
        results["folds"][fold] = {
            "metrics": metrics,
            "threshold": thr,
            "patient_threshold": thr_pat,
        }
        pooled.append(metrics)

    if pooled:
        def _m(level, key):
            vals = [p[level][key] for p in pooled if key in p.get(level, {})]
            return float(np.mean(vals)) if vals else float("nan")
        results["comparison"] = {
            "patient_f1_mean": _m("patient", "f1"),
            "patient_auc_mean": _m("patient", "auc"),
            "patient_sens_mean": _m("patient", "sensitivity"),
            "patient_spec_mean": _m("patient", "specificity"),
            "image_f1_mean": _m("image", "f1"),
            "image_auc_mean": _m("image", "auc"),
        }
    return results


def collect_embed_items(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Collect (sample_id, path) without per-frame is_file/resolve (NFS stall)."""
    items: list[tuple[str, Path]] = []
    if args.cohort == "cardium":
        for fold in args.folds.split(","):
            for sp in ("train", "test"):
                for rec in load_fold_split(fold.strip(), sp, args.cardium_processed):  # type: ignore[arg-type]
                    items.append((rec.sample_id, rec.image_path))
    else:
        studies = load_private_studies(args.manifest, "")
        if args.max_studies > 0:
            studies = studies[: args.max_studies]
        for s in studies:
            for rel in s.get("frame_paths") or []:
                p = Path(rel)
                img = p if p.is_absolute() else (args.data_root / rel)
                items.append((f"{s['study_id']}|{rel}", img))
    return items


def collect_crop_boxes(
    tags: dict[str, dict],
    items: list[tuple[str, Path]],
) -> dict[str, tuple[float, float, float, float] | None]:
    """Map sample_id → normalized plane xyxy (None → full-image fallback)."""
    out: dict[str, tuple[float, float, float, float] | None] = {}
    by_rel = tags.get("_by_rel") or {}
    for sid, _path in items:
        tag = tags.get(sid, {})
        if not tag and "|" in sid:
            _study, rel = sid.split("|", 1)
            tag = by_rel.get(rel, {})
        xy = tag.get("plane_xyxy")
        if isinstance(xy, (list, tuple)) and len(xy) == 4:
            try:
                out[sid] = (float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]))
                continue
            except (TypeError, ValueError):
                pass
        out[sid] = None
    n_box = sum(1 for v in out.values() if v is not None)
    print(f"plane_xyxy available: {n_box}/{len(out)} (rest → full-image FetalCLIP)")
    return out


def needs_fetalclip_embeddings(args: argparse.Namespace) -> bool:
    if args.model == "m1":
        return True
    if args.frame_select == "fetalclip_diverse":
        return True
    if args.frame_select == "fetalclip_diverse_gate" and args.k_per_view > 0:
        return True
    return False


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        variant = args.model
        if args.model == "m1" and args.clip_crop == "plane":
            variant = "m1_crop"
        args.output_dir = PROJECT_ROOT / "outputs" / f"masvf_{variant}" / args.cohort
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_tags is None:
        args.image_tags = DEFAULT_PRIVATE_TAGS if args.cohort == "private" else DEFAULT_CARDIUM_TAGS
    if args.feature_cache is None:
        args.feature_cache = PRIVATE_FEATURE_CACHE if args.cohort == "private" else CARDIUM_FEATURE_CACHE
    if args.fetalclip_cache is None:
        if args.clip_crop == "plane":
            args.fetalclip_cache = PRIVATE_CROP_CACHE if args.cohort == "private" else CARDIUM_CROP_CACHE
        else:
            args.fetalclip_cache = PRIVATE_FETALCLIP_CACHE if args.cohort == "private" else FETALCLIP_CACHE

    tags = load_tags(args.image_tags)
    cache = load_feature_cache(args.feature_cache)

    embed_cache: dict[str, np.ndarray] | None = None

    yolo_model = None
    need_yolo = args.rebuild_cache
    if args.cohort == "private":
        studies = load_private_studies(args.manifest, "")
        if args.max_studies > 0:
            studies = studies[: args.max_studies]
        need_ids = {f"{s['study_id']}|{rel}" for s in studies for rel in s.get("frame_paths", [])}
        need_yolo = need_yolo or any(sid not in cache for sid in need_ids)
    else:
        recs = []
        for fold in args.folds.split(","):
            for sp in ("train", "test"):
                recs.extend(load_fold_split(fold.strip(), sp, args.cardium_processed))  # type: ignore[arg-type]
        need_yolo = need_yolo or any(r.sample_id not in cache for r in recs)

    if need_yolo:
        if args.features_only:
            print("ERROR: --features-only but feature cache incomplete", file=sys.stderr)
            return 1
        weights = resolve_weights(args.yolo_weights)
        print(f"Loading YOLO: {weights}")
        yolo_model = load_yolo_model(weights)
    else:
        print("Feature cache hit → skip YOLO")

    if needs_fetalclip_embeddings(args):
        items = collect_embed_items(args)
        crop_boxes = None
        if args.clip_crop == "plane":
            crop_boxes = collect_crop_boxes(tags, items)
            if args.model == "m1" and sum(1 for v in crop_boxes.values() if v) == 0:
                src = "private" if args.cohort == "private" else "cardium"
                print(
                    "ERROR: --clip-crop plane but no plane_xyxy in tags.\n"
                    f"  Fix: cd experiments && CUDA_VISIBLE_DEVICES=0 python agcd/enrich_tag_cache_plane_xyxy.py "
                    f"--source {src} --device 0 --inplace\n"
                    f"  Then re-run with the same tags path (or TAGS=..._xyxy.jsonl).",
                    file=sys.stderr,
                )
                return 1
        embed_cache = build_or_load_embeddings(
            items,
            cache_path=args.fetalclip_cache,
            device=args.fetalclip_device,
            batch_size=args.batch_size,
            rebuild=args.rebuild_fetalclip_cache,
            crop_boxes=crop_boxes,
            crop_pad=args.crop_pad,
        )
        print(
            f"FetalCLIP embeddings: {len(embed_cache)} keys "
            f"(cache={args.fetalclip_cache.name} crop={args.clip_crop})"
        )
        if args.model == "m1":
            missing = sum(
                1 for sid, path in items
                if lookup_embedding(embed_cache, sample_id=sid, image_path=str(path)) is None
            )
            if missing and args.features_only:
                print(f"ERROR: M1 --features-only but {missing} frames lack FetalCLIP cache", file=sys.stderr)
                return 1
    elif args.frame_select.endswith("_gate"):
        embed_cache = {}

    print(
        f"MASVF-{args.model.upper()} cohort={args.cohort} frame_select={args.frame_select} "
        f"patient_agg={args.patient_agg} clip_crop={args.clip_crop} m0_features={args.m0_features}"
    )

    if args.cohort == "private":
        results = run_private(args, tags, cache, yolo_model, embed_cache)
    else:
        results = run_cardium(args, tags, cache, yolo_model, embed_cache)

    m0_names = resolve_m0_feature_names(args.m0_features)
    results["config"] = {
        "model": f"MASVF-{args.model.upper()}",
        "variant": args.model,
        "frame_select": args.frame_select,
        "k_per_view": args.k_per_view,
        "view_mode": args.view_mode,
        "patient_agg": args.patient_agg,
        "clip_crop": args.clip_crop,
        "crop_pad": args.crop_pad,
        "m0_features": args.m0_features,
        "m0_feature_dim": len(m0_names),
        "fetalclip_dim": FETALCLIP_EMBED_DIM if args.model == "m1" else 0,
        "feature_dim": len(m0_names) + (FETALCLIP_EMBED_DIM if args.model == "m1" else 0),
        "m0_feature_names": m0_names,
        "fetalclip_cache": str(args.fetalclip_cache),
        "image_tags": str(args.image_tags),
    }

    out_name = f"{args.cohort}_{args.model}"
    if args.model == "m1" and args.clip_crop == "plane":
        out_name += "_crop"
    if args.m0_features != "all":
        out_name += f"_{args.m0_features}"
    if args.patient_agg != "max":
        out_name += f"_{args.patient_agg}"
    out = args.output_dir / f"{out_name}_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved → {out}")
    if "comparison" in results:
        c = results["comparison"]
        print(
            f"POOLED patient F1={c.get('patient_f1_mean', float('nan')):.3f} "
            f"AUC={c.get('patient_auc_mean', float('nan')):.3f} "
            f"Sens={c.get('patient_sens_mean', float('nan')):.3f} "
            f"Spec={c.get('patient_spec_mean', float('nan')):.3f}"
        )
    elif "splits" in results and "test" in results["splits"]:
        p = results["splits"]["test"]["patient"]
        print(
            f"TEST patient F1={p['f1']:.3f} AUC={p['auc']:.3f} "
            f"Sens={p['sensitivity']:.3f} Spec={p['specificity']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
