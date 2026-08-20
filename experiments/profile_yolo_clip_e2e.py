#!/usr/bin/env python3
"""End-to-end YOLO + FetalCLIP wall-clock (ms/exam) for BSPC efficiency.

This is NOT fusion-head-only profiling. It measures:
  A) Full-frame FetalCLIP: encode every acquired frame
  B) Gated path: YOLO on every frame + FetalCLIP only on 4CH/LVOT/RVOT/3VT
  C) ALVG fusion head on 4 view tokens (usually <<1 ms; reported separately)

Usage (GPU box, from repo root):
  python -u experiments/profile_yolo_clip_e2e.py --device 0 --max-exams 32
  python -u experiments/profile_yolo_clip_e2e.py --device 0 --max-exams 64 --warmup 4

Outputs JSON with GPU name, CUDA, per-exam times, mean±std ms/exam, speedup.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

_OPEN_FAIL = (
    OSError,
    FileNotFoundError,
    ValueError,
    UnidentifiedImageError,
    Image.DecompressionBombError,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from agcd.cardium_views import (  # noqa: E402
    CARDIUM_STANDARD_VIEWS,
    dominant_plane_from_dets,
    map_to_cardium_view,
)
from agcd.fetalclip_embed import load_fetalclip_encoder  # noqa: E402
from agcd.view_anatomy_stats import view_anat_frame_dim  # noqa: E402
from yolo_io import resolve_weights  # noqa: E402
from view_graph_fusion import ViewGraphFusion  # noqa: E402

STANDARD = set(CARDIUM_STANDARD_VIEWS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--yolo-batch", type=int, default=16)
    p.add_argument("--clip-batch", type=int, default=16)
    p.add_argument("--max-exams", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/tertiary_e2e_latency.json",
    )
    p.add_argument("--max-frames-per-exam", type=int, default=0,
                   help="0=all frames; >0 caps for a smoke test")
    return p.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_mem_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def load_exams(manifest: Path, split: str, data_root: Path) -> list[dict]:
    exams: list[dict] = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if split != "all" and s.get("split") != split:
                continue
            rels = list(s.get("frame_paths") or [])
            paths: list[Path] = []
            for rel in rels:
                p = Path(rel)
                paths.append(p if p.is_absolute() else data_root / rel)
            exams.append(
                {
                    "study_id": str(s.get("study_id") or s.get("patient_id")),
                    "patient_id": str(s.get("patient_id") or ""),
                    "label": int(s.get("label_binary", 0)),
                    "paths": paths,
                }
            )
    return exams


def _take(xs: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    if k <= 0 or not xs:
        return []
    idx = rng.choice(len(xs), size=min(k, len(xs)), replace=False)
    return [xs[int(i)] for i in np.atleast_1d(idx)]


def sample_exams(exams: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pos = [e for e in exams if e["label"] == 1 and e["paths"]]
    neg = [e for e in exams if e["label"] == 0 and e["paths"]]
    n_pos = min(len(pos), max(1, n // 4)) if pos else 0
    n_neg = min(len(neg), max(0, n - n_pos))
    pick = _take(pos, n_pos, rng) + _take(neg, n_neg, rng)
    order = rng.permutation(len(pick))
    return [pick[int(i)] for i in order][:n]


def try_load_rgb(path: Path) -> np.ndarray | None:
    """Return HxWx3 uint8, or None if PIL cannot decode the file."""
    try:
        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert("RGB"))
    except _OPEN_FAIL:
        return None


def _dets_from_result(res) -> list[tuple[int, float]]:
    dets: list[tuple[int, float]] = []
    if res.boxes is not None and len(res.boxes):
        cls = res.boxes.cls.detach().cpu().tolist()
        cfs = res.boxes.conf.detach().cpu().tolist()
        dets = [(int(c), float(cf)) for c, cf in zip(cls, cfs)]
    return dets


def _view_from_result(res, conf: float) -> str | None:
    pa = dominant_plane_from_dets(_dets_from_result(res), min_conf=conf)
    return map_to_cardium_view(pa.plane_code) if pa else None


def yolo_tag_arrays(
    model,
    images: list[np.ndarray],
    *,
    conf: float,
    imgsz: int,
    device: str,
    batch_size: int,
) -> list[str | None]:
    """Run YOLO on already-decoded RGB arrays so Ultralytics never opens bad JPEGs."""
    views: list[str | None] = [None] * len(images)
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        try:
            results = model.predict(
                batch, conf=conf, imgsz=imgsz, device=device, verbose=False,
            )
        except Exception as exc:
            print(f"  WARN YOLO batch failed ({exc}); retry one-by-one", flush=True)
            results = []
            for img in batch:
                try:
                    results.extend(
                        model.predict(
                            img, conf=conf, imgsz=imgsz, device=device, verbose=False,
                        )
                    )
                except Exception as exc2:
                    print(f"  SKIP YOLO frame: {exc2}", flush=True)
                    results.append(None)
        for j, res in enumerate(results):
            if res is None:
                continue
            views[i + j] = _view_from_result(res, conf)
    return views


@torch.no_grad()
def clip_encode_paths(
    paths: list[Path],
    *,
    encoder,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> int:
    n_ok = 0
    encoder.eval()
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        tensors = []
        for p in chunk:
            try:
                img = Image.open(p).convert("RGB")
            except _OPEN_FAIL:
                continue
            tensors.append(preprocess(img))
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device, non_blocking=True)
        feats = encoder(batch)
        if feats.dim() > 2:
            feats = feats.mean(dim=tuple(range(1, feats.dim() - 1)))
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        n_ok += int(feats.shape[0])
        del batch, feats
    return n_ok


def device_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "device": str(device),
        "gpu_name": None,
        "gpu_mem_total_mb": None,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        info["gpu_mem_total_mb"] = round(props.total_memory / (1024 ** 2), 1)
    return info


def mean_std_ms(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean_ms": None, "std_ms": None, "p50_ms": None}
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50_ms": float(np.median(a)),
    }


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"ERROR missing {args.manifest}")
        return 1

    cuda_ok = torch.cuda.is_available()
    if str(args.device).isdigit() and cuda_ok:
        device = torch.device(f"cuda:{args.device}")
        yolo_dev = str(args.device)
    elif cuda_ok and str(args.device).startswith("cuda"):
        device = torch.device(args.device)
        yolo_dev = str(device).replace("cuda:", "") or "0"
    else:
        device = torch.device("cpu")
        yolo_dev = "cpu"
        print("WARN CUDA not available — times will be CPU and not paper-usable")

    exams = load_exams(args.manifest, args.split, args.data_root)
    picked = sample_exams(exams, args.max_exams + args.warmup, args.seed)
    if not picked:
        print("ERROR no exams sampled")
        return 1

    weights = resolve_weights(args.yolo_weights)
    from ultralytics import YOLO

    print(f"YOLO weights={weights}")
    yolo = YOLO(str(weights))
    print("loading FetalCLIP…")
    encoder, preprocess = load_fetalclip_encoder()
    encoder = encoder.to(device)
    encoder.eval()

    d_in = 768 + int(view_anat_frame_dim())
    fusion = ViewGraphFusion(
        d_in=d_in, d_model=64, n_layers=1, dropout=0.0, n_views=4, n_m0=0,
        use_anatomy_adj=True,
    ).to(device).eval()
    dummy_tok = torch.randn(1, 4, d_in, device=device)
    dummy_pres = torch.ones(1, 4, dtype=torch.bool, device=device)

    info = device_info(device)
    info["yolo_weights"] = str(weights)
    print(json.dumps(info, indent=2))

    per_exam: list[dict] = []
    timed: list[dict] = []
    n_skip_total = 0

    for ei, exam in enumerate(picked):
        paths = [p for p in exam["paths"] if p.is_file()]
        if args.max_frames_per_exam > 0:
            paths = paths[: args.max_frames_per_exam]
        if not paths:
            continue
        is_warm = ei < args.warmup

        ok_paths: list[Path] = []
        ok_arrs: list[np.ndarray] = []
        skipped: list[str] = []
        for p in paths:
            arr = try_load_rgb(p)
            if arr is None:
                skipped.append(str(p))
                continue
            ok_paths.append(p)
            ok_arrs.append(arr)
        n_skip_total += len(skipped)
        if skipped:
            print(
                f"  SKIP {len(skipped)} unreadable frame(s) sid={exam['study_id']} "
                f"e.g. {skipped[0]}",
                flush=True,
            )
        if not ok_paths:
            print(f"  SKIP exam (no readable frames) sid={exam['study_id']}", flush=True)
            continue

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        _sync(device)
        t0 = time.perf_counter()
        views = yolo_tag_arrays(
            yolo, ok_arrs,
            conf=args.conf, imgsz=args.imgsz, device=yolo_dev,
            batch_size=args.yolo_batch,
        )
        _sync(device)
        yolo_ms = 1000.0 * (time.perf_counter() - t0)
        yolo_mem = _gpu_mem_mb(device)

        gated = [p for p, v in zip(ok_paths, views) if v in STANDARD]
        n_std = len(gated)
        paths = ok_paths

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t1 = time.perf_counter()
        n_full_ok = clip_encode_paths(
            paths, encoder=encoder, preprocess=preprocess,
            device=device, batch_size=args.clip_batch,
        )
        _sync(device)
        clip_full_ms = 1000.0 * (time.perf_counter() - t1)
        clip_full_mem = _gpu_mem_mb(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        t2 = time.perf_counter()
        n_g_ok = clip_encode_paths(
            gated, encoder=encoder, preprocess=preprocess,
            device=device, batch_size=args.clip_batch,
        ) if gated else 0
        _sync(device)
        clip_gated_ms = 1000.0 * (time.perf_counter() - t2)
        clip_gated_mem = _gpu_mem_mb(device)

        _sync(device)
        t3 = time.perf_counter()
        with torch.no_grad():
            _ = fusion(dummy_tok, dummy_pres)
        _sync(device)
        fusion_ms = 1000.0 * (time.perf_counter() - t3)

        rec = {
            "study_id": exam["study_id"],
            "label": exam["label"],
            "warmup": is_warm,
            "n_frames": len(paths),
            "n_skipped_unreadable": len(skipped),
            "n_std_frames": n_std,
            "n_clip_full_ok": n_full_ok,
            "n_clip_gated_ok": n_g_ok,
            "yolo_ms": yolo_ms,
            "clip_full_ms": clip_full_ms,
            "clip_gated_ms": clip_gated_ms,
            "fusion_ms": fusion_ms,
            "pipeline_full_ms": clip_full_ms,
            "pipeline_gated_ms": yolo_ms + clip_gated_ms + fusion_ms,
            "gpu_mem_mb_yolo": yolo_mem,
            "gpu_mem_mb_clip_full": clip_full_mem,
            "gpu_mem_mb_clip_gated": clip_gated_mem,
        }
        per_exam.append(rec)
        tag = "WARM" if is_warm else "TIME"
        print(
            f"[{tag} {ei+1}/{len(picked)}] sid={exam['study_id']} "
            f"frames={len(paths)} std={n_std} "
            f"yolo={yolo_ms:.0f}ms clip_full={clip_full_ms:.0f}ms "
            f"clip_gated={clip_gated_ms:.0f}ms "
            f"gated_pipe={rec['pipeline_gated_ms']:.0f}ms",
            flush=True,
        )
        if not is_warm:
            timed.append(rec)

    if not timed:
        print("ERROR no timed exams (increase --max-exams)")
        return 1

    full_ms = [r["pipeline_full_ms"] for r in timed]
    gated_ms = [r["pipeline_gated_ms"] for r in timed]
    yolo_ms = [r["yolo_ms"] for r in timed]
    clip_g = [r["clip_gated_ms"] for r in timed]
    n_frames = [r["n_frames"] for r in timed]
    n_std = [r["n_std_frames"] for r in timed]
    speedups = [
        a / b for a, b in zip(full_ms, gated_ms) if b and b > 0
    ]
    out = {
        "device": info,
        "protocol": {
            "manifest": str(args.manifest),
            "split": args.split,
            "n_pool_exams": len(exams),
            "n_timed": len(timed),
            "warmup": args.warmup,
            "max_exams": args.max_exams,
            "seed": args.seed,
            "yolo_conf": args.conf,
            "imgsz": args.imgsz,
            "yolo_batch": args.yolo_batch,
            "clip_batch": args.clip_batch,
            "n_skipped_unreadable": int(n_skip_total),
            "note": (
                "pipeline_full = FetalCLIP on readable frames (no YOLO). "
                "pipeline_gated = YOLO(readable frames) + FetalCLIP(std-view) + ALVG dummy. "
                "Unreadable JPEGs are skipped before Ultralytics (PIL UnidentifiedImageError). "
                "Do not use fusion-head-only TABLE_efficiency.json as BSPC wall-clock."
            ),
        },
        "frames": {
            "mean_acquired": float(np.mean(n_frames)),
            "mean_std": float(np.mean(n_std)),
            "std_frac": float(np.sum(n_std) / max(np.sum(n_frames), 1)),
        },
        "ms_per_exam": {
            "yolo": mean_std_ms(yolo_ms),
            "clip_full": mean_std_ms([r["clip_full_ms"] for r in timed]),
            "clip_gated": mean_std_ms(clip_g),
            "fusion_dummy": mean_std_ms([r["fusion_ms"] for r in timed]),
            "pipeline_full_clip": mean_std_ms(full_ms),
            "pipeline_gated_yolo_clip_alvg": mean_std_ms(gated_ms),
        },
        "speedup_full_over_gated": {
            "mean": float(np.mean(speedups)) if speedups else None,
            "std": float(np.std(speedups, ddof=1)) if len(speedups) > 1 else 0.0,
        },
        "peak_gpu_mem_mb": {
            "yolo_mean": float(np.nanmean([r["gpu_mem_mb_yolo"] or np.nan for r in timed])),
            "clip_full_mean": float(np.nanmean([r["gpu_mem_mb_clip_full"] or np.nan for r in timed])),
            "clip_gated_mean": float(np.nanmean([r["gpu_mem_mb_clip_gated"] or np.nan for r in timed])),
        },
        "per_exam": per_exam,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    pf = out["ms_per_exam"]["pipeline_full_clip"]
    pg = out["ms_per_exam"]["pipeline_gated_yolo_clip_alvg"]
    print("\n========== E2E LATENCY ==========")
    print(f"GPU: {info.get('gpu_name')}  torch={info.get('torch')}")
    print(
        f"timed exams={len(timed)}  mean frames={out['frames']['mean_acquired']:.1f} "
        f"mean std={out['frames']['mean_std']:.1f} "
        f"std_frac={out['frames']['std_frac']:.3f}  "
        f"skipped_unreadable={n_skip_total}"
    )
    print(f"full-frame CLIP:     {pf['mean_ms']:.0f} ± {pf['std_ms']:.0f} ms/exam")
    print(f"YOLO:                {out['ms_per_exam']['yolo']['mean_ms']:.0f} ± "
          f"{out['ms_per_exam']['yolo']['std_ms']:.0f} ms/exam")
    print(f"gated CLIP:          {out['ms_per_exam']['clip_gated']['mean_ms']:.0f} ± "
          f"{out['ms_per_exam']['clip_gated']['std_ms']:.0f} ms/exam")
    print(f"gated YOLO+CLIP+ALVG:{pg['mean_ms']:.0f} ± {pg['std_ms']:.0f} ms/exam")
    print(f"speedup full/gated:  {out['speedup_full_over_gated']['mean']:.2f}×")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
