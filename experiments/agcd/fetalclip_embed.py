"""FetalCLIP image embeddings for frame diversity (MMR) selection and M1 features."""

from __future__ import annotations

import io
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from agcd.frame_quality import crop_pil_by_xyxy_norm  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "fetalclip" / "FetalCLIP_config.json"
DEFAULT_WEIGHTS = PROJECT_ROOT / "fetalclip" / "FetalCLIP_weights.pt"
DEFAULT_CACHE = PROJECT_ROOT / "data" / "study_screening" / "cardium_fetalclip_embed_cache.jsonl"
PRIVATE_CACHE = PROJECT_ROOT / "data" / "study_screening" / "masvf_private_fetalclip_embed_cache.jsonl"
CARDIUM_CROP_CACHE = PROJECT_ROOT / "data" / "study_screening" / "cardium_fetalclip_embed_cache_plane_crop.jsonl"
PRIVATE_CROP_CACHE = PROJECT_ROOT / "data" / "study_screening" / "masvf_private_fetalclip_embed_cache_plane_crop.jsonl"
PRIVATE_CROP_LORA_CACHE = (
    PROJECT_ROOT / "data" / "study_screening" / "masvf_private_fetalclip_embed_cache_plane_crop_homologous_lora.jsonl"
)
CARDIUM_CROP_LORA_CACHE = (
    PROJECT_ROOT / "data" / "study_screening" / "cardium_fetalclip_embed_cache_plane_crop_homologous_lora.jsonl"
)
FETALCLIP_EMBED_DIM = 768
DEFAULT_CROP_PAD = 0.12
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET = "mlp.c_proj"


def lookup_embedding(
    cache: dict[str, np.ndarray],
    *,
    sample_id: str,
    image_path: str,
) -> np.ndarray | None:
    """Resolve embedding by sample_id or absolute image path."""
    if sample_id in cache:
        return cache[sample_id]
    if image_path in cache:
        return cache[image_path]
    return None


def resolve_fetalclip_paths(
    config: Path | None = None,
    weights: Path | None = None,
    fetalclip_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    fdir = fetalclip_dir or PROJECT_ROOT / "fetalclip"
    cfg = config or fdir / "FetalCLIP_config.json"
    wts = weights or fdir / "FetalCLIP_weights.pt"
    return cfg, wts, fdir


def load_embed_cache(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not path.is_file():
        return out
    print(f"  loading embed cache {path} ...", flush=True)
    n_lines = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "embedding" not in row:
                continue
            vec = np.asarray(row["embedding"], dtype=np.float32)
            # Index both identities. Patient regrouping changes sample_id while
            # the physical image path stays unchanged, so path lookup safely
            # reuses the same image embedding after rebuilding the manifest.
            if row.get("sample_id"):
                out[row["sample_id"]] = vec
            if row.get("image_path"):
                out[row["image_path"]] = vec
            n_lines += 1
            if n_lines % 50000 == 0:
                print(f"    cache rows={n_lines} keys={len(out)}", flush=True)
    print(f"  loaded embed cache rows={n_lines} keys={len(out)}", flush=True)
    return out


def append_embed_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_image_for_encode(
    path: Path,
    *,
    xyxy_norm: tuple[float, float, float, float] | None,
    crop_pad: float,
    grayscale: bool = False,
    remove_non_rb_color: bool = False,
    canon_size: int = 0,
    canon_jpeg_quality: int = 0,
    intensity_normalize: bool = False,
    hist_equalize: bool = False,
    fixed_crop_frac: float = 0.0,
    gaussian_blur_radius: float = 0.0,
    patch_shuffle_grid: int = 0,
    center_mask_frac: float = 0.0,
    frequency_transform: str = "none",
) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    if xyxy_norm is not None and fixed_crop_frac > 0:
        xyxy_norm = fixed_square_crop_box(xyxy_norm, fixed_crop_frac)
    if xyxy_norm is not None:
        img = crop_pil_by_xyxy_norm(img, xyxy_norm, pad=crop_pad)
    if canon_size or canon_jpeg_quality:
        img = canonicalize_acquisition(
            img, size=canon_size, jpeg_quality=canon_jpeg_quality,
        )
    if grayscale:
        # Preserve the visual encoder's expected three-channel input while
        # removing hue/saturation and color-Doppler shortcuts.
        img = ImageOps.grayscale(img).convert("RGB")
    elif remove_non_rb_color:
        img = remove_colored_annotations_keep_red_blue(img)
    if intensity_normalize:
        img = percentile_intensity_normalize(img)
    if hist_equalize:
        img = ImageOps.equalize(img)
    if gaussian_blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=gaussian_blur_radius))
    if patch_shuffle_grid > 1:
        img = deterministic_patch_shuffle(img, patch_shuffle_grid, str(path))
    if center_mask_frac > 0:
        img = mask_image_center(img, center_mask_frac)
    if frequency_transform != "none":
        img = apply_frequency_control(img, frequency_transform, str(path))
    return img


def fixed_square_crop_box(
    xyxy: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Fixed-size square centered on the detected plane.

    Variable YOLO box area is itself highly label-correlated. This keeps the
    detector's location but removes its width/height as an input cue.
    """
    side = min(max(float(fraction), 0.05), 1.0)
    cx = (float(xyxy[0]) + float(xyxy[2])) / 2.0
    cy = (float(xyxy[1]) + float(xyxy[3])) / 2.0
    x0 = min(max(cx - side / 2.0, 0.0), 1.0 - side)
    y0 = min(max(cy - side / 2.0, 0.0), 1.0 - side)
    return x0, y0, x0 + side, y0 + side


def percentile_intensity_normalize(img: Image.Image) -> Image.Image:
    """Map each image's luminance p1/p99 to 0/255 while preserving hue."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    lo, hi = np.percentile(gray, (1.0, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1.0:
        return img
    out = np.clip((rgb - lo) * (255.0 / (hi - lo)), 0.0, 255.0)
    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def deterministic_patch_shuffle(img: Image.Image, grid: int, key: str) -> Image.Image:
    """Destroy global anatomy while retaining local texture and histograms."""
    grid = max(2, int(grid))
    rgb = img.convert("RGB")
    width, height = rgb.size
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    patches = [
        rgb.crop((x_edges[x], y_edges[y], x_edges[x + 1], y_edges[y + 1]))
        for y in range(grid)
        for x in range(grid)
    ]
    seed = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:16], 16)
    order = np.random.default_rng(seed).permutation(len(patches))
    out = Image.new("RGB", rgb.size)
    for dst, src in enumerate(order):
        y, x = divmod(dst, grid)
        patch = patches[int(src)].resize(
            (x_edges[x + 1] - x_edges[x], y_edges[y + 1] - y_edges[y]),
            Image.Resampling.BILINEAR,
        )
        out.paste(patch, (x_edges[x], y_edges[y]))
    return out


def mask_image_center(img: Image.Image, fraction: float) -> Image.Image:
    """Remove central anatomy, leaving only the peripheral acquisition texture."""
    fraction = min(max(float(fraction), 0.05), 0.95)
    out = img.convert("RGB").copy()
    width, height = out.size
    half_w = int(width * fraction / 2.0)
    half_h = int(height * fraction / 2.0)
    cx, cy = width // 2, height // 2
    fill = tuple(
        int(v) for v in np.median(np.asarray(out), axis=(0, 1)).tolist()
    )
    block = Image.new("RGB", (2 * half_w, 2 * half_h), fill)
    out.paste(block, (cx - half_w, cy - half_h))
    return out


def apply_frequency_control(img: Image.Image, mode: str, key: str) -> Image.Image:
    """Anatomy-destruction controls in the frequency/pixel domain.

    phase_randomized:
        Keep each channel's Fourier magnitude approximately intact while
        replacing phase, which destroys spatial anatomy but preserves spectrum.
    pixel_shuffle:
        Preserve the exact RGB histogram but destroy all spatial organization.
    fft_magnitude:
        Render only the log Fourier magnitude; phase/anatomy is absent.
    highpass:
        Keep fine texture/speckle residual and remove coarse cardiac structure.
    """
    valid = {"phase_randomized", "pixel_shuffle", "fft_magnitude", "highpass"}
    if mode not in valid:
        raise ValueError(f"unknown frequency_transform={mode!r}; choose {sorted(valid)}")

    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    seed = int(
        hashlib.sha1(f"{key}|{mode}".encode("utf-8")).hexdigest()[:16],
        16,
    )
    rng = np.random.default_rng(seed)

    if mode == "pixel_shuffle":
        flat = rgb.reshape(-1, 3)
        order = rng.permutation(len(flat))
        return Image.fromarray(flat[order].reshape(rgb.shape).astype(np.uint8))

    if mode == "fft_magnitude":
        gray = rgb.mean(axis=2)
        magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        lo, hi = np.percentile(magnitude, (1.0, 99.8))
        magnitude = np.clip((magnitude - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255)
        out = np.repeat(magnitude[..., None], 3, axis=2)
        return Image.fromarray(out.astype(np.uint8))

    if mode == "highpass":
        low = np.asarray(
            img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=6.0)),
            dtype=np.float32,
        )
        residual = rgb - low
        scale = float(np.percentile(np.abs(residual), 99.0))
        out = np.clip(127.5 + residual * (127.5 / max(scale, 1.0)), 0, 255)
        return Image.fromarray(out.astype(np.uint8))

    # Phase controls spatial position. Operate on luminance and repeat it over
    # channels so independent random phases cannot create artificial colours.
    plane = rgb.mean(axis=2)
    spectrum = np.fft.rfft2(plane)
    magnitude = np.abs(spectrum)
    phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    randomized = magnitude * np.exp(1j * phase)
    randomized[0, 0] = spectrum[0, 0]
    reconstructed = np.fft.irfft2(randomized, s=plane.shape).real
    rec_std = float(reconstructed.std())
    if rec_std > 1e-6:
        reconstructed = (
            (reconstructed - reconstructed.mean())
            * (float(plane.std()) / rec_std)
            + float(plane.mean())
        )
    reconstructed = np.clip(reconstructed, 0, 255)
    out = np.repeat(reconstructed[..., None], 3, axis=2)
    return Image.fromarray(out.astype(np.uint8))


def canonicalize_acquisition(
    img: Image.Image,
    *,
    size: int = 0,
    jpeg_quality: int = 0,
) -> Image.Image:
    """Erase export-pipeline identity: one resolution, one compression level.

    Cohorts archived years apart differ in stored resolution and JPEG quality,
    which a linear probe reads off directly. Squashing both to a single canonical
    size and re-encoding at a fixed quality removes that channel while leaving
    anatomy intact, so it isolates whether a margin is anatomical.
    """
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    if jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(jpeg_quality))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def remove_colored_annotations_keep_red_blue(img: Image.Image) -> Image.Image:
    """Inpaint saturated non-red/non-blue overlays from an ultrasound frame.

    Keeps grayscale B-mode tissue and red/blue Doppler. Removes saturated
    yellow/green/cyan/magenta calipers, measurement lines and text, then fills
    their pixels from the local neighborhood. This is deliberately an ablation:
    some scanners use wider Doppler palettes, so original images are untouched.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "remove_non_rb_color requires opencv-python (cv2)"
        ) from exc

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    colorful = (sat >= 70) & (val >= 50)
    # OpenCV hue ∈ [0,179]. Preserve true red and blue Doppler families.
    keep_red = (hue <= 12) | (hue >= 168)
    keep_blue = (hue >= 90) & (hue <= 135)
    mask = (colorful & ~(keep_red | keep_blue)).astype(np.uint8) * 255
    if not mask.any():
        return Image.fromarray(rgb)

    # Cover antialiased edges of thin text/calipers before local inpainting.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cleaned = cv2.inpaint(rgb, mask, 3.0, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)


@torch.no_grad()
def encode_images(
    paths: list[Path],
    *,
    encoder,
    preprocess,
    device: str = "cuda",
    batch_size: int = 32,
    crop_boxes: list[tuple[float, float, float, float] | None] | None = None,
    crop_pad: float = DEFAULT_CROP_PAD,
    grayscale: bool = False,
    remove_non_rb_color: bool = False,
    canon_size: int = 0,
    canon_jpeg_quality: int = 0,
    intensity_normalize: bool = False,
    hist_equalize: bool = False,
    fixed_crop_frac: float = 0.0,
    gaussian_blur_radius: float = 0.0,
    patch_shuffle_grid: int = 0,
    center_mask_frac: float = 0.0,
    frequency_transform: str = "none",
) -> dict[str, np.ndarray]:
    """Return image_path str -> L2-normalized embedding.

    If ``crop_boxes`` is provided (same length as paths), crop with padding
    before preprocess; None entry → full image fallback.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(dev)
    encoder.eval()
    out: dict[str, np.ndarray] = {}
    n_crop = 0
    n_full = 0
    n_skip = 0
    started = time.monotonic()
    total_batches = (len(paths) + batch_size - 1) // batch_size
    report_every = max(1, min(10, total_batches))

    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        box_chunk = (
            crop_boxes[i : i + batch_size]
            if crop_boxes is not None
            else [None] * len(chunk)
        )
        tensors = []
        valid: list[Path] = []
        for p, box in zip(chunk, box_chunk):
            # No Path.is_file() — NFS stall; _load_image_for_encode returns None on failure.
            img = _load_image_for_encode(
                p,
                xyxy_norm=box,
                crop_pad=crop_pad,
                grayscale=grayscale,
                remove_non_rb_color=remove_non_rb_color,
                canon_size=canon_size,
                canon_jpeg_quality=canon_jpeg_quality,
                intensity_normalize=intensity_normalize,
                hist_equalize=hist_equalize,
                fixed_crop_frac=fixed_crop_frac,
                gaussian_blur_radius=gaussian_blur_radius,
                patch_shuffle_grid=patch_shuffle_grid,
                center_mask_frac=center_mask_frac,
                frequency_transform=frequency_transform,
            )
            if img is None:
                n_skip += 1
                continue
            if box is not None:
                n_crop += 1
            else:
                n_full += 1
            try:
                tensors.append(preprocess(img))
                valid.append(p)
            except (OSError, ValueError):
                n_skip += 1
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(dev)
        feats = encoder(batch)
        if feats.dim() > 2:
            feats = feats.mean(dim=tuple(range(1, feats.dim() - 1)))
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for p, vec in zip(valid, feats.cpu().numpy()):
            out[str(p)] = vec.astype(np.float32)
        done = min(i + batch_size, len(paths))
        batch_index = i // batch_size + 1
        if batch_index % report_every == 0 or done == len(paths):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = done / elapsed
            eta = (len(paths) - done) / rate if rate > 0 else float("inf")
            print(
                f"  encode progress {done}/{len(paths)} "
                f"({100.0 * done / max(len(paths), 1):.1f}%) "
                f"rate={rate:.1f} img/s ETA={eta / 60.0:.1f} min",
                flush=True,
            )

    if crop_boxes is not None:
        print(f"  FetalCLIP crops applied={n_crop} full_fallback={n_full}")
    return out


def load_fetalclip_encoder(
    config: Path | None = None,
    weights: Path | None = None,
    fetalclip_dir: Path | None = None,
    *,
    lora_adapter: Path | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target: str = DEFAULT_LORA_TARGET,
):
    """Load FetalCLIP visual encoder; optionally inject LoRA and load adapter weights.

    Adapter ckpt may be:
      - dict with key ``adapter_state`` (from homologous / private LoRA trainers)
      - raw state_dict containing ``lora_a`` / ``lora_b`` keys
    Head weights (``head.*``) are ignored for embedding extraction.
    """
    from chd_baseline.lora import inject_lora
    from chd_baseline.models import load_fetalclip_visual

    cfg, wts, fdir = resolve_fetalclip_paths(config, weights, fetalclip_dir)
    if not cfg.is_file() or not wts.is_file():
        raise FileNotFoundError(f"FetalCLIP not found: {cfg} / {wts}")
    encoder, preprocess = load_fetalclip_visual(str(cfg), str(wts), str(fdir))
    if lora_adapter is None:
        return encoder, preprocess

    adapter_path = Path(lora_adapter)
    if not adapter_path.is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    blob = torch.load(adapter_path, map_location="cpu", weights_only=False)
    meta = blob.get("args") if isinstance(blob, dict) else None
    if isinstance(meta, dict):
        lora_r = int(meta.get("lora_r", lora_r))
        lora_alpha = int(meta.get("lora_alpha", lora_alpha))
        lora_dropout = float(meta.get("lora_dropout", lora_dropout))
        lora_target = str(meta.get("lora_target", lora_target))
    inject_lora(
        encoder,
        target_pattern=lora_target,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    state = blob["adapter_state"] if isinstance(blob, dict) and "adapter_state" in blob else blob
    if not isinstance(state, dict):
        raise ValueError(f"unexpected LoRA adapter format: {adapter_path}")
    # Keep only encoder LoRA tensors (drop classification head).
    cleaned = {
        k: v
        for k, v in state.items()
        if ("lora_a" in k or "lora_b" in k) and not k.startswith("head.")
    }
    # Trainer saves keys under FetalCLIPClassifier ("encoder.xxx"); strip prefix.
    remapped = {}
    for k, v in cleaned.items():
        nk = k[len("encoder.") :] if k.startswith("encoder.") else k
        remapped[nk] = v
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    n_lora = sum(1 for k in remapped if "lora_" in k)
    print(
        f"  LoRA adapter={adapter_path.name} loaded_tensors={n_lora} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"(r={lora_r} alpha={lora_alpha} target={lora_target})"
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, preprocess


def build_or_load_embeddings(
    items: list[tuple[str, Path]],
    *,
    cache_path: Path = DEFAULT_CACHE,
    device: str = "0",
    batch_size: int = 32,
    rebuild: bool = False,
    crop_boxes: dict[str, tuple[float, float, float, float] | None] | None = None,
    crop_pad: float = DEFAULT_CROP_PAD,
    grayscale: bool = False,
    remove_non_rb_color: bool = False,
    canon_size: int = 0,
    canon_jpeg_quality: int = 0,
    intensity_normalize: bool = False,
    hist_equalize: bool = False,
    fixed_crop_frac: float = 0.0,
    gaussian_blur_radius: float = 0.0,
    patch_shuffle_grid: int = 0,
    center_mask_frac: float = 0.0,
    frequency_transform: str = "none",
    lora_adapter: Path | None = None,
    seed_caches: list[Path] | None = None,
) -> dict[str, np.ndarray]:
    """items: (sample_id, image_path). Cache keyed by sample_id and image_path.

    ``crop_boxes`` maps sample_id → normalized xyxy (or None for full-image).
    ``lora_adapter``: optional homologous/private LoRA ckpt for domain-adapted embeds.
    ``seed_caches``: extra read-only jsonl caches merged in-memory (path/sample_id hit);
    only newly encoded rows are appended to ``cache_path``.
    """
    cache = {} if rebuild else load_embed_cache(cache_path)
    if not rebuild and seed_caches:
        for sp in seed_caches:
            if not sp or not Path(sp).is_file():
                continue
            if Path(sp).resolve() == Path(cache_path).resolve():
                continue
            before = len(cache)
            seeded = load_embed_cache(Path(sp))
            # Prefer existing primary keys; fill holes from seed.
            for k, v in seeded.items():
                if k not in cache:
                    cache[k] = v
            print(
                f"  seed-cache {Path(sp).name}: +{len(cache) - before} keys "
                f"(file_keys≈{len(seeded)}, merged={len(cache)})"
            )
    to_run: list[tuple[str, Path]] = []
    for index, (sid, path) in enumerate(items, start=1):
        if sid in cache or str(path) in cache:
            continue
        to_run.append((sid, path))
        if index % 100000 == 0:
            print(
                f"  cache-miss scan {index}/{len(items)} pending={len(to_run)}",
                flush=True,
            )

    if to_run:
        mode = "plane-crop" if crop_boxes is not None else "full-image"
        if grayscale:
            mode += "+grayscale"
        elif remove_non_rb_color:
            mode += "+remove-non-rb-color"
        if canon_size or canon_jpeg_quality:
            mode += f"+canon{canon_size or 'x'}q{canon_jpeg_quality or 'x'}"
        if intensity_normalize:
            mode += "+pctl-norm"
        if hist_equalize:
            mode += "+hist-eq"
        if fixed_crop_frac:
            mode += f"+fixed-fov{fixed_crop_frac:g}"
        if gaussian_blur_radius:
            mode += f"+blur{gaussian_blur_radius:g}"
        if patch_shuffle_grid:
            mode += f"+shuffle{patch_shuffle_grid}"
        if center_mask_frac:
            mode += f"+mask{center_mask_frac:g}"
        if frequency_transform != "none":
            mode += f"+{frequency_transform}"
        if lora_adapter is not None:
            mode += "+homologous-lora"
        print(f"FetalCLIP encoding {len(to_run)} images mode={mode} (cached={len(cache)})")
        encoder, preprocess = load_fetalclip_encoder(lora_adapter=lora_adapter)
        dev = f"cuda:{device}" if device.isdigit() else device
        paths = [p for _, p in to_run]
        boxes = None
        if crop_boxes is not None:
            boxes = [crop_boxes.get(sid) for sid, _ in to_run]
        vecs = encode_images(
            paths,
            encoder=encoder,
            preprocess=preprocess,
            device=dev,
            batch_size=batch_size,
            crop_boxes=boxes,
            crop_pad=crop_pad,
            grayscale=grayscale,
            remove_non_rb_color=remove_non_rb_color,
            canon_size=canon_size,
            canon_jpeg_quality=canon_jpeg_quality,
            intensity_normalize=intensity_normalize,
            hist_equalize=hist_equalize,
            fixed_crop_frac=fixed_crop_frac,
            gaussian_blur_radius=gaussian_blur_radius,
            patch_shuffle_grid=patch_shuffle_grid,
            center_mask_frac=center_mask_frac,
            frequency_transform=frequency_transform,
        )
        new_rows: list[dict] = []
        for sid, path in to_run:
            vec = vecs.get(str(path))
            if vec is None:
                continue
            cache[sid] = vec
            cache[str(path)] = vec
            row: dict = {
                "sample_id": sid,
                "image_path": str(path),
                "embedding": vec.tolist(),
                "crop": "plane" if (crop_boxes and crop_boxes.get(sid)) else "full",
                "grayscale": grayscale,
                "remove_non_rb_color": remove_non_rb_color,
                "canon_size": canon_size,
                "canon_jpeg_quality": canon_jpeg_quality,
                "intensity_normalize": intensity_normalize,
                "hist_equalize": hist_equalize,
                "fixed_crop_frac": fixed_crop_frac,
                "gaussian_blur_radius": gaussian_blur_radius,
                "patch_shuffle_grid": patch_shuffle_grid,
                "center_mask_frac": center_mask_frac,
                "frequency_transform": frequency_transform,
                "lora_adapter": str(lora_adapter) if lora_adapter else None,
            }
            if crop_boxes and crop_boxes.get(sid):
                row["plane_xyxy"] = list(crop_boxes[sid])  # type: ignore[arg-type]
            new_rows.append(row)
        if new_rows and not rebuild:
            append_embed_cache(cache_path, new_rows)
        elif rebuild and cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            seen: set[str] = set()
            with cache_path.open("w", encoding="utf-8") as f:
                for sid, path in items:
                    if sid in seen:
                        continue
                    vec = cache.get(sid)
                    if vec is not None:
                        f.write(json.dumps({
                            "sample_id": sid,
                            "image_path": str(path),
                            "embedding": vec.tolist(),
                            "crop": "plane" if (crop_boxes and crop_boxes.get(sid)) else "full",
                            "grayscale": grayscale,
                            "remove_non_rb_color": remove_non_rb_color,
                            "lora_adapter": str(lora_adapter) if lora_adapter else None,
                        }, ensure_ascii=False) + "\n")
                        seen.add(sid)
    return cache
