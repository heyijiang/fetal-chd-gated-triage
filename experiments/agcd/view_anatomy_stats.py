"""Per-view compact anatomy *statistics* (no ROI FetalCLIP).

Each CARDIUM view gets its own short stats vector (same dim pad, only active
view filled). Designed to be denser / more view-specific than the previous
74-d geo slots that diluted C2-b.

Per structure (among that view's keys): area/plane_area, conf
Plus view-level: n_present/n_keys, mean_conf, plane_area, and 2–3 area ratios.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from atvs.plane_anatomy import PLANE_KEY_ANATOMY  # noqa: E402

from agcd.anatomy_boxes import boxes_from_tag  # noqa: E402
from agcd.cardium_views import CARDIUM_STANDARD_VIEWS  # noqa: E402

VIEW_KEYS: dict[str, tuple[int, ...]] = {
    "four_chamber": tuple(dict.fromkeys((*PLANE_KEY_ANATOMY["four_chamber"], 27))),
    "lvot": PLANE_KEY_ANATOMY["lvot"],
    "rvot": PLANE_KEY_ANATOMY["rvot"],
    "vvt": PLANE_KEY_ANATOMY["vvt"],
}

# (id_a, id_b) area ratios per view
VIEW_RATIOS: dict[str, tuple[tuple[int, int], ...]] = {
    "four_chamber": ((24, 21), (26, 11), (25, 24)),  # LV/RV, LA/RA, IVS/LV
    "lvot": ((17, 24), (32, 17)),  # AO/LV, LAAO/AO
    "rvot": ((29, 21), (29, 17)),  # MPA/RV, MPA/AO
    "vvt": ((2, 29), (2, 6), (7, 2)),  # AAO/MPA, AAO/SVC, T/AAO
}

# Fixed per-view dim: for each key → (area_rel, conf) + 3 ratios + 3 summary
# Use max keys (=6 for 4C) so pad shorter views
MAX_KEYS = max(len(v) for v in VIEW_KEYS.values())
MAX_RATIOS = max(len(v) for v in VIEW_RATIOS.values())
PER_VIEW_DIM = MAX_KEYS * 2 + MAX_RATIOS + 3  # + n_frac, mean_conf, plane_area
# relative centers (cx, cy) per key structure — important for 3VT vessel/trachea layout
PER_VIEW_POS_DIM = MAX_KEYS * 2
PER_VIEW_ANAT_DIM = PER_VIEW_DIM + PER_VIEW_POS_DIM
TOTAL_DIM = PER_VIEW_DIM * len(CARDIUM_STANDARD_VIEWS)
TOTAL_ANAT_DIM = PER_VIEW_ANAT_DIM  # active-view only (for frame concat)


def _area(box: dict) -> float:
    return float(box.get("area", float(box["w"]) * float(box["h"])))


def _safe_ratio(a: float, b: float) -> float:
    if b <= 1e-8 or a <= 0:
        return float("nan")
    return a / b


def view_stats_active(
    cardium_view: str,
    boxes: dict[int, dict],
    *,
    plane_area_norm: float = 0.0,
) -> np.ndarray:
    """Compact stats for one view only (length PER_VIEW_DIM)."""
    keys = VIEW_KEYS[cardium_view]
    plane_a = float(plane_area_norm) if plane_area_norm and plane_area_norm > 0 else float("nan")
    dens = plane_a if np.isfinite(plane_a) and plane_a > 1e-8 else 1.0

    area_conf: list[float] = []
    confs: list[float] = []
    for cid in keys:
        b = boxes.get(cid)
        if b:
            area_conf.extend([_area(b) / dens, float(b.get("conf", 0.0))])
            confs.append(float(b.get("conf", 0.0)))
        else:
            area_conf.extend([0.0, 0.0])
    # pad to MAX_KEYS
    while len(area_conf) < MAX_KEYS * 2:
        area_conf.extend([0.0, 0.0])

    ratios: list[float] = []
    for a, b in VIEW_RATIOS[cardium_view]:
        ba, bb = boxes.get(a), boxes.get(b)
        if ba and bb:
            ratios.append(_safe_ratio(_area(ba), _area(bb)))
        else:
            ratios.append(float("nan"))
    while len(ratios) < MAX_RATIOS:
        ratios.append(0.0)

    n_keys = len(keys)
    n_hit = sum(1 for cid in keys if cid in boxes)
    n_frac = n_hit / max(n_keys, 1)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    plane_feat = plane_a if np.isfinite(plane_a) else 0.0

    vec = np.array(area_conf + ratios + [n_frac, mean_conf, plane_feat], dtype=np.float64)
    assert vec.shape[0] == PER_VIEW_DIM
    return vec


def view_stats_active_with_pos(
    cardium_view: str,
    boxes: dict[int, dict],
    *,
    plane_area_norm: float = 0.0,
) -> np.ndarray:
    """view_stats_active + per-key (cx, cy) for spatial layout (3VT vessels/trachea)."""
    base = view_stats_active(cardium_view, boxes, plane_area_norm=plane_area_norm)
    keys = VIEW_KEYS[cardium_view]
    pos: list[float] = []
    for cid in keys:
        b = boxes.get(cid)
        if b and "cx" in b and "cy" in b:
            pos.extend([float(b["cx"]), float(b["cy"])])
        else:
            pos.extend([float("nan"), float("nan")])
    while len(pos) < PER_VIEW_POS_DIM:
        pos.extend([0.0, 0.0])
    pos = pos[:PER_VIEW_POS_DIM]
    out = np.concatenate([base, np.asarray(pos, dtype=np.float64)])
    assert out.shape[0] == PER_VIEW_ANAT_DIM
    return out


def view_stats_packed(
    cardium_view: str | None,
    boxes: dict[int, dict],
    *,
    plane_area_norm: float = 0.0,
) -> np.ndarray:
    """4-view packed: only active view block filled, others zero."""
    parts: list[np.ndarray] = []
    for v in CARDIUM_STANDARD_VIEWS:
        if cardium_view == v:
            parts.append(view_stats_active(v, boxes, plane_area_norm=plane_area_norm))
        else:
            parts.append(np.zeros(PER_VIEW_DIM, dtype=np.float64))
    return np.concatenate(parts)


def view_stats_from_tag(tag: dict, cardium_view: str | None) -> np.ndarray:
    boxes = boxes_from_tag(tag)
    plane_a = float(tag.get("plane_area_norm") or 0.0)
    return view_stats_packed(cardium_view, boxes, plane_area_norm=plane_a)


def view_anat_frame_from_tag(tag: dict, cardium_view: str | None) -> np.ndarray:
    """Active-view anatomy (+pos) for frame concat; zeros if view unknown/fallback."""
    if cardium_view not in VIEW_KEYS:
        return np.zeros(PER_VIEW_ANAT_DIM, dtype=np.float64)
    boxes = boxes_from_tag(tag)
    plane_a = float(tag.get("plane_area_norm") or 0.0)
    return view_stats_active_with_pos(cardium_view, boxes, plane_area_norm=plane_a)


def view_stats_dim() -> int:
    return TOTAL_DIM


def view_anat_frame_dim() -> int:
    """Dim of active-view anatomy stats (+pos) concatenated onto each frame."""
    return PER_VIEW_ANAT_DIM
