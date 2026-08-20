"""Anatomy box attributes + view-conditioned geometry (no clinical tabular).

Uses YOLO dets for CARDIUM key structures (chambers + outflow vessels).
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

from agcd.cardium_views import CARDIUM_STANDARD_VIEWS  # noqa: E402

# id → short name (label.txt)
ANATOMY_ID_NAME: dict[int, str] = {
    1: "RPA",
    2: "AAO",
    3: "DAO",
    6: "SVC",
    7: "T",
    11: "RA",
    17: "AO",
    21: "RV",
    24: "LV",
    25: "IVS",
    26: "LA",
    27: "IAVS",
    29: "MPA",
    30: "LPA",
    32: "LAAO",
}

VIEW_KEY_IDS: dict[str, tuple[int, ...]] = {
    "four_chamber": tuple(dict.fromkeys((*PLANE_KEY_ANATOMY["four_chamber"], 27))),
    "lvot": PLANE_KEY_ANATOMY["lvot"],
    "rvot": PLANE_KEY_ANATOMY["rvot"],
    "vvt": PLANE_KEY_ANATOMY["vvt"],
}

TRACK_IDS: tuple[int, ...] = tuple(sorted({i for ids in VIEW_KEY_IDS.values() for i in ids}))

# Per structure attrs in geo vector (view slots): present, conf, area, aspect
ATTRS_PER_STRUCT = 4


def view_slot_dim() -> int:
    return sum(len(VIEW_KEY_IDS[v]) * ATTRS_PER_STRUCT for v in CARDIUM_STANDARD_VIEWS)


def dets_to_anatomy_boxes(
    dets: list[tuple[int, float, float, float, float, float]],
    *,
    keep_ids: set[int] | None = None,
) -> dict[str, dict]:
    """Best-conf box per class id → JSON-serializable dict keyed by str(cid).

    dets: (cid, conf, cx, cy, w, h) normalized.
    """
    keep = keep_ids or set(TRACK_IDS)
    best: dict[int, tuple[float, float, float, float, float]] = {}
    for cid, conf, cx, cy, w, h in dets:
        cid = int(cid)
        if cid not in keep:
            continue
        prev = best.get(cid)
        if prev is None or conf > prev[0]:
            best[cid] = (float(conf), float(cx), float(cy), float(w), float(h))
    out: dict[str, dict] = {}
    for cid, (cf, cx, cy, w, h) in best.items():
        out[str(cid)] = {
            "conf": round(cf, 4),
            "cx": round(cx, 6),
            "cy": round(cy, 6),
            "w": round(w, 6),
            "h": round(h, 6),
            "area": round(w * h, 6),
            "name": ANATOMY_ID_NAME.get(cid, str(cid)),
        }
    return out


def boxes_from_tag(tag: dict) -> dict[int, dict]:
    raw = tag.get("anatomy_boxes") or {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def cxcywh_to_xyxy(box: dict) -> tuple[float, float, float, float]:
    cx, cy, w, h = float(box["cx"]), float(box["cy"]), float(box["w"]), float(box["h"])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _struct_attrs(box: dict | None) -> list[float]:
    if not box:
        return [0.0, 0.0, 0.0, 0.0]
    w, h = float(box["w"]), float(box["h"])
    area = float(box.get("area", w * h))
    aspect = w / h if h > 1e-8 else 0.0
    return [1.0, float(box.get("conf", 0.0)), area, aspect]


def view_geometry_vector(cardium_view: str | None, boxes: dict[int, dict]) -> np.ndarray:
    """View-slot packing: only active view's key structs filled."""
    parts: list[float] = []
    for v in CARDIUM_STANDARD_VIEWS:
        keys = VIEW_KEY_IDS[v]
        if cardium_view == v:
            for cid in keys:
                parts.extend(_struct_attrs(boxes.get(cid)))
        else:
            parts.extend([0.0] * (len(keys) * ATTRS_PER_STRUCT))
    return np.array(parts, dtype=np.float64)


def pairwise_ratio_vector(cardium_view: str | None, boxes: dict[int, dict]) -> np.ndarray:
    """Compact within-view area ratios for clinically related pairs."""
    # fixed order; missing → nan
    pairs_by_view: dict[str, tuple[tuple[int, int], ...]] = {
        "four_chamber": ((24, 21), (26, 11), (24, 26)),  # LV/RV, LA/RA, LV/LA
        "lvot": ((24, 17), (17, 32)),  # LV/AO, AO/LAAO
        "rvot": ((21, 29), (29, 17)),  # RV/MPA, MPA/AO
        "vvt": ((2, 29), (2, 6), (29, 6)),  # AAO/MPA, AAO/SVC, MPA/SVC
    }
    # Flatten all pairs across views (zeros when view inactive)
    vals: list[float] = []
    for v in CARDIUM_STANDARD_VIEWS:
        for a, b in pairs_by_view[v]:
            if cardium_view != v:
                vals.append(0.0)
                continue
            ba, bb = boxes.get(a), boxes.get(b)
            if not ba or not bb:
                vals.append(float("nan"))
                continue
            aa = float(ba.get("area", ba["w"] * ba["h"]))
            ab = float(bb.get("area", bb["w"] * bb["h"]))
            vals.append(aa / ab if ab > 1e-8 else float("nan"))
    return np.array(vals, dtype=np.float64)


def geometry_feature_vector(cardium_view: str | None, boxes: dict[int, dict]) -> np.ndarray:
    return np.concatenate([
        view_geometry_vector(cardium_view, boxes),
        pairwise_ratio_vector(cardium_view, boxes),
    ])


def geometry_dim() -> int:
    # pairs: 4C3 + LVOT2 + RVOT2 + VVT3 = 10
    return view_slot_dim() + 10
