"""Paradigm-A frame quality scoring for fetal cardiac view selection.

Ranking (lexicographic, descending):
  1. key-anatomy completeness for the assigned CARDIUM view
  2. normalized plane-tag bounding-box area
  3. plane-tag confidence

See docs/MASVF_MAIN_METHOD_SPEC.md §4.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from atvs.plane_anatomy import PLANE_KEY_ANATOMY  # noqa: E402

from agcd.cardium_views import CARDIUM_STANDARD_VIEWS, PLANE_TAG_TO_CODE  # noqa: E402

# CARDIUM view bucket → pp3 plane_anatomy keys
CARDIUM_VIEW_TO_PLANE: dict[str, str] = {
    "four_chamber": "four_chamber",
    "lvot": "lvot",
    "rvot": "rvot",
    "vvt": "vvt",
}

# YOLO plane class ids per CARDIUM view (for plane box area)
CARDIUM_VIEW_PLANE_IDS: dict[str, tuple[int, ...]] = {
    "four_chamber": (28, 48),
    "lvot": (33,),
    "rvot": (23,),
    "vvt": (9, 5),  # 3VT + 3VV (vv maps to vvt bucket)
}

DEFAULT_MIN_PLANE_CONF = 0.15
DEFAULT_MIN_COMPLETENESS = 0.34  # e.g. 2/6 key structures on 4C


@dataclass(frozen=True)
class ParadigmAScores:
    """Quality scores for one frame (not anomaly)."""

    cardium_view: str
    completeness: float
    completeness_soft: float
    plane_area_norm: float
    plane_conf: float
    rank_key: tuple[float, float, float]
    usable: bool


def _plane_code_for_view(cardium_view: str) -> str:
    return CARDIUM_VIEW_TO_PLANE.get(cardium_view, cardium_view)


def detected_class_ids_from_dets(
    dets: list[tuple[int, float]] | list[tuple[int, float, float, float, float, float]],
) -> set[int]:
    out: set[int] = set()
    for d in dets:
        out.add(int(d[0]))
    return out


def key_anatomy_completeness(
    cardium_view: str,
    detected_ids: set[int],
    *,
    soft_confs: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Return (hard completeness, soft mean conf of hit key structures)."""
    plane = _plane_code_for_view(cardium_view)
    keys = PLANE_KEY_ANATOMY.get(plane, ())
    if not keys:
        return 0.0, 0.0
    hits = [k for k in keys if k in detected_ids]
    hard = len(hits) / len(keys)
    if soft_confs and hits:
        soft = sum(soft_confs.get(k, 0.0) for k in hits) / len(keys)
    else:
        soft = hard
    return float(hard), float(soft)


def plane_box_area_norm(
    dets: list[tuple[int, float, float, float, float, float]],
    cardium_view: str,
) -> tuple[float, float]:
    """Max conf plane-tag box area (w*h in norm coords) for this view."""
    plane_ids = set(CARDIUM_VIEW_PLANE_IDS.get(cardium_view, ()))
    if not plane_ids:
        return 0.0, 0.0

    best_area = 0.0
    best_conf = 0.0
    for cid, conf, _cx, _cy, w, h in dets:
        if int(cid) not in plane_ids:
            continue
        area = float(w) * float(h)
        if conf > best_conf or (conf == best_conf and area > best_area):
            best_conf = float(conf)
            best_area = area
    return best_area, best_conf


def plane_box_xyxy_norm(
    dets: list[tuple[int, float, float, float, float, float]],
    cardium_view: str,
) -> tuple[float, float, float, float] | None:
    """Best-conf plane-tag box as normalized xyxy in [0,1], or None if missing."""
    plane_ids = set(CARDIUM_VIEW_PLANE_IDS.get(cardium_view, ()))
    if not plane_ids:
        return None

    best: tuple[float, float, tuple[float, float, float, float]] | None = None
    for cid, conf, cx, cy, w, h in dets:
        if int(cid) not in plane_ids:
            continue
        x0 = float(cx) - float(w) / 2.0
        y0 = float(cy) - float(h) / 2.0
        x1 = float(cx) + float(w) / 2.0
        y1 = float(cy) + float(h) / 2.0
        area = float(w) * float(h)
        cand = (float(conf), area, (x0, y0, x1, y1))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    if best is None:
        return None
    x0, y0, x1, y1 = best[2]
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def crop_pil_by_xyxy_norm(
    img,
    xyxy_norm: tuple[float, float, float, float],
    *,
    pad: float = 0.12,
):
    """Crop PIL image by normalized xyxy with relative padding; clamp to image."""
    w_img, h_img = img.size
    x0, y0, x1, y1 = xyxy_norm
    bw = max(x1 - x0, 1e-6)
    bh = max(y1 - y0, 1e-6)
    x0 = max(0.0, x0 - pad * bw)
    y0 = max(0.0, y0 - pad * bh)
    x1 = min(1.0, x1 + pad * bw)
    y1 = min(1.0, y1 + pad * bh)
    box = (
        int(round(x0 * w_img)),
        int(round(y0 * h_img)),
        int(round(x1 * w_img)),
        int(round(y1 * h_img)),
    )
    if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        return img
    return img.crop(box)


def paradigm_a_scores(
    *,
    cardium_view: str | None,
    dets: list[tuple[int, float, float, float, float, float]] | None = None,
    plane_conf: float = 0.0,
    detected_ids: set[int] | None = None,
    plane_area_norm: float | None = None,
    min_plane_conf: float = DEFAULT_MIN_PLANE_CONF,
    min_completeness: float = DEFAULT_MIN_COMPLETENESS,
) -> ParadigmAScores | None:
    """Score one frame for Paradigm-A selection.

    Can pass full ``dets`` (cid, conf, cx, cy, w, h) or precomputed ids/area.
    """
    if not cardium_view or cardium_view not in CARDIUM_STANDARD_VIEWS:
        return None

    if dets is not None:
        ids = detected_class_ids_from_dets(dets)
        conf_map = {int(d[0]): float(d[1]) for d in dets}
        area, pconf_det = plane_box_area_norm(dets, cardium_view)
        plane_conf = max(plane_conf, pconf_det)
        if plane_area_norm is None:
            plane_area_norm = area
    else:
        ids = detected_ids or set()
        conf_map = None
        if plane_area_norm is None:
            plane_area_norm = 0.0

    hard, soft = key_anatomy_completeness(cardium_view, ids, soft_confs=conf_map)
    rank = (hard, float(plane_area_norm), float(plane_conf))
    usable = plane_conf >= min_plane_conf and hard >= min_completeness

    return ParadigmAScores(
        cardium_view=cardium_view,
        completeness=hard,
        completeness_soft=soft,
        plane_area_norm=float(plane_area_norm),
        plane_conf=float(plane_conf),
        rank_key=rank,
        usable=usable,
    )


def paradigm_a_rank_key(frame: dict) -> tuple[float, float, float]:
    """Rank key from a cached tag row or analyze_frame dict.

    Expected keys: cardium_view, plane_conf, present (optional),
    detected_class_ids (optional), plane_area_norm (optional),
    or raw dets list.
    """
    view = frame.get("cardium_view")
    if not view:
        return (0.0, 0.0, 0.0)

    dets = frame.get("dets")
    detected_ids = frame.get("detected_class_ids")
    if detected_ids is None and frame.get("present"):
        # Legacy chamber-only cache: map present dict → not enough for vascular views
        from chd_baseline.anatomy import ANATOMY_YOLO_IDS
        inv = {v: k for k, v in ANATOMY_YOLO_IDS.items()}
        detected_ids = {inv[k] for k, v in frame["present"].items() if v and k in inv}

    if detected_ids is not None and not isinstance(detected_ids, set):
        detected_ids = set(detected_ids)

    scores = paradigm_a_scores(
        cardium_view=view,
        dets=dets,
        plane_conf=float(frame.get("plane_conf", 0.0)),
        detected_ids=detected_ids,
        plane_area_norm=frame.get("plane_area_norm"),
    )
    if scores is None:
        return (0.0, 0.0, 0.0)
    frame["_paradigm_a"] = {
        "completeness": scores.completeness,
        "plane_area_norm": scores.plane_area_norm,
        "usable": scores.usable,
    }
    return scores.rank_key
