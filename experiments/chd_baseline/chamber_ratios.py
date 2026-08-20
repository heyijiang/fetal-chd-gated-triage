"""Chamber bbox ratio features for quick geometry-based CHD classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .anatomy import ANATOMY_YOLO_IDS, parse_anatomy_boxes

BBOX_KEYS: tuple[str, ...] = ("lv", "rv", "la", "ra", "ivs", "crux")
CHAMBER_KEYS: tuple[str, ...] = ("lv", "rv", "la", "ra")


def _area(w: float, h: float) -> float:
    return w * h


def _safe_ratio(num: float, den: float, default: float = np.nan) -> float:
    if den <= 0 or num <= 0:
        return default
    return num / den


@dataclass(frozen=True)
class ChamberFeatures:
    """Named ratio features derived from YOLO cxcywh boxes."""

    values: dict[str, float]

    def to_vector(self, names: list[str]) -> np.ndarray:
        return np.array([self.values.get(n, np.nan) for n in names], dtype=np.float64)

    @property
    def chamber_count(self) -> int:
        return int(sum(self.values.get(f"{k}_present", 0.0) for k in CHAMBER_KEYS))


FEATURE_NAMES: list[str] = [
    *[f"{k}_area" for k in BBOX_KEYS],
    *[f"{k}_w" for k in BBOX_KEYS],
    *[f"{k}_h" for k in BBOX_KEYS],
    *[f"{k}_present" for k in BBOX_KEYS],
    "chamber_count",
    "lv_rv_area_ratio",
    "la_ra_area_ratio",
    "lv_rv_w_ratio",
    "la_ra_w_ratio",
    "left_right_area_ratio",
    "vent_area_total",
    "atrium_area_total",
    "ivs_vent_area_ratio",
    "crux_vent_area_ratio",
    "atrium_vent_area_ratio",
    "ivs_present",
    "crux_present",
]


def boxes_from_yolo_dets(
    dets: list[tuple[int, float, float, float, float]],
) -> dict[str, tuple[float, float, float, float]]:
    """Best-confidence det per anatomy key from raw YOLO outputs."""
    best: dict[str, tuple[float, float, float, float, float]] = {}
    for cid, conf, cx, cy, w, h in dets:
        key = ANATOMY_YOLO_IDS.get(int(cid))
        if key is None:
            continue
        if key not in best or conf > best[key][0]:
            best[key] = (conf, cx, cy, w, h)
    return {k: v[1:] for k, v in best.items()}


def extract_chamber_features(boxes: Mapping[str, tuple[float, float, float, float]]) -> ChamberFeatures:
    """Build ratio features from anatomy_key -> (cx, cy, w, h)."""
    areas = {k: _area(*boxes[k][2:4]) if k in boxes else 0.0 for k in BBOX_KEYS}
    widths = {k: boxes[k][2] if k in boxes else 0.0 for k in BBOX_KEYS}
    heights = {k: boxes[k][3] if k in boxes else 0.0 for k in BBOX_KEYS}
    present = {k: float(k in boxes) for k in BBOX_KEYS}

    vent_total = areas["lv"] + areas["rv"]
    atrium_total = areas["la"] + areas["ra"]
    left_total = areas["lv"] + areas["la"]
    right_total = areas["rv"] + areas["ra"]

    values: dict[str, float] = {}
    for k in BBOX_KEYS:
        values[f"{k}_area"] = areas[k]
        values[f"{k}_w"] = widths[k]
        values[f"{k}_h"] = heights[k]
        values[f"{k}_present"] = present[k]

    values["chamber_count"] = float(sum(present[k] for k in CHAMBER_KEYS))
    values["lv_rv_area_ratio"] = _safe_ratio(areas["lv"], areas["rv"])
    values["la_ra_area_ratio"] = _safe_ratio(areas["la"], areas["ra"])
    values["lv_rv_w_ratio"] = _safe_ratio(widths["lv"], widths["rv"])
    values["la_ra_w_ratio"] = _safe_ratio(widths["la"], widths["ra"])
    values["left_right_area_ratio"] = _safe_ratio(left_total, right_total)
    values["vent_area_total"] = vent_total
    values["atrium_area_total"] = atrium_total
    values["ivs_vent_area_ratio"] = _safe_ratio(areas["ivs"], vent_total)
    values["crux_vent_area_ratio"] = _safe_ratio(areas["crux"], vent_total)
    values["atrium_vent_area_ratio"] = _safe_ratio(atrium_total, vent_total)
    values["ivs_present"] = present["ivs"]
    values["crux_present"] = present["crux"]
    return ChamberFeatures(values=values)


def parse_boxes_from_label_file(label_path) -> dict[str, tuple[float, float, float, float]]:
    from pathlib import Path

    return parse_anatomy_boxes(Path(label_path))


def impute_features(X: np.ndarray, fill: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN/inf with column medians (or provided fill vector)."""
    X = np.asarray(X, dtype=np.float64)
    bad = ~np.isfinite(X)
    if fill is None:
        fill = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
    X_out = X.copy()
    X_out[bad] = np.take(fill, np.where(bad)[1])
    return X_out, fill
