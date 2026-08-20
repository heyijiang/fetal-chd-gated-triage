"""Plane-conditioned M0 feature extraction for MASVF screening."""

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
from agcd.frame_quality import key_anatomy_completeness  # noqa: E402
from detect_rule_binary_screening import RULE_FEATURE_NAMES, compute_rule_signals  # noqa: E402

VIEW_TO_PLANE = {
    "four_chamber": "four_chamber",
    "lvot": "lvot",
    "rvot": "rvot",
    "vvt": "vvt",
}

# M0 extras beyond RULE_FEATURE_NAMES (16-d screening vector).
# Atrial / crux ratios already live in extract_chamber_features but were
# historically omitted from RULE_FEATURE_NAMES — wire them into M0 here.
M0_EXTRA_NAMES = [
    "vent_area_norm",
    "plane_area_norm",
    "plane_conf",
    "key_anatomy_completeness",
    "view_four_chamber",
    "view_lvot",
    "view_rvot",
    "view_vvt",
    # atrium / width / crux ratios (scale-ish; used by true_ratios subset)
    "la_ra_area_ratio",
    "la_ra_w_ratio",
    "lv_rv_w_ratio",
    "crux_vent_area_ratio",
    "atrium_vent_area_ratio",
]

M0_FEATURE_NAMES: list[str] = list(RULE_FEATURE_NAMES) + M0_EXTRA_NAMES

# Cross-domain anatomical ratios only (no FOV abs, no presence/protocol, no view).
# Mapped to private CHD spectrum (rhd/lhd/sv_as/avsd/ebstein/pa_ivs):
#   lv_rv_* / la_ra_*     — chamber imbalance (hypoplasia / Ebstein RA)
#   left_right_*          — whole-side asymmetry
#   ivs_vent / crux_vent  — septum & AV junction (AVSD; IAS has no YOLO bbox)
#   atrium_vent           — atrial vs ventricular bulk (Ebstein)
M0_TRUE_RATIOS: list[str] = [
    "lv_rv_area_ratio",
    "la_ra_area_ratio",
    "left_right_area_ratio",
    "lv_rv_w_ratio",
    "la_ra_w_ratio",
    "ivs_vent_area_ratio",
    "crux_vent_area_ratio",
    "atrium_vent_area_ratio",
]

# Cross-eval default after LOO (2026-07-16): core3 only.
# Optional: core3 + lv_rv_w_ratio (add-one ≈0.740 vs core3 0.738).
M0_TRUE_RATIOS_CORE3: list[str] = [
    "lv_rv_area_ratio",
    "left_right_area_ratio",
    "ivs_vent_area_ratio",
]
M0_TRUE_RATIOS_CORE3_W: list[str] = [
    *M0_TRUE_RATIOS_CORE3,
    "lv_rv_w_ratio",
]

M0_FEATURE_SUBSETS: dict[str, list[str]] = {
    "all": list(M0_FEATURE_NAMES),
    "true_ratios": list(M0_TRUE_RATIOS),  # exploratory 8-d
    "true_ratios_core3": list(M0_TRUE_RATIOS_CORE3),  # default cross
    "true_ratios_core3_w": list(M0_TRUE_RATIOS_CORE3_W),
    "none": [],  # clip-only when feat_model=m1 (no M0 block)
}


def resolve_m0_feature_names(subset: str | list[str] | None = "all") -> list[str]:
    """Return ordered M0 feature names for a named subset or explicit list."""
    if subset is None or subset == "all":
        return list(M0_FEATURE_NAMES)
    if isinstance(subset, (list, tuple)):
        names = list(subset)
    else:
        key = str(subset).strip().lower()
        if key not in M0_FEATURE_SUBSETS:
            raise ValueError(
                f"unknown m0 feature subset {subset!r}; "
                f"choose from {sorted(M0_FEATURE_SUBSETS)}"
            )
        names = list(M0_FEATURE_SUBSETS[key])
    missing = [n for n in names if n not in M0_FEATURE_NAMES]
    if missing:
        raise ValueError(f"unknown M0 features: {missing}")
    return names


def select_m0_columns(X_full: np.ndarray, names: list[str]) -> np.ndarray:
    idx = [M0_FEATURE_NAMES.index(n) for n in names]
    return X_full[:, idx]


def _view_one_hot(cardium_view: str | None) -> dict[str, float]:
    out = {f"view_{v}": 0.0 for v in CARDIUM_STANDARD_VIEWS}
    if cardium_view in CARDIUM_STANDARD_VIEWS:
        out[f"view_{cardium_view}"] = 1.0
    return out


def vent_area_normalized(feat: dict[str, float], plane_area_norm: float) -> float:
    vent = float(feat.get("vent_area_total", 0.0))
    denom = float(plane_area_norm) if plane_area_norm and plane_area_norm > 0 else np.nan
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return vent / denom


def key_anatomy_frac(cardium_view: str | None, detected_ids: set[int]) -> float:
    if not cardium_view:
        return 0.0
    plane = VIEW_TO_PLANE.get(cardium_view, cardium_view)
    hard, _ = key_anatomy_completeness(plane, detected_ids)
    return hard


def build_m0_feature_dict(
    feat: dict[str, float],
    *,
    cardium_view: str | None,
    plane_conf: float = 0.0,
    plane_area_norm: float = 0.0,
    detected_ids: set[int] | None = None,
) -> dict[str, float]:
    """Merge chamber/rule features with M0 plane context."""
    merged = dict(feat)
    merged.update(compute_rule_signals(feat))
    ids = detected_ids or set()
    merged["vent_area_norm"] = vent_area_normalized(feat, plane_area_norm)
    merged["plane_area_norm"] = float(plane_area_norm)
    merged["plane_conf"] = float(plane_conf)
    merged["key_anatomy_completeness"] = key_anatomy_frac(cardium_view, ids)
    # Derive atrium/vent if missing (legacy caches may lack atrium_vent_area_ratio)
    if not np.isfinite(float(merged.get("atrium_vent_area_ratio", np.nan) or np.nan)):
        atr = float(merged.get("atrium_area_total", 0.0) or 0.0)
        vent = float(merged.get("vent_area_total", 0.0) or 0.0)
        merged["atrium_vent_area_ratio"] = (
            atr / vent if atr > 0 and vent > 0 else float("nan")
        )
    merged.update(_view_one_hot(cardium_view))
    return merged


def m0_feat_vector(feat: dict[str, float], meta: dict | None = None) -> np.ndarray:
    """Fixed-length M0 vector."""
    meta = meta or {}
    merged = build_m0_feature_dict(
        feat,
        cardium_view=meta.get("cardium_view"),
        plane_conf=float(meta.get("plane_conf", 0.0)),
        plane_area_norm=float(meta.get("plane_area_norm", 0.0)),
        detected_ids=set(meta.get("detected_class_ids") or []),
    )
    return np.array([merged.get(n, np.nan) for n in M0_FEATURE_NAMES], dtype=np.float64)


def detected_ids_from_tag(tag: dict) -> set[int]:
    if tag.get("detected_class_ids"):
        return set(int(x) for x in tag["detected_class_ids"])
    present = tag.get("present") or {}
    if not present:
        return set()
    from chd_baseline.anatomy import ANATOMY_YOLO_IDS
    inv = {v: k for k, v in ANATOMY_YOLO_IDS.items()}
    return {inv[k] for k, v in present.items() if v and k in inv}
