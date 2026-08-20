"""CARDIUM-aligned cardiac view taxonomy via pp3 YOLO plane tags."""

from __future__ import annotations

from dataclasses import dataclass

# CARDIUM paper standard fetal cardiac views (exam protocol)
CARDIUM_STANDARD_VIEWS: tuple[str, ...] = (
    "four_chamber",
    "lvot",
    "rvot",
    "vvt",
)

VIEW_LABEL_EN: dict[str, str] = {
    "four_chamber": "four-chamber (4C)",
    "lvot": "left ventricular outflow tract (LVOT)",
    "rvot": "right ventricular outflow tract (RVOT)",
    "vvt": "three-vessel trachea (3VT)",
}

# pp3 YOLO plane-tag id → plane code (experiments/agcd uses yolo_full_with_tags)
PLANE_TAG_TO_CODE: dict[int, str] = {
    5: "vv",
    9: "vvt",
    12: "sivc",
    18: "other",
    19: "ao_arch",
    22: "dao_arch",
    23: "rvot",
    28: "four_chamber",
    31: "lrpa",
    33: "lvot",
    34: "sa_base",
    35: "other",
    48: "four_chamber",
}

CARDIAC_PLANE_CODES: frozenset[str] = frozenset(
    {
        "vv",
        "vvt",
        "sivc",
        "ao_arch",
        "dao_arch",
        "rvot",
        "four_chamber",
        "lrpa",
        "lvot",
        "sa_base",
    }
)


@dataclass(frozen=True)
class PlaneAssignment:
    plane_code: str
    conf: float
    yolo_class_id: int


def dominant_plane_from_dets(
    dets: list[tuple[int, float]],
    *,
    min_conf: float = 0.15,
) -> PlaneAssignment | None:
    """Pick highest-conf plane tag among YOLO detections."""
    best: PlaneAssignment | None = None
    for cid, cf in dets:
        if cf < min_conf:
            continue
        code = PLANE_TAG_TO_CODE.get(int(cid))
        if code is None:
            continue
        pa = PlaneAssignment(code, float(cf), int(cid))
        if best is None or pa.conf > best.conf:
            best = pa
    return best


def map_to_cardium_view(plane_code: str) -> str | None:
    """Map pp3 plane code to CARDIUM standard view bucket (or None if non-standard)."""
    if plane_code in CARDIUM_STANDARD_VIEWS:
        return plane_code
    # 三血管切面 sometimes used interchangeably with 3VT in practice — optional merge
    if plane_code == "vv":
        return "vvt"
    return None


def is_cardiac_plane(plane_code: str) -> bool:
    return plane_code in CARDIAC_PLANE_CODES
