"""Plane-specific key anatomy ids (YOLO class ids). Vendored for the public drop."""

from __future__ import annotations

PLANE_KEY_ANATOMY: dict[str, tuple[int, ...]] = {
    "vv": (2, 6, 29),
    "vvt": (2, 6, 7, 29),
    "sivc": (6, 10, 11),
    "ao_arch": (8, 17, 3),
    "other": (),
    "dao_arch": (3, 4, 17),
    "rvot": (21, 29, 17),
    "four_chamber": (11, 21, 24, 25, 26),
    "lrpa": (1, 30, 29),
    "lvot": (24, 17, 32),
    "sa_base": (21, 24, 26, 11, 29),
}
