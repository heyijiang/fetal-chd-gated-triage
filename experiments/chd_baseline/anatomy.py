"""YOLO class ids for four-chamber anatomy (no detector weights)."""

from __future__ import annotations

from pathlib import Path

ANATOMY_YOLO_IDS: dict[int, str] = {
    11: "ra",
    21: "rv",
    24: "lv",
    25: "ivs",
    26: "la",
    27: "crux",
}


def parse_anatomy_boxes(label_path: Path) -> dict[str, tuple[float, float, float, float]]:
    if not Path(label_path).is_file():
        return {}
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(parts[0])
        key = ANATOMY_YOLO_IDS.get(cid)
        if key is None:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes[key] = (cx, cy, w, h)
    return boxes
