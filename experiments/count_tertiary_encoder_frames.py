#!/usr/bin/env python3
"""B4: count acquired frames vs YOLO-standard-view frames (encoder budget).

  python -u experiments/count_tertiary_encoder_frames.py --split test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ("four_chamber", "lvot", "rvot", "vvt")
ALIASES = {
    "four_chamber": "four_chamber",
    "4ch": "four_chamber",
    "lvot": "lvot",
    "rvot": "rvot",
    "vvt": "vvt",
    "3vt": "vvt",
    "3VT": "vvt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument(
        "--tags",
        type=Path,
        default=ROOT / "data/study_screening/yolo_image_tags_tertiary_20241125.jsonl",
    )
    p.add_argument("--split", default="test")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/tertiary_encoder_frame_counts.json",
    )
    return p.parse_args()


def norm_view(v) -> str | None:
    if not v:
        return None
    s = str(v)
    return ALIASES.get(s) or ALIASES.get(s.lower())


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"ERROR missing {args.manifest}")
        return 1
    tags: dict[str, str | None] = {}
    if args.tags.is_file():
        with args.tags.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row.get("sample_id")
                if sid:
                    tags[str(sid)] = row.get("cardium_view")
        print(f"loaded {len(tags)} tags")
    else:
        print(f"WARN no tags {args.tags}; will count all frames only")

    by = defaultdict(lambda: {"n_exams": 0, "n_frames": 0, "n_std": 0, "n_gated_exams": 0})
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if args.split != "all" and s.get("split") != args.split:
                continue
            lab = "chd" if int(s.get("label_binary", 0)) == 1 else "norm"
            frames = s.get("frame_paths") or []
            n_std = 0
            views = set()
            for rel in frames:
                sid = f"{s['study_id']}|{rel}"
                v = norm_view(tags.get(sid))
                if v in STANDARD:
                    n_std += 1
                    views.add(v)
            by[lab]["n_exams"] += 1
            by[lab]["n_frames"] += len(frames)
            by[lab]["n_std"] += n_std
            if views:
                by[lab]["n_gated_exams"] += 1

    out = {
        "split": args.split,
        "by_label": dict(by),
        "note": "gated FetalCLIP encodes n_std frames, then pools to ≤4 tokens; full-frame encodes n_frames",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
