#!/usr/bin/env python3
"""Real patient-level multi-view bags (no Frankenstein stitching).

Unified slots: four_chamber, lvot, rvot, vvt (3VT).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

VIEW_SLOTS: tuple[str, ...] = ("four_chamber", "lvot", "rvot", "vvt")
DOMAIN_PRIVATE = "private"
DOMAIN_CARDIUM = "cardium"


@dataclass
class RealPatientBag:
    patient_id: str
    domain: str
    y: int
    present: tuple[bool, bool, bool, bool]
    n_views: int
    split: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _domain_from_row(row) -> str:
    meta = getattr(row, "meta", None) or {}
    dom = str(meta.get("domain") or DOMAIN_PRIVATE).lower()
    return DOMAIN_CARDIUM if dom == DOMAIN_CARDIUM else DOMAIN_PRIVATE


def rows_to_bags(
    rows: Sequence,
    views: Sequence[str] = VIEW_SLOTS,
) -> list[RealPatientBag]:
    by_pid: dict[str, list] = defaultdict(list)
    for r in rows:
        by_pid[str(r.patient_id)].append(r)

    bags: list[RealPatientBag] = []
    for pid, items in by_pid.items():
        label = int(items[0].label)
        split = str(getattr(items[0], "split", "") or "")
        domain = _domain_from_row(items[0])
        present = [False] * len(views)
        for vi, view in enumerate(views):
            present[vi] = any(str(it.cardium_view) == view for it in items)
        n_views = int(sum(present))
        if n_views == 0:
            continue
        bags.append(
            RealPatientBag(
                patient_id=pid,
                domain=domain,
                y=label,
                present=tuple(present),
                n_views=n_views,
                split=split,
            )
        )
    return bags


def coverage_report(
    rows: Sequence,
    views: Sequence[str] = VIEW_SLOTS,
) -> dict:
    bags = rows_to_bags(rows, views=views)
    hist = Counter(b.n_views for b in bags)
    per_view = Counter()
    per_domain: dict[str, Counter] = defaultdict(Counter)
    per_split: dict[str, Counter] = defaultdict(Counter)
    for b in bags:
        per_domain[b.domain][b.n_views] += 1
        per_split[b.split or "unknown"][b.n_views] += 1
        for vi, ok in enumerate(b.present):
            if ok:
                per_view[views[vi]] += 1
    return {
        "n_patients": len(bags),
        "n_frames": len(rows),
        "views": list(views),
        "n_views_hist": dict(sorted(hist.items())),
        "per_view_present": dict(per_view),
        "per_domain_n_views": {k: dict(sorted(v.items())) for k, v in per_domain.items()},
        "per_split_n_views": {k: dict(sorted(v.items())) for k, v in per_split.items()},
        "label_counts": dict(Counter(b.y for b in bags)),
    }


def print_coverage_report(report: dict, title: str = "Real multi-view coverage") -> None:
    print(f"=== {title} ===")
    print(f"  patients={report.get('n_patients')} frames={report.get('n_frames')}")
    print(f"  n_views_hist={report.get('n_views_hist')}")
    print(f"  per_view_present={report.get('per_view_present')}")
    print(f"  per_domain={report.get('per_domain_n_views')}")
    print(f"  per_split={report.get('per_split_n_views')}")
    print(f"  labels={report.get('label_counts')}")


def tag_row_domain(row, domain: str):
    meta = dict(getattr(row, "meta", None) or {})
    meta["domain"] = domain
    row.meta = meta
    return row


def prefix_patient_id(row, prefix: str):
    row.patient_id = f"{prefix}{row.patient_id}"
    return row


def write_coverage_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
