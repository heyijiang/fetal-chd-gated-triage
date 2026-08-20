"""Load organized CARDIUM dataset for external validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FoldName = Literal["1", "2", "3"]
SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class CardiumRecord:
    sample_id: str
    image_path: Path
    patient_id: str
    label_binary: int
    label_name: str
    fold: int
    split: str


def default_cardium_processed_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "CARDIUM dataset" / "processed"


def default_cardium_root() -> Path:
    return Path(__file__).resolve().parents[1] / "CARDIUM dataset"


def load_manifest(processed_dir: Path | None = None) -> list[CardiumRecord]:
    processed_dir = processed_dir or default_cardium_processed_dir()
    cardium_root = processed_dir.parent
    records: list[CardiumRecord] = []

    with (processed_dir / "manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                CardiumRecord(
                    sample_id=row["sample_id"],
                    image_path=cardium_root / row["image_path"],
                    patient_id=row["patient_id"],
                    label_binary=int(row["label_binary"]),
                    label_name=row["label_name"],
                    fold=int(row["fold"]),
                    split=row["split"],
                )
            )
    return records


def load_fold_split(
    fold: FoldName,
    split: SplitName,
    processed_dir: Path | None = None,
    *,
    filter_manifest: Path | None = None,
) -> list[CardiumRecord]:
    processed_dir = processed_dir or default_cardium_processed_dir()
    folds = json.loads((processed_dir / "folds.json").read_text(encoding="utf-8"))
    allowed = set(folds[fold]["sample_ids"][split])
    records = [r for r in load_manifest(processed_dir) if r.sample_id in allowed]
    if filter_manifest is not None and filter_manifest.is_file():
        keep_ids = set()
        with filter_manifest.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("is_4c", "True") in ("True", "true", "1"):
                    keep_ids.add(row["sample_id"])
        records = [r for r in records if r.sample_id in keep_ids]
    return records
