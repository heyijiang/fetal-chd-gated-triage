#!/usr/bin/env python3
"""Build year-holdout manifests for R6-F4.

Default mode ``chd_year`` (recommended):
  Normals stay in their original train/val splits for all years (otherwise
  holding out 2024 removes nearly all negatives and LR sees one class).
  CHD with year == Y move to test; CHD with year != Y keep train/val.
  Test negatives = original test normals (stable negative pool).

Strict mode ``both`` (legacy): drop year==Y from train/val for BOTH classes;
  skip years where train lacks two classes (e.g. 2024).

  python -u experiments/build_tertiary_year_holdout_manifests.py \\
    --manifest data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl \\
    --hold-years 2023,2024,2025 \\
    --mode chd_year \\
    --out-dir outputs/tertiary_r6_year_holdout/manifests
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from audit_homologous_real_shortcuts import year_of_study  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument("--hold-years", default="2023,2024,2025")
    p.add_argument(
        "--mode",
        choices=["chd_year", "both"],
        default="chd_year",
        help="chd_year=hold out CHD year only (default); both=hold out both classes",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_r6_year_holdout/manifests",
    )
    p.add_argument(
        "--embed-cache",
        type=Path,
        default=None,
        help="If set, drop studies with zero FetalCLIP embeddings before counting years / writing holds.",
    )
    p.add_argument("--min-test-pos", type=int, default=5)
    p.add_argument("--min-train-pos", type=int, default=10)
    p.add_argument("--min-train-neg", type=int, default=10)
    return p.parse_args()


def _strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def _counts(rows: list[dict]) -> tuple[int, int]:
    pos = sum(1 for r in rows if int(r.get("label_binary", 0)) == 1)
    neg = sum(1 for r in rows if int(r.get("label_binary", 0)) == 0)
    return pos, neg


def _load_embed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "embedding" not in row:
                continue
            for k in ("sample_id", "image_path", "path"):
                if row.get(k):
                    keys.add(str(row[k]))
    return keys


def _study_has_embed(s: dict, keys: set[str]) -> bool:
    sid0 = str(s.get("study_id") or "")
    for rel in s.get("frame_paths") or []:
        rel = str(rel)
        if rel in keys or (sid0 and f"{sid0}|{rel}" in keys):
            return True
    return False


def build_chd_year(studies: list[dict], y: int) -> tuple[list, list, list]:
    """Hold out CHD from year Y; keep all normals in original train/val."""
    train, val, test = [], [], []
    for s in studies:
        lab = int(s.get("label_binary", 0))
        yy = s.get("_acq_year")
        sp = s.get("split")
        row = _strip(s)
        if yy is not None:
            row["acquisition_year"] = int(yy)

        if lab == 0:
            # normals: keep original train/val; original test normals → test pool
            if sp == "train":
                train.append(row)
            elif sp == "val":
                val.append(row)
            elif sp == "test":
                test.append(row)
            continue

        # CHD
        if yy == y:
            row["split"] = "test"
            test.append(row)
        elif yy is None:
            continue
        elif sp == "train":
            train.append(row)
        elif sp == "val":
            val.append(row)
        # original test CHD with year!=Y dropped from this holdout
    return train, val, test


def build_both(studies: list[dict], y: int) -> tuple[list, list, list]:
    """Hold out both classes with year==Y (may leave train with one class)."""
    train, val, test = [], [], []
    for s in studies:
        yy = s.get("_acq_year")
        row = _strip(s)
        if yy == y:
            row["split"] = "test"
            row["acquisition_year"] = y
            test.append(row)
        elif yy is None:
            continue
        else:
            row["acquisition_year"] = int(yy)
            sp = row.get("split")
            if sp == "train":
                train.append(row)
            elif sp == "val":
                val.append(row)
    return train, val, test


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        print(f"ERROR: missing {args.manifest}")
        return 1
    years = [int(x.strip()) for x in args.hold_years.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    studies: list[dict] = []
    for line in args.manifest.open(encoding="utf-8"):
        if not line.strip():
            continue
        s = json.loads(line)
        y, src = year_of_study(s)
        s = dict(s)
        s["_acq_year"] = y
        s["_acq_year_src"] = src
        studies.append(s)

    n_dropped = 0
    if args.embed_cache is not None:
        if not args.embed_cache.is_file():
            print(f"ERROR: missing embed cache {args.embed_cache}", file=sys.stderr)
            return 1
        keys = _load_embed_keys(args.embed_cache)
        kept: list[dict] = []
        for s in studies:
            if _study_has_embed(s, keys):
                kept.append(s)
            else:
                n_dropped += 1
        studies = kept
        print(f"embed-cache filter: dropped {n_dropped} studies with 0 embeddings; kept {len(studies)}")

    year_hist = Counter()
    year_by_lab = defaultdict(Counter)
    for s in studies:
        y = s.get("_acq_year")
        key = str(y) if y is not None else "unknown"
        year_hist[key] += 1
        year_by_lab[int(s.get("label_binary", 0))][key] += 1

    inventory = {
        "source_manifest": str(args.manifest),
        "mode": args.mode,
        "n_dropped_no_embed": n_dropped,
        "embed_cache": str(args.embed_cache) if args.embed_cache else None,
        "year_hist_all": dict(year_hist),
        "year_hist_normal": dict(year_by_lab[0]),
        "year_hist_chd": dict(year_by_lab[1]),
        "holds": [],
        "note": (
            "chd_year: normals always in train/val; CHD year==Y → test. "
            "Avoids one-class train when normals concentrate in one year (e.g. 2024)."
        ),
    }
    print(f"mode={args.mode}")
    print(f"normal years: {dict(year_by_lab[0])}")
    print(f"CHD years:    {dict(year_by_lab[1])}")

    for y in years:
        if args.mode == "chd_year":
            train, val, test = build_chd_year(studies, y)
        else:
            train, val, test = build_both(studies, y)

        tr_pos, tr_neg = _counts(train)
        va_pos, va_neg = _counts(val)
        te_pos, te_neg = _counts(test)
        entry = {
            "hold_year": y,
            "mode": args.mode,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_train_pos": tr_pos,
            "n_train_neg": tr_neg,
            "n_val_pos": va_pos,
            "n_val_neg": va_neg,
            "n_test_pos": te_pos,
            "n_test_neg": te_neg,
            "skipped": False,
            "reason": None,
            "manifest": None,
        }

        reasons = []
        if te_pos < args.min_test_pos:
            reasons.append(f"test_pos={te_pos}<{args.min_test_pos}")
        if te_neg < 1:
            reasons.append(f"test_neg={te_neg}")
        if tr_pos < args.min_train_pos:
            reasons.append(f"train_pos={tr_pos}<{args.min_train_pos}")
        if tr_neg < args.min_train_neg:
            reasons.append(f"train_neg={tr_neg}<{args.min_train_neg} (one-class risk)")
        if va_pos < 1 or va_neg < 1:
            reasons.append(f"val needs both classes (pos={va_pos}, neg={va_neg})")

        if reasons:
            entry["skipped"] = True
            entry["reason"] = "; ".join(reasons)
            inventory["holds"].append(entry)
            print(f"SKIP year={y}: {entry['reason']}")
            continue

        out_path = args.out_dir / f"manifest_hold_{y}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in train + val + test:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        entry["manifest"] = str(out_path)
        inventory["holds"].append(entry)
        print(
            f"OK year={y}: train={len(train)} (pos={tr_pos}/neg={tr_neg}) "
            f"val={len(val)} (pos={va_pos}/neg={va_neg}) "
            f"test={len(test)} (pos={te_pos}/neg={te_neg}) → {out_path}"
        )

    inv_path = args.out_dir / "inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Wrote {inv_path}")
    n_ok = sum(1 for h in inventory["holds"] if not h["skipped"])
    if n_ok == 0:
        print("ERROR: no usable holdout years")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
