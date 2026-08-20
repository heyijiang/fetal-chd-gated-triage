#!/usr/bin/env python3
"""CHD-only appearance probe: <2020 vs ≥2020 from frozen FetalCLIP patient means.

Not a screening experiment. Both classes are CHD. The question is whether
pre-2020 studies are linearly separable from ≥2020 studies in the same
embedding used by the paper. Unknown-year studies are dropped.

  python -u experiments/probe_chd_era_appearance.py --embed-load-only

Outputs under --out-dir:
  era_appearance.json
  LATEX_SNIPPET.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from agcd.fetalclip_embed import FETALCLIP_EMBED_DIM  # noqa: E402
from audit_homologous_real_shortcuts import year_of_study  # noqa: E402
from private_fetalclip_fullframe_linear import resolve_frame_path  # noqa: E402

CUT_YEAR = 2020


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHD-only <2020 vs ≥2020 linear probe")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_all_years.jsonl",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/study_screening/tertiary_20241125_vs_chd_raw_base_embeddings.jsonl",
    )
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_bspc_final_probes/chd_era_appearance",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--embed-load-only", action="store_true", default=True)
    return p.parse_args()


def load_chd_studies(manifest: Path, data_root: Path) -> list[dict]:
    studies = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if int(s.get("label_binary", 0)) != 1:
                continue
            year, src = year_of_study(s)
            frames = []
            for rel in s.get("frame_paths") or []:
                rel_s = str(rel)
                path = resolve_frame_path(data_root, rel_s)
                sid = f"{s['study_id']}|{rel_s}"
                frames.append((sid, str(path)))
            studies.append(
                {
                    "patient_id": str(s["patient_id"]),
                    "study_id": str(s["study_id"]),
                    "year": year,
                    "year_source": src,
                    "frames": frames,
                }
            )
    return studies


def stream_patient_means(
    cache: Path,
    studies: list[dict],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    sid_to_pid: dict[str, str] = {}
    path_to_pid: dict[str, str] = {}
    needed_sid: set[str] = set()
    needed_path: set[str] = set()
    for st in studies:
        pid = st["patient_id"]
        for sid, path in st["frames"]:
            sid_to_pid[sid] = pid
            path_to_pid[path] = pid
            needed_sid.add(sid)
            needed_path.add(path)

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    n_hit = 0
    n_lines = 0
    with cache.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            row = json.loads(line)
            if "embedding" not in row:
                continue
            sid = row.get("sample_id") or ""
            ip = row.get("image_path") or row.get("path") or ""
            pid = None
            if sid and sid in needed_sid:
                pid = sid_to_pid[sid]
            elif ip and ip in needed_path:
                pid = path_to_pid[ip]
            if pid is None:
                continue
            vec = np.asarray(row["embedding"], dtype=np.float64)
            if pid not in sums:
                sums[pid] = np.zeros(FETALCLIP_EMBED_DIM, dtype=np.float64)
            sums[pid] += vec
            counts[pid] += 1
            n_hit += 1
            if n_lines % 100000 == 0:
                print(
                    f"  cache lines={n_lines} chd_frame_hits={n_hit} patients={len(counts)}",
                    flush=True,
                )
    print(f"  cache done lines={n_lines} chd_frame_hits={n_hit} patients={len(counts)}", flush=True)
    means = {pid: sums[pid] / counts[pid] for pid in sums if counts[pid] > 0}
    return means, dict(counts)


def fit_lr(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> Pipeline:
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    clf.fit(X, y)
    return clf


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        print(f"ERROR: missing manifest {args.manifest}")
        return 1
    if not args.cache.is_file():
        print(f"ERROR: missing embed cache {args.cache}")
        print("  backfill: bash experiments/run_tertiary_full_era_embed_backfill.sh")
        return 1

    print(f"Loading CHD studies from {args.manifest}", flush=True)
    studies = load_chd_studies(args.manifest, args.data_root)
    n_unknown = sum(1 for s in studies if s["year"] is None)
    usable = [s for s in studies if s["year"] is not None]
    print(f"  CHD studies={len(studies)} unknown_year={n_unknown} dated={len(usable)}")

    # One row per patient. If several studies exist, keep the earliest year only
    # (do not mix frames across years).
    by_pid: dict[str, dict] = {}
    for s in usable:
        pid = s["patient_id"]
        prev = by_pid.get(pid)
        if prev is None or int(s["year"]) < int(prev["year"]):
            by_pid[pid] = s

    dated = list(by_pid.values())
    print(f"  unique dated CHD patients={len(dated)}", flush=True)
    print(f"Streaming embeddings from {args.cache}", flush=True)
    means, counts = stream_patient_means(args.cache, dated)

    X, y, pids, years = [], [], [], []
    dropped_no_embed = 0
    for s in dated:
        pid = s["patient_id"]
        if pid not in means:
            dropped_no_embed += 1
            continue
        year = int(s["year"])
        X.append(means[pid])
        y.append(1 if year < CUT_YEAR else 0)  # 1 = pre-2020
        pids.append(pid)
        years.append(year)
    X = np.stack(X) if X else np.empty((0, FETALCLIP_EMBED_DIM))
    y = np.asarray(y, dtype=np.int64)
    years = np.asarray(years, dtype=np.int64)
    print(
        f"  embed-present CHD={len(y)} pre2020={int((y == 1).sum())} "
        f"ge2020={int((y == 0).sum())} dropped_no_embed={dropped_no_embed}"
    )
    if len(set(y.tolist())) < 2 or (y == 1).sum() < 20 or (y == 0).sum() < 20:
        print("ERROR: not enough CHD in both era bins")
        return 1

    pids_arr = np.asarray(pids)
    idx = np.arange(len(y))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )
    clf = fit_lr(X[idx_tr], y[idx_tr], args.C, args.seed)
    p_va = clf.predict_proba(X[idx_va])[:, 1]
    p_te = clf.predict_proba(X[idx_te])[:, 1]
    auc_va = float(roc_auc_score(y[idx_va], p_va))
    auc_te = float(roc_auc_score(y[idx_te], p_te))
    pred = (p_te >= 0.5).astype(np.int64)
    acc = float((pred == y[idx_te]).mean())

    year_hist = Counter(int(v) for v in years.tolist())
    out = {
        "task": "CHD-only linear probe: y=1 if acquisition_year<2020 else 0",
        "cut_year": CUT_YEAR,
        "n_chd_manifest": len(studies),
        "n_unknown_year": n_unknown,
        "n_embed_present": int(len(y)),
        "n_pre2020": int((y == 1).sum()),
        "n_ge2020": int((y == 0).sum()),
        "n_dropped_no_embed": dropped_no_embed,
        "split": {
            "n_train": int(len(idx_tr)),
            "n_val": int(len(idx_va)),
            "n_test": int(len(idx_te)),
            "n_pre2020_test": int((y[idx_te] == 1).sum()),
            "n_ge2020_test": int((y[idx_te] == 0).sum()),
        },
        "val_auc": auc_va,
        "test_auc": auc_te,
        "test_acc_thr0.5": acc,
        "C": args.C,
        "seed": args.seed,
        "year_hist_embed_present": dict(sorted(year_hist.items())),
        "test_patient_scores": [
            {
                "patient_id": str(pids_arr[i]),
                "year": int(years[i]),
                "y_pre2020": int(y[i]),
                "p_pre2020": float(p_te[k]),
            }
            for k, i in enumerate(idx_te)
        ],
        "n_frames_by_patient_mean": int(np.mean(list(counts.values()))) if counts else 0,
    }
    dest = args.out_dir / "era_appearance.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    snippet = (
        f"% CHD-only Frozen FetalCLIP patient-mean: <2020 vs $\\geq$2020\n"
        f"% n_pre={out['n_pre2020']} n_ge2020={out['n_ge2020']} "
        f"test n={out['split']['n_test']}\n"
        f"test AUC = {auc_te:.3f} (val {auc_va:.3f})\n"
    )
    (args.out_dir / "LATEX_SNIPPET.txt").write_text(snippet, encoding="utf-8")
    print(snippet)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
