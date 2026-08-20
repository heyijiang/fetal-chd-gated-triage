#!/usr/bin/env python3
"""Binary screening: normal vs abnormal via YOLO detection + rules.

All images use YOLO inference (online-consistent). Features from chamber
geometry; rule score flags incomplete four-chamber pattern (typical in
单心室单心房 and other CHD). Optional LR on same features.

Rules (interpretable):
  - 四腔齐全：左/右心室、左/右心房 均检出
  - 间隔齐全：室间隔、十字交叉均检出
  → 不满足则计「异常票」；验证集上选阈值

Usage:
  cd experiments
  CUDA_VISIBLE_DEVICES=0 python detect_rule_binary_screening.py --device 0
  bash run_detect_rule_binary.sh

  # 仅：正常 + 单心室单心房/肺动脉闭锁/右心发育不良/左心发育不良
  bash run_detect_rule_binary_4struct.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from chd_baseline.chamber_ratios import extract_chamber_features, impute_features  # noqa: E402
from chd_baseline.metrics import (  # noqa: E402
    binary_metrics_at_threshold,
    find_best_binary_threshold,
)
from yolo_io import _make_logistic_regression, load_yolo_model  # noqa: E402

# 四腔结构异常子集：正常 vs 这四类
STRUCTURE_SUBSET: list[str] = ["normal", "sv_as", "pa_ivs", "rhd", "lhd"]

DISEASE_CN: dict[str, str] = {
    "normal": "正常",
    "rhd": "右心发育不良",
    "lhd": "左心发育不良",
    "sv_as": "单心室单心房",
    "avsd": "房室间隔缺损",
    "ebstein": "三尖瓣下移",
    "pa_ivs": "肺动脉闭锁室间隔完整",
    "tof": "法洛四联症",
    "tga": "大动脉转位",
    "dorv": "右室双出口",
    "papvr": "肺静脉异位引流",
}


@dataclass
class SampleRow:
    sample_id: str
    disease: str
    label: int  # 0 normal, 1 abnormal
    split: str
    boxes: dict[str, tuple[float, float, float, float]]
    feat: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + rules binary screening")
    p.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    p.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="Detector checkpoint (not shipped). Required only for this CLI, not for cache fusion.",
    )
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "detect_rule_binary")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--rules-only", action="store_true")
    p.add_argument(
        "--include-diseases",
        default="",
        help="Comma-separated disease codes to keep (e.g. normal,sv_as,pa_ivs,rhd,lhd). "
             "Empty = all diseases in manifest.",
    )
    return p.parse_args()


def parse_disease_filter(text: str) -> set[str] | None:
    if not text.strip():
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def filter_records_by_disease(
    records: list[Record],
    allowed: set[str],
    *,
    split: str | None = None,
) -> list[Record]:
    kept = [r for r in records if r.label_disease in allowed]
    dropped = len(records) - len(kept)
    if dropped:
        bad = Counter(r.label_disease for r in records if r.label_disease not in allowed)
        tag = f" [{split}]" if split else ""
        print(f"subset filter{tag}: kept {len(kept)}, dropped {dropped} {dict(bad)}")
    return kept


def yolo_to_anatomy_boxes(result) -> dict[str, tuple[float, float, float, float]]:
    from chd_baseline.chamber_ratios import boxes_from_yolo_dets

    dets = []
    if result.boxes is not None and len(result.boxes):
        xywhn = result.boxes.xywhn.cpu().tolist()
        cls = result.boxes.cls.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        for c, cf, box in zip(cls, confs, xywhn):
            dets.append((int(c), float(cf), float(box[0]), float(box[1]), float(box[2]), float(box[3])))
    return boxes_from_yolo_dets(dets)


def compute_rule_signals(feat: dict[str, float]) -> dict[str, float]:
    """Higher = more abnormal-looking. All in [0,1] scale per signal."""
    chamber_count = feat.get("chamber_count", 0)
    ivs = feat.get("ivs_present", 0)
    crux = feat.get("crux_present", 0)

    signals = {
        "chamber_lt4": float(chamber_count < 4),
        "chamber_le2": float(chamber_count <= 2),
        "ivs_missing": float(ivs < 0.5),
        "crux_missing": float(crux < 0.5),
        "chamber_count": float(chamber_count),
        "ivs_present": float(ivs),
        "crux_present": float(crux),
    }
    # weighted vote score in [0,1]
    votes = (
        signals["chamber_lt4"] * 1.0
        + signals["chamber_le2"] * 1.0
        + signals["ivs_missing"] * 1.0
        + signals["crux_missing"] * 0.5
    )
    signals["anomaly_vote"] = votes / 3.5
    return signals


def rule_abnormal_prob(feat: dict[str, float], thresholds: dict[str, float] | None = None) -> float:
    """Return P(abnormal) from rules. Uses train-derived thresholds if given."""
    sig = compute_rule_signals(feat)
    if thresholds is None:
        return sig["anomaly_vote"]

    score = 0.0
    # chamber count below normal train p10 → abnormal
    if feat.get("chamber_count", 0) < thresholds.get("chamber_count_p10", 4):
        score += 0.35
    if feat.get("ivs_present", 0) < 0.5:
        score += 0.35
    if feat.get("crux_present", 0) < 0.5:
        score += 0.15
    if feat.get("chamber_count", 0) <= 2:
        score += 0.25
    lv_rv = feat.get("lv_rv_area_ratio", float("nan"))
    if np.isfinite(lv_rv):
        if lv_rv < thresholds.get("lv_rv_p5", 0.5) or lv_rv > thresholds.get("lv_rv_p95", 1.5):
            score += 0.15
    return min(score, 1.0)


def fit_rule_thresholds(normal_feats: list[dict[str, float]]) -> dict[str, float]:
    if not normal_feats:
        return {"chamber_count_p10": 3.0, "lv_rv_p5": 0.5, "lv_rv_p95": 1.5}
    cc = [f["chamber_count"] for f in normal_feats]
    ratios = [f["lv_rv_area_ratio"] for f in normal_feats if np.isfinite(f.get("lv_rv_area_ratio", np.nan))]
    return {
        "chamber_count_p10": float(np.percentile(cc, 10)),
        "lv_rv_p5": float(np.percentile(ratios, 5)) if ratios else 0.5,
        "lv_rv_p95": float(np.percentile(ratios, 95)) if ratios else 1.5,
    }


RULE_FEATURE_NAMES = [
    "chamber_count",
    "ivs_present",
    "crux_present",
    "lv_present",
    "rv_present",
    "la_present",
    "ra_present",
    "chamber_lt4",
    "chamber_le2",
    "ivs_missing",
    "crux_missing",
    "anomaly_vote",
    "lv_rv_area_ratio",
    "left_right_area_ratio",
    "ivs_vent_area_ratio",
    "vent_area_total",
]


def feat_vector(feat: dict[str, float]) -> np.ndarray:
    sig = compute_rule_signals(feat)
    merged = {**feat, **sig}
    return np.array([merged.get(n, np.nan) for n in RULE_FEATURE_NAMES], dtype=np.float64)


def collect_rows(
    split: str,
    records: list[Record],
    yolo_model,
    args: argparse.Namespace,
) -> list[SampleRow]:
    rows: list[SampleRow] = []
    batch_size = args.batch_size

    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        paths = [str(r.image_path) for r in chunk]
        try:
            results = yolo_model.predict(
                source=paths,
                conf=args.yolo_conf,
                imgsz=args.yolo_imgsz,
                device=args.device,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            print(f"  WARN: YOLO batch failed ({exc}); per-image fallback", flush=True)
            results = []
            for p in paths:
                try:
                    results.append(yolo_model.predict(
                        source=[p], conf=args.yolo_conf, imgsz=args.yolo_imgsz,
                        device=args.device, verbose=False, stream=False,
                    )[0])
                except Exception:
                    results.append(None)
        for rec, res in zip(chunk, results):
            if res is None:
                continue
            try:
                boxes = yolo_to_anatomy_boxes(res)
                feat = extract_chamber_features(boxes).values
            except Exception:
                continue
            rows.append(SampleRow(
                sample_id=rec.sample_id,
                disease=rec.label_disease,
                label=rec.label_binary,
                split=split,
                boxes=boxes,
                feat=feat,
            ))

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    return rows


def eval_split(
    name: str,
    rows: list[SampleRow],
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_true = np.array([r.label for r in rows], dtype=np.int64)
    metrics = binary_metrics_at_threshold(y_true, y_prob, threshold)
    cm = metrics["confusion_matrix"]
    print(f"\n=== {name} (n={len(rows)}, thr={threshold:.3f}) ===")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"混淆矩阵 [行=真实, 列=预测] 正常/异常:\n  {cm}")
    return metrics


def per_disease_recall(rows: list[SampleRow], y_pred: np.ndarray) -> dict[str, dict]:
    by_dis: dict[str, list[int]] = defaultdict(list)
    for r, p in zip(rows, y_pred):
        by_dis[r.disease].append(int(p == r.label))
    out = {}
    for dis, hits in sorted(by_dis.items()):
        cn = DISEASE_CN.get(dis, dis)
        out[dis] = {
            "cn": cn,
            "n": len(hits),
            "correct": sum(hits),
            "acc": sum(hits) / len(hits) if hits else 0.0,
        }
    return out


def main() -> int:
    args = parse_args()
    if args.yolo_weights is None or not Path(args.yolo_weights).is_file():
        print(
            "ERROR: pass --yolo-weights (checkpoint is not in this repository)",
            file=sys.stderr,
        )
        return 1

    try:
        from dataset import load_split  # private split helper; not shipped
    except ImportError:
        print(
            "detect_rule CLI is not part of the paper recipe. "
            "Use precomputed YOLO tag caches instead of running this detector loop.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading YOLO: {args.yolo_weights}")
    yolo_model = load_yolo_model(args.yolo_weights)

    disease_filter = parse_disease_filter(args.include_diseases)
    if disease_filter:
        names = [DISEASE_CN.get(d, d) for d in sorted(disease_filter)]
        print(f"Subset: {', '.join(names)}")
    else:
        print("Subset: 全部病种")

    all_rows: dict[str, list[SampleRow]] = {}
    for split in ("train", "val", "test"):
        recs = load_split(split, args.processed_dir)
        if disease_filter:
            recs = filter_records_by_disease(recs, disease_filter, split=split)
        rows = collect_rows(split, recs, yolo_model, args)
        n_norm = sum(1 for r in rows if r.label == 0)
        n_abn = sum(1 for r in rows if r.label == 1)
        print(f"{split}: {len(rows)}  正常={n_norm}  异常={n_abn}  病种={dict(Counter(r.disease for r in rows))}")
        all_rows[split] = rows

    train_norm_feats = [r.feat for r in all_rows["train"] if r.label == 0]
    thresholds = fit_rule_thresholds(train_norm_feats)

    # --- Rule baseline ---
    rule_train_prob = np.array([rule_abnormal_prob(r.feat, thresholds) for r in all_rows["train"]])
    rule_val_prob = np.array([rule_abnormal_prob(r.feat, thresholds) for r in all_rows["val"]])
    rule_test_prob = np.array([rule_abnormal_prob(r.feat, thresholds) for r in all_rows["test"]])

    thr_rule, val_rule_m = find_best_binary_threshold(
        np.array([r.label for r in all_rows["val"]]),
        rule_val_prob,
    )
    test_rule_m = eval_split("Test RULE", all_rows["test"], rule_test_prob, thr_rule)
    rule_pred = (rule_test_prob >= thr_rule).astype(np.int64)
    rule_by_dis = per_disease_recall(all_rows["test"], rule_pred)

    print("\n--- 规则分病种（测试集，预测是否正确）---")
    for dis, info in rule_by_dis.items():
        print(f"  {info['cn']:16s}  n={info['n']:4d}  acc={info['acc']:.3f}")

    results: dict = {
        "task": "binary_screening",
        "subset": sorted(disease_filter) if disease_filter else "all",
        "subset_cn": [DISEASE_CN.get(d, d) for d in sorted(disease_filter)] if disease_filter else "all",
        "label_source": "yolo_all",
        "yolo_conf": args.yolo_conf,
        "rule_thresholds": thresholds,
        "rule_val_threshold": thr_rule,
        "rule_description": (
            "异常票：四腔不齐(腔室<4)、腔室≤2、室间隔缺失、十字交叉缺失、"
            "左/右心室面积比偏离正常训练集分位"
        ),
        "test_rule": test_rule_m,
        "test_rule_per_disease": rule_by_dis,
    }

    # --- LR (optional) ---
    if not args.rules_only:
        X_train = np.stack([feat_vector(r.feat) for r in all_rows["train"]])
        y_train = np.array([r.label for r in all_rows["train"]], dtype=np.int64)
        X_val = np.stack([feat_vector(r.feat) for r in all_rows["val"]])
        y_val = np.array([r.label for r in all_rows["val"]], dtype=np.int64)
        X_test = np.stack([feat_vector(r.feat) for r in all_rows["test"]])
        y_test = np.array([r.label for r in all_rows["test"]], dtype=np.int64)

        X_tr, fill = impute_features(X_train)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", _make_logistic_regression()),
        ])
        clf.fit(X_tr, y_train)

        def lr_prob(X: np.ndarray) -> np.ndarray:
            Xi, _ = impute_features(X, fill)
            return clf.predict_proba(Xi)[:, 1]

        lr_val_prob = lr_prob(X_val)
        thr_lr, _ = find_best_binary_threshold(y_val, lr_val_prob)
        lr_test_prob = lr_prob(X_test)
        test_lr_m = eval_split("Test LR", all_rows["test"], lr_test_prob, thr_lr)
        results["test_lr"] = test_lr_m
        results["lr_val_threshold"] = thr_lr
        results["feature_names"] = RULE_FEATURE_NAMES

        bundle = {
            "clf": clf,
            "impute_fill": fill,
            "rule_thresholds": thresholds,
            "lr_val_threshold": thr_lr,
            "rule_val_threshold": thr_rule,
            "feature_names": RULE_FEATURE_NAMES,
            "subset": sorted(disease_filter) if disease_filter else "all",
            "yolo_conf": args.yolo_conf,
        }
        bundle_path = args.output_dir / "train_bundle.pkl"
        bundle_path.write_bytes(pickle.dumps(bundle))
        print(f"Saved LR bundle → {bundle_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")

    print("\n--- 规则说明 ---")
    print("正常：四腔（左/右心室、左/右心房）+ 室间隔 + 十字交叉 检测较完整")
    print("异常：上述任一明显缺损 → 计异常票（单心室单心房常腔室≤2且间隔缺）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
