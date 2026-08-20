"""Evaluation metrics for CHD screening and diagnosis."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

TaskName = Literal["binary", "multiclass"]


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def find_best_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, dict]:
    """Pick threshold on val that maximizes F1."""
    best_t = 0.5
    best_f1 = -1.0
    best_metrics: dict = {}
    for t in np.linspace(0.05, 0.95, 91):
        pred = (y_prob >= t).astype(np.int64)
        m = _binary_metrics(y_true, pred, y_prob)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def find_youden_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Val threshold maximizing Youden J = sensitivity + specificity - 1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, ths = roc_curve(y_true, y_prob)
    if len(ths) == 0:
        return 0.5
    j = tpr[: len(ths)] - fpr[: len(ths)]
    return float(ths[int(np.argmax(j))])


def threshold_at_min_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    target_spec: float = 0.90,
) -> float:
    """Most sensitive threshold on val whose specificity ≥ target_spec (screening OP).

    Searches unique score cutpoints plus a dense grid on (0,1]. A previous coarse
    grid ``linspace(0.99, 0.01, 99)`` missed thr∈(0.99,1) and silently fell back to
    thr=1.0 (Sens=0) for overconfident models — that artifact is fixed here.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_neg = int((y_true == 0).sum())
    if n_neg == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    # High→low: keep the most sensitive thr that still meets Spec.
    grid = np.unique(
        np.concatenate(
            [
                y_prob.astype(np.float64),
                np.linspace(0.999, 0.001, 999),
                np.asarray([0.0, 1.0], dtype=np.float64),
            ]
        )
    )[::-1]
    best_t = 1.0
    best_sens = -1.0
    for t in grid:
        pred = (y_prob >= t).astype(np.int64)
        tn = int(((y_true == 0) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        if spec + 1e-12 >= float(target_spec) and sens >= best_sens:
            best_sens = sens
            best_t = float(t)
    return best_t


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    target_spec: float = 0.90,
    threshold: float | None = None,
) -> dict:
    """Sensitivity when specificity is constrained (default Spec≥90%).

    If ``threshold`` is None, pick thr on the same split (ROC-style report).
    Prefer: pick thr on val via ``threshold_at_min_specificity``, then call on test.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    thr = float(threshold) if threshold is not None else threshold_at_min_specificity(
        y_true, y_prob, target_spec=target_spec,
    )
    m = binary_metrics_at_threshold(y_true, y_prob, thr)
    return {
        "threshold": thr,
        "target_specificity": float(target_spec),
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "f1": m["f1"],
        "achieved_spec_ok": (
            m["specificity"] is not None and m["specificity"] + 1e-12 >= float(target_spec)
        ),
    }


def patient_metrics_at_threshold_modes(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
) -> dict[str, dict]:
    """Report test patient metrics under three operating points + Sens@90%Spec."""
    thr_f1, _ = find_best_binary_threshold(y_val, prob_val)
    thr_youden = find_youden_binary_threshold(y_val, prob_val)
    thr_spec90 = threshold_at_min_specificity(y_val, prob_val, target_spec=0.90)
    out: dict[str, dict] = {}
    for name, thr in (
        ("val_f1_tuned", thr_f1),
        ("fixed_0.5", 0.5),
        ("youden", thr_youden),
        ("sens_at_spec_0.90", thr_spec90),
    ):
        m = binary_metrics_at_threshold(y_test, prob_test, float(thr))
        out[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}
        out[name]["threshold"] = float(thr)
        if name == "sens_at_spec_0.90":
            out[name]["target_specificity"] = 0.90
            # Also record val Spec at the chosen thr (calibration fidelity)
            m_val = binary_metrics_at_threshold(y_val, prob_val, float(thr))
            out[name]["val_specificity"] = m_val["specificity"]
    return out


def _subset_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """Metrics on a patient subset; AUC only if both classes present."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "n_pos": 0, "n_neg": 0}
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    pred = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
    out = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sensitivity": float(sens) if n_pos > 0 else None,
        "specificity": float(spec) if n_neg > 0 else None,
        "fpr": float(fpr) if n_neg > 0 else None,
        "auc": _safe_auc(y_true, y_prob) if n_pos > 0 and n_neg > 0 else None,
        "f1": float(f1_score(y_true, pred, zero_division=0)) if n_pos > 0 and n_neg > 0 else None,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    return out


def stratified_patient_screening_metrics(
    y_test: np.ndarray,
    prob_test: np.ndarray,
    pids_test: list[str] | np.ndarray,
    patient_info: dict[str, dict],
    *,
    thresholds: dict[str, float],
) -> dict[str, dict[str, dict]]:
    """Stratified test metrics for homologous / cross-device screening.

    Strata (patient-level):
      - all: full mixed test
      - screening_ood: new-device normals + homologous abnorm (excludes midlate etc.)
      - abnorm_homologous: label==1 only → sensitivity / detection rate
      - norm_new_device: new-device normals only → specificity / FPR
      - norm_all: all test normals
    """
    y = np.asarray(y_test, dtype=np.int64)
    p = np.asarray(prob_test, dtype=np.float64)
    pids = [str(x) for x in pids_test]

    def _mask(name: str) -> np.ndarray:
        m = np.zeros(len(pids), dtype=bool)
        for i, pid in enumerate(pids):
            info = patient_info.get(pid, {})
            lab = int(info.get("label", y[i]))
            new_norm = bool(info.get("is_new_device_norm", False))
            hom_abn = bool(info.get("is_homologous_abnorm", lab == 1))
            if name == "all":
                m[i] = True
            elif name == "screening_ood":
                m[i] = (lab == 1 and hom_abn) or (lab == 0 and new_norm)
            elif name == "abnorm_homologous":
                m[i] = lab == 1 and hom_abn
            elif name == "norm_new_device":
                m[i] = lab == 0 and new_norm
            elif name == "norm_all":
                m[i] = lab == 0
        return m

    strata = (
        "all",
        "screening_ood",
        "abnorm_homologous",
        "norm_new_device",
        "norm_all",
    )
    out: dict[str, dict[str, dict]] = {}
    for stratum in strata:
        mask = _mask(stratum)
        ys, ps = y[mask], p[mask]
        out[stratum] = {}
        for mode, thr in thresholds.items():
            out[stratum][mode] = _subset_binary_metrics(ys, ps, float(thr))
            out[stratum][mode]["n_patients"] = int(mask.sum())
    return out


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    pred = (y_prob >= threshold).astype(np.int64)
    return _binary_metrics(y_true, pred, y_prob)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": _safe_auc(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "confusion_matrix": cm.tolist(),
    }


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            str(c): float(f1_score(y_true, y_pred, labels=[c], average="macro", zero_division=0))
            for c in range(num_classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    task: TaskName,
    num_classes: int,
    *,
    return_group_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, preds, probs = [], [], []
    group_ids: list[str] = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].cpu().numpy()
        logits = model(x)
        if task == "binary":
            if logits.shape[-1] > 1:
                prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                prob = torch.sigmoid(logits[:, 0]).cpu().numpy()
            pred = (prob >= 0.5).astype(np.int64)
            ys.append(y)
            preds.append(pred)
            probs.append(prob)
        else:
            prob = F.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            ys.append(y)
            preds.append(pred)
            probs.append(prob)
        if return_group_ids:
            ids = batch[2]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            group_ids.extend(list(ids))

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    if return_group_ids:
        return y_true, y_pred, y_prob, np.asarray(group_ids)
    return y_true, y_pred, y_prob


def aggregate_patient_max_confidence(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Patient-level aggregation: one score per patient = max abnormal confidence."""
    from collections import defaultdict

    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for pid, yt, yp in zip(patient_ids, y_true, y_prob):
        groups[str(pid)].append((int(yt), float(yp)))

    ordered = sorted(groups.keys())
    y_true_p = np.empty(len(ordered), dtype=np.int64)
    y_prob_p = np.empty(len(ordered), dtype=np.float64)
    images_per_patient = []

    for i, pid in enumerate(ordered):
        items = groups[pid]
        labels = {t for t, _ in items}
        if len(labels) > 1:
            raise ValueError(f"Inconsistent labels for patient {pid}: {labels}")
        y_true_p[i] = items[0][0]
        y_prob_p[i] = max(p for _, p in items)
        images_per_patient.append(len(items))

    y_pred_p = (y_prob_p >= threshold).astype(np.int64)
    meta = {
        "aggregation": "max_abnormal_confidence",
        "threshold": threshold,
        "num_patients": len(ordered),
        "avg_images_per_patient": float(np.mean(images_per_patient)),
        "max_images_per_patient": int(max(images_per_patient)),
    }
    return y_true_p, y_pred_p, y_prob_p, meta


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    task: TaskName,
    num_classes: int,
) -> dict:
    if task == "binary":
        return _binary_metrics(y_true, y_pred, y_prob)
    return _multiclass_metrics(y_true, y_pred, num_classes)
