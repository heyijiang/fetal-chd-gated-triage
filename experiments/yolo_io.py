"""Detector I/O helpers (Ultralytics backend). Weights are NOT shipped; pass --yolo-weights or YOLO_WEIGHTS.

Main paper fusion runs with precomputed embedding/tag caches (--embed-load-only)
and never needs a detector checkpoint.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression


class YoloWeightsNotProvided(FileNotFoundError):
    """Raised when a detector checkpoint is requested but not supplied."""


def resolve_weights(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("YOLO_WEIGHTS") or ""
    p = Path(raw) if raw else Path()
    if not p.is_file():
        raise YoloWeightsNotProvided(
            "YOLO weights are not included in this repository. "
            "For tagging new images, pass --yolo-weights / set YOLO_WEIGHTS. "
            "Published fusion tables use --embed-load-only on cached embeddings."
        )
    return p


def load_yolo_model(weights: Path):
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("pip install ultralytics") from e
    return YOLO(str(weights))


def make_logistic_regression() -> LogisticRegression:
    kwargs: dict = {
        "max_iter": 2000,
        "class_weight": "balanced",
        "solver": "lbfgs",
    }
    if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


# Aliases used by the original training scripts.
_make_logistic_regression = make_logistic_regression
