#!/usr/bin/env python3
"""Domain-adversarial components (GRL + domain classifier + λ schedule)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.alpha)


class DomainClassifier(nn.Module):
    """Binary domain head: private=0 vs cardium=1."""

    def __init__(self, d_in: int, hidden: int = 128, n_domains: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dann_lambda(progress: float, lambda_max: float = 0.3) -> float:
    """Standard DANN schedule: ramp GRL strength with training progress p in [0,1]."""
    p = max(0.0, min(1.0, float(progress)))
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def domain_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def domain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.long())
