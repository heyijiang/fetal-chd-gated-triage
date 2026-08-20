"""LoRA utilities (no peft dependency)."""

from __future__ import annotations

import re

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def inject_lora(
    model: nn.Module,
    target_pattern: str = "mlp.c_proj",
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> list[str]:
    """Replace matching nn.Linear modules with LoRALinear."""
    pattern = re.compile(target_pattern)
    replaced: list[str] = []
    modules = dict(model.named_modules())

    for name, module in list(modules.items()):
        if not isinstance(module, nn.Linear):
            continue
        if not pattern.search(name):
            continue

        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if parent_name == "" else modules[parent_name]
        lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
        replaced.append(name)

    return replaced


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_a.weight, module.lora_b.weight])
    return params


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
