"""CPU smoke test: ALVG / MIL / Transformer forward without images or a detector."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from view_graph_fusion import ViewGraphFusion  # noqa: E402
from masvf_view_token_fusion import AttentionMIL, ViewTokenFusion  # noqa: E402


def test_alvg_forward() -> None:
    model = ViewGraphFusion(
        d_in=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        n_m0=0,
        use_anatomy_adj=True,
    )
    x = torch.randn(2, 4, 32)
    present = torch.tensor([[True, True, True, False], [True, True, True, True]])
    out = model(x, present)
    assert out.shape == (2,)


def test_mil_and_transformer_forward() -> None:
    mil = AttentionMIL(d_in=32, d_model=32, dropout=0.0)
    tr = ViewTokenFusion(d_in=32, d_model=32, n_layers=1, n_heads=4, dropout=0.0)
    x = torch.randn(2, 4, 32)
    present = torch.ones(2, 4, dtype=torch.bool)
    assert mil(x, present).shape == (2,)
    assert tr(x, present).shape == (2,)


if __name__ == "__main__":
    test_alvg_forward()
    test_mil_and_transformer_forward()
    print("ok")
