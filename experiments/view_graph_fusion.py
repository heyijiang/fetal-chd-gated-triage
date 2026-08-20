#!/usr/bin/env python3
"""Cross-view fusion with optional anatomy-linked attention masks (ALVG).

``graph_transformer``: generic cross-view imputation + Transformer.
``anatomy_graph`` (ALVG): same stack but message passing / self-attention
restricted to view pairs that share YOLO-detected cardiac structures
(e.g., LV in four-chamber and LVOT).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard CARDIUM slot order (must match masvf_view_token_fusion.VIEWS).
DEFAULT_VIEW_NAMES: tuple[str, ...] = ("four_chamber", "lvot", "rvot", "vvt")


GRAPH_ADJ_MODES = ("anatomy", "random", "full", "mean")
_GRAPH_ADJ_ALIASES = {
    "fixed": "anatomy",
    "alvg": "anatomy",
    "fully_connected": "full",
    "fully-connected": "full",
    "fc": "full",
    "none": "mean",
    "no_graph": "mean",
    "no-graph": "mean",
    "nograph": "mean",
}


def build_anatomy_adjacency(
    view_names: tuple[str, ...] = DEFAULT_VIEW_NAMES,
) -> np.ndarray:
    """Return bool [V, V] adjacency: True if views share ≥1 anatomy structure ID."""
    from agcd.view_anatomy_stats import VIEW_KEYS

    n = len(view_names)
    adj = np.eye(n, dtype=bool)
    for i, vi in enumerate(view_names):
        ki = set(VIEW_KEYS.get(vi, ()))
        for j, vj in enumerate(view_names):
            if i == j:
                continue
            kj = set(VIEW_KEYS.get(vj, ()))
            if ki & kj:
                adj[i, j] = True
    return adj


def resolve_graph_adjacency(
    mode: str,
    view_names: tuple[str, ...] = DEFAULT_VIEW_NAMES,
    seed: int = 42,
) -> dict:
    """Anatomy / random / fully-connected / mean-pool imputation control.

    ``random`` permutes node labels of the *anatomical* graph (same degree
    sequence, shuffled pairing). This is a structure control, not a claim
    that anatomy edges outperform a random graph.
    ``mean`` drops graph attention: missing slots are filled with the mean
    of present slots, then the Transformer still runs without an anatomy mask.
    """
    raw = str(mode or "anatomy").lower().strip()
    canon = _GRAPH_ADJ_ALIASES.get(raw, raw)
    if canon not in GRAPH_ADJ_MODES:
        raise ValueError(f"unknown graph-adj {mode!r}; expected {GRAPH_ADJ_MODES}")
    anat = build_anatomy_adjacency(view_names)
    n = int(anat.shape[0])
    perm = None
    if canon == "anatomy":
        adj, impute = anat, "attn"
    elif canon == "random":
        rng = np.random.default_rng(int(seed) + 9173)
        perm = rng.permutation(n)
        adj, impute = anat[np.ix_(perm, perm)], "attn"
        perm = perm.tolist()
    elif canon == "full":
        adj, impute = np.ones((n, n), dtype=bool), "attn"
    else:
        adj, impute = None, "mean"
    return {
        "mode": canon,
        "adjacency": adj,
        "impute_mode": impute,
        "perm": perm,
        "n_offdiag": int(adj.sum() - n) if adj is not None else 0,
    }


def anatomy_attn_mask(
    view_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Transformer attn mask: -inf where views are not anatomically linked."""
    adj = build_anatomy_adjacency(view_names)
    mask = torch.zeros(len(view_names), len(view_names), device=device, dtype=dtype)
    mask[~torch.as_tensor(adj, device=device)] = float("-inf")
    return mask


class ViewGraphImputer(nn.Module):
    """One cross-attention layer: missing views query present views."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_m0: int = 3,
        dropout: float = 0.1,
        *,
        anatomy_adj: np.ndarray | None = None,
        impute_mode: str = "attn",
    ):
        super().__init__()
        self.n_m0 = int(n_m0)
        self.impute_mode = str(impute_mode or "attn").lower()
        self.register_buffer(
            "anatomy_adj",
            torch.as_tensor(anatomy_adj, dtype=torch.bool)
            if anatomy_adj is not None
            else None,
            persistent=False,
        )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        if self.n_m0 > 0:
            self.geom_gate = nn.Linear(self.n_m0, d_model)
        else:
            self.geom_gate = None

    def _geom_bias(self, raw_tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor | None:
        if self.geom_gate is None or raw_tokens.shape[-1] < self.n_m0:
            return None
        geom = raw_tokens[..., : self.n_m0]
        bias = self.geom_gate(geom) * present.unsqueeze(-1).float()
        return bias

    def _filter_present(
        self,
        miss_idx: torch.Tensor,
        pres_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.anatomy_adj is None or miss_idx.numel() == 0:
            return pres_idx
        adj = self.anatomy_adj
        allowed = []
        for mi in miss_idx.tolist():
            row = adj[mi]
            for pi in pres_idx.tolist():
                if bool(row[pi].item()):
                    allowed.append(pi)
        if not allowed:
            return pres_idx
        return torch.as_tensor(sorted(set(allowed)), device=miss_idx.device, dtype=pres_idx.dtype)

    def forward(
        self,
        x: torch.Tensor,
        present: torch.Tensor,
        raw_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill missing slots; return (filled_x, imputed_mask)."""
        imputed = (~present).clone()
        if not imputed.any():
            return x, imputed

        geom_bias = self._geom_bias(raw_tokens, present)
        if geom_bias is not None:
            x = x + geom_bias

        out = x.clone()
        for bi in range(x.shape[0]):
            pres_idx = present[bi].nonzero(as_tuple=False).squeeze(-1)
            miss_idx = (~present[bi]).nonzero(as_tuple=False).squeeze(-1)
            if pres_idx.numel() == 0 or miss_idx.numel() == 0:
                continue
            for mi in miss_idx.tolist():
                pres_use = self._filter_present(
                    torch.as_tensor([mi], device=x.device),
                    pres_idx,
                )
                if pres_use.numel() == 0:
                    continue
                if self.impute_mode == "mean":
                    out[bi, mi] = x[bi, pres_use].mean(dim=0)
                    continue
                q = x[bi : bi + 1, mi : mi + 1]
                kv = x[bi : bi + 1, pres_use]
                attn_out, _ = self.attn(q, kv, kv, need_weights=False)
                attn_out = self.norm(attn_out + q)
                attn_out = self.ff_norm(self.ff(attn_out) + attn_out)
                out[bi, mi] = attn_out.squeeze(0).squeeze(0)
        return out, imputed


class ViewGraphFusion(nn.Module):
    """Graph imputation + view-token Transformer + present-masked readout."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.2,
        n_views: int = 4,
        n_m0: int = 3,
        use_view_embed: bool = True,
        pool_mask: str = "present",
        graph_layers: int = 1,
        *,
        use_anatomy_adj: bool = False,
        encoder_anatomy_mask: bool = True,
        view_names: tuple[str, ...] = DEFAULT_VIEW_NAMES,
        adjacency: np.ndarray | None = None,
        impute_mode: str = "attn",
    ):
        super().__init__()
        self.n_views = n_views
        self.n_m0 = int(n_m0)
        self.use_view_embed = bool(use_view_embed)
        self.pool_mask = str(pool_mask or "present").lower()
        self.impute_mode = str(impute_mode or "attn").lower()
        self.encoder_anatomy_mask = bool(encoder_anatomy_mask)
        self.view_names = tuple(view_names[:n_views])
        if adjacency is not None:
            adj_np = np.asarray(adjacency, dtype=bool)
        elif use_anatomy_adj:
            adj_np = build_anatomy_adjacency(self.view_names)
        else:
            adj_np = None
        self.use_anatomy_adj = adj_np is not None
        self.register_buffer(
            "_anatomy_adj",
            torch.as_tensor(adj_np, dtype=torch.bool) if adj_np is not None else None,
            persistent=False,
        )
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.view_embed = nn.Embedding(n_views, d_model)
        self.imputed_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        imputer_adj = None if self.impute_mode == "mean" else adj_np
        self.imputers = nn.ModuleList([
            ViewGraphImputer(
                d_model, n_heads=n_heads, n_m0=n_m0, dropout=dropout,
                anatomy_adj=imputer_adj,
                impute_mode=self.impute_mode,
            )
            for _ in range(max(1, int(graph_layers)))
        ])
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def _attn_mask(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if not self.use_anatomy_adj or not self.encoder_anatomy_mask or self._anatomy_adj is None:
            return None
        mask = torch.zeros(self.n_views, self.n_views, device=device, dtype=dtype)
        mask[~self._anatomy_adj] = float("-inf")
        return mask

    def _embed(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.in_norm(tokens))
        if self.use_view_embed:
            x = x + self.view_embed.weight.unsqueeze(0)
        return x

    def impute(
        self,
        tokens: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._embed(tokens, present)
        imputed = torch.zeros_like(present)
        for imp in self.imputers:
            x, imp_mask = imp(x, present | imputed, tokens)
            imputed = imputed | imp_mask
        x = torch.where(
            imputed.unsqueeze(-1),
            x + self.imputed_embed,
            x,
        )
        return x, imputed

    def pool_readout(self, h: torch.Tensor, present: torch.Tensor, imputed: torch.Tensor) -> torch.Tensor:
        mask = present | imputed
        if self.pool_mask == "uniform":
            w = torch.full((h.shape[0], h.shape[1]), 1.0 / float(h.shape[1]), device=h.device, dtype=h.dtype)
        else:
            w = mask.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head((h * w.unsqueeze(-1)).sum(dim=1)).squeeze(-1)

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        x, imputed = self.impute(tokens, present)
        attn_mask = self._attn_mask(tokens.device, tokens.dtype)
        h = self.encoder(x, mask=attn_mask)
        return self.pool_readout(h, present, imputed)

    def pooled_features(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        x, imputed = self.impute(tokens, present)
        attn_mask = self._attn_mask(tokens.device, tokens.dtype)
        h = self.encoder(x, mask=attn_mask)
        mask = present | imputed
        w = mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * w.unsqueeze(-1)).sum(dim=1)

    def mvp_loss(
        self,
        tokens: torch.Tensor,
        present: torch.Tensor,
        mask_prob: float = 0.25,
    ) -> torch.Tensor:
        """Masked view prediction on randomly hidden *present* slots."""
        b, v, d = tokens.shape
        if present.sum() <= 1:
            return tokens.new_tensor(0.0)
        x = self._embed(tokens, present)
        targets = x.detach()
        mvp_present = present.clone()
        losses = []
        for bi in range(b):
            idx = mvp_present[bi].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() < 2:
                continue
            n_mask = max(1, int(round(float(mask_prob) * idx.numel())))
            perm = torch.randperm(idx.numel(), device=tokens.device)[:n_mask]
            hide = idx[perm]
            keep = mvp_present[bi].clone()
            keep[hide] = False
            if keep.sum() == 0:
                continue
            xi = x[bi : bi + 1]
            tok_i = tokens[bi : bi + 1]
            keep_b = keep.unsqueeze(0)
            for imp in self.imputers:
                xi, _ = imp(xi, keep_b, tok_i)
            pred = xi[0, hide]
            tgt = targets[bi, hide]
            losses.append(F.mse_loss(pred, tgt))
        if not losses:
            return tokens.new_tensor(0.0)
        return torch.stack(losses).mean()


class ViewStatPoolFusion(nn.Module):
    """Masked mean or max over gated slots. No adjacency prior."""

    def __init__(self, d_in: int, d_model: int = 64, dropout: float = 0.2, *, mode: str = "mean"):
        super().__init__()
        self.mode = str(mode)
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        x = self.proj(tokens)
        m = present.unsqueeze(-1).to(dtype=x.dtype)
        if self.mode == "max":
            fill = torch.finfo(x.dtype).min
            x = x.masked_fill(m == 0, fill)
            pooled = x.max(dim=1).values
            empty = present.sum(dim=1) == 0
            pooled = torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)
        else:
            pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)
