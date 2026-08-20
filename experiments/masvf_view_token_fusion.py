#!/usr/bin/env python3
"""Track B: replace patient-max with learned view fusion (image-only).

Same C2-b frames (YOLO gate + M0 ‖ FetalCLIP crop). Two fusion levels:

  score_*   (recommended B-v2): shared LR → per-frame score → 4 view scores
             → small MLP / attention. Same inductive bias as C2-b, only fusion changes.
  feature_transformer (B-v1): mid-level [4, D] tokens → Transformer.
             Overfits easily (val≪test gap); kept for ablation.

Usage:
  cd experiments
  CUDA_VISIBLE_DEVICES=0 bash run_masvf_view_token.sh          # CARDIUM
  CUDA_VISIBLE_DEVICES=0 bash run_masvf_view_token_private.sh  # private manifest
  FUSION=cross_plane_cls bash run_cardium_outline_phase_b.sh   # CARD Phase B
"""

from __future__ import annotations

import argparse
import random
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from agcd.fetalclip_embed import (  # noqa: E402
    CARDIUM_CROP_CACHE,
    CARDIUM_CROP_LORA_CACHE,
    FETALCLIP_EMBED_DIM,
    PRIVATE_CROP_CACHE,
    PRIVATE_CROP_LORA_CACHE,
    build_or_load_embeddings,
)
from agcd.plane_features import (  # noqa: E402
    M0_FEATURE_NAMES,
    M0_FEATURE_SUBSETS,
    resolve_m0_feature_names,
)
from agcd.view_anatomy_stats import (  # noqa: E402
    view_anat_frame_dim,
    view_anat_frame_from_tag,
)
from cardium_c2b_coverage_fusion import resolve_tags  # noqa: E402
from cardium_dataset import load_fold_split  # noqa: E402
from chd_baseline.chamber_ratios import impute_features  # noqa: E402
from chd_baseline.metrics import (  # noqa: E402
    binary_metrics_at_threshold,
    find_best_binary_threshold,
    patient_metrics_at_threshold_modes,
    stratified_patient_screening_metrics,
)
from yolo_io import load_yolo_model, resolve_weights  # noqa: E402
from masvf_m0_screening import (  # noqa: E402
    CARDIUM_FEATURE_CACHE,
    PRIVATE_FEATURE_CACHE,
    apply_frame_selection,
    build_cardium_frame_rows,
    build_midlate_frame_rows,
    build_private_frame_rows,
    collect_crop_boxes,
    collect_embed_items,
    filter_view_mode,
    fit_lr,
    print_frame_lr_coef_summary,
    summarize_frame_lr_coefs,
    load_feature_cache,
    load_private_studies,
    load_tags,
    predict_lr,
    rows_to_X,
    split_val_patients,
    standardize_feature_blocks,
)
from real_multiview_bags import (  # noqa: E402
    DOMAIN_CARDIUM,
    DOMAIN_PRIVATE,
    coverage_report,
    print_coverage_report,
    tag_row_domain,
    prefix_patient_id,
    write_coverage_json,
)
from dann_grl import (  # noqa: E402
    DomainClassifier,
    GradientReversal,
    dann_lambda,
    domain_accuracy,
    domain_loss,
)
from view_graph_fusion import ViewGraphFusion, ViewStatPoolFusion, resolve_graph_adjacency  # noqa: E402

VIEWS = ("four_chamber", "lvot", "rvot", "vvt")  # vvt = 3VT (not 3VV)
FALLBACK_VIEW = "fallback"
WITHIN_VIEW_POOLS = (
    "nanmax",
    "mean",
    "score_argmax",
    "split_mean_max",
    "rep_frame",          # one quality frame → full [M0‖CLIP] (同源)
    "m0_mean_clip_rep",   # M0=mean, CLIP=same quality rep frame
    "frame_attn",         # quality-weighted softmax mean over frames in view
)

# Path/folder cues that the frame is color Doppler (血流), not B-mode structure.
# TODO(doppler): 四腔心 B-mode 与彩色多普勒任务不同（形态 vs 血流），但 manifest/标签
# 常把「四腔心切面彩色多普勒」映射进同一个 four_chamber 桶。短期：rep_frame 优先选
# 非多普勒帧；中期：多普勒单独 view 槽或单独筛查头，避免与结构 4view 混池。
_DOPPLER_PATH_MARKERS = (
    "多普勒",
    "four_chamber_color",
    "/color/",
    "_color/",
    "cdfi",
)
# split_mean_max: within-view M0=mean, CLIP=nanmax, then concat
D_FEAT_DEFAULT = len(M0_FEATURE_NAMES) + FETALCLIP_EMBED_DIM


def active_views(fallback: str) -> tuple[str, ...]:
    if fallback == "token":
        return VIEWS + (FALLBACK_VIEW,)
    return VIEWS


class ViewTokenFusion(nn.Module):
    """Mid-level feature Transformer (B-v1): mean-pool over view tokens.

    Ablation switches (TMI E5):
      use_view_embed / missing_fill / pool_mask control view-id, missing fill, pooling.
    """

    def __init__(
        self,
        d_in: int = D_FEAT_DEFAULT,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.2,
        n_views: int = 4,
        use_view_embed: bool = True,
        missing_fill: str = "zero",
        pool_mask: str = "present",
    ):
        super().__init__()
        self.n_views = n_views
        self.use_view_embed = bool(use_view_embed)
        self.missing_fill = str(missing_fill or "zero").lower()
        self.pool_mask = str(pool_mask or "present").lower()
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.view_embed = nn.Embedding(n_views, d_model)
        self.missing = nn.Parameter(torch.zeros(1, 1, d_model))
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

    def _encode(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        b, v, _ = tokens.shape
        x = self.proj(self.in_norm(tokens))
        if self.use_view_embed:
            x = x + self.view_embed.weight.unsqueeze(0)
        if self.missing_fill == "zero":
            fill = torch.zeros(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
            x = torch.where(present.unsqueeze(-1), x, fill.expand(b, v, -1))
        else:
            x = torch.where(present.unsqueeze(-1), x, self.missing.expand(b, v, -1))
        return self.encoder(x)

    def pooled_features(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        b, v, _ = tokens.shape
        h = self._encode(tokens, present)
        if self.pool_mask == "uniform":
            w = torch.full((b, v), 1.0 / float(v), device=h.device, dtype=h.dtype)
        else:
            w = present.float()
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled_features(tokens, present)).squeeze(-1)


class ViewTokenFusionCLS(nn.Module):
    """Cross-plane CLS readout (Phase B): prepend learnable CLS, read after Transformer."""

    def __init__(
        self,
        d_in: int = D_FEAT_DEFAULT,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        n_views: int = 4,
    ):
        super().__init__()
        self.n_views = n_views
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.view_embed = nn.Embedding(n_views, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.missing = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        b, v, _ = tokens.shape
        x = self.proj(self.in_norm(tokens)) + self.view_embed.weight.unsqueeze(0)
        x = torch.where(present.unsqueeze(-1), x, self.missing.expand(b, v, -1))
        cls = self.cls_token.expand(b, -1, -1)
        h = self.encoder(torch.cat([cls, x], dim=1))
        return self.head(h[:, 0]).squeeze(-1)


class ScoreMLPFusion(nn.Module):
    """View scores (+ present mask) → patient logit. Replaces patient-max."""

    def __init__(self, hidden: int = 32, n_views: int = 4):
        super().__init__()
        self.n_views = n_views
        self.net = nn.Sequential(
            nn.Linear(n_views * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, scores: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        s = scores * present.float()
        x = torch.cat([s, present.float()], dim=-1)
        return self.net(x).squeeze(-1)


class ScoreAttnFusion(nn.Module):
    """Learned soft aggregation over view scores (soft max / gated mean)."""

    def __init__(self, n_views: int = 4):
        super().__init__()
        self.n_views = n_views
        self.gate = nn.Linear(2, 1)

    def forward(self, scores: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        feat = torch.stack([scores, present.float()], dim=-1)
        logits = self.gate(feat).squeeze(-1)
        logits = logits.masked_fill(~present, -1e4)
        w = torch.softmax(logits, dim=-1)
        p = (w * scores).sum(dim=-1).clamp(1e-4, 1 - 1e-4)
        return torch.log(p / (1 - p))


class ViewBiLSTMFusion(nn.Module):
    """Sequence control: BiLSTM over the 4 view tokens (fixed order + missing fill).

    Same token interface as Transformer/MIL feature heads; models cross-view order
    rather than intra-video temporal dynamics (cf. TPA-style temporal extractors).
    """

    def __init__(
        self,
        d_in: int = D_FEAT_DEFAULT,
        d_model: int = 64,
        dropout: float = 0.2,
        n_views: int = 4,
    ):
        super().__init__()
        self.n_views = n_views
        self.in_norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_model)
        self.view_embed = nn.Embedding(n_views, d_model)
        self.missing = nn.Parameter(torch.zeros(1, 1, d_model))
        hid = max(d_model // 2, 16)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hid,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hid * 2),
            nn.Dropout(dropout),
            nn.Linear(hid * 2, 1),
        )

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        b, v, _ = tokens.shape
        x = self.proj(self.in_norm(tokens)) + self.view_embed.weight.unsqueeze(0)
        x = torch.where(present.unsqueeze(-1), x, self.missing.expand(b, v, -1))
        h, _ = self.lstm(x)
        w = present.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head((h * w.unsqueeze(-1)).sum(dim=1)).squeeze(-1)


class AttentionMIL(nn.Module):
    """Classic Attention-MIL over variable-length frame bag (no view identity)."""

    def __init__(self, d_in: int = D_FEAT_DEFAULT, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, tokens: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        h = self.proj(tokens)
        logits = self.attn(h).squeeze(-1)
        logits = logits.masked_fill(~present, -1e4)
        w = torch.softmax(logits, dim=-1)
        bag = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.head(bag).squeeze(-1)


class PatientTokenDS(Dataset):
    def __init__(self, tokens, present, labels, domains=None):
        self.tokens = tokens
        self.present = present
        self.labels = labels
        self.domains = domains

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.domains is not None:
            return self.tokens[i], self.present[i], self.labels[i], self.domains[i]
        return self.tokens[i], self.present[i], self.labels[i]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MASVF view fusion (Track B)")
    p.add_argument("--cohort", default="cardium", choices=["cardium", "private"])
    p.add_argument("--device", default="0")
    p.add_argument("--folds", default="1,2,3")
    p.add_argument("--cardium-processed", type=Path,
                   default=PROJECT_ROOT / "CARDIUM dataset" / "processed")
    p.add_argument("--manifest", type=Path,
                   default=PROJECT_ROOT / "data" / "study_screening" / "manifest.jsonl")
    p.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    p.add_argument("--image-tags", type=Path, default=None)
    p.add_argument(
        "--swap-views",
        default="",
        help="Swap two slot names in tags, e.g. rvot,vvt (YOLO confusion stress).",
    )
    p.add_argument("--feature-cache", type=Path, default=None)
    p.add_argument("--fetalclip-cache", type=Path, default=None)
    p.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Homologous/private FetalCLIP LoRA ckpt; embeds go to *_homologous_lora cache",
    )
    p.add_argument(
        "--train-protocol",
        default="private",
        choices=["private", "homologous", "real_joint"],
        help="private=manifest train; homologous=Frankenstein midlate or zy_unused; "
             "real_joint=real zy train normals + private abn + CARDIUM train fold",
    )
    p.add_argument(
        "--homologous-train-norm",
        default="midlate",
        choices=["midlate", "zy_unused", "zy_real", "zy_plus_1view", "cardium_only"],
        help="homologous/real_joint: midlate=Frankenstein synth 4-view bags; "
             "zy_unused/zy_real=manifest train-split real zy bags; "
             "zy_plus_1view=zy train bags + midlate single-plane as 1-view (3 missing) patients; "
             "cardium_only=real_joint normals from CARDIUM train only (no zy train normals)",
    )
    p.add_argument(
        "--joint-cardium-fold",
        type=int,
        default=1,
        help="real_joint: CARDIUM official fold merged into training (1-3)",
    )
    p.add_argument("--norm-corpus", type=Path,
                   default=PROJECT_ROOT / "心脏中晚孕数据")
    p.add_argument("--frames-per-view-norm", type=int, default=0,
                   help="homologous: explicit midlate size. "
                        "multiview mode → n_synth_patients (0=auto balance); "
                        "frame / zy_plus_1view → samples per Chinese view folder "
                        "(0=all frames as 1-view patients); "
                        "zy_unused → optional patient cap (0=all train zy)",
    )
    p.add_argument(
        "--oneview-per-folder",
        type=int,
        default=-1,
        help="zy_plus_1view: override samples per midlate view folder "
             "(-1=use --frames-per-view-norm; 0=all)",
    )
    p.add_argument(
        "--midlate-synth-mode",
        default="multiview",
        choices=["multiview", "frame"],
        help="multiview=each synth norm patient has 4CH+LVOT+RVOT+VVT (fusion); "
             "frame=1 frame→1 patient (legacy image LoRA)",
    )
    p.add_argument(
        "--balance-norm-to-eval",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="homologous: set midlate synth count so train norm:abn ≈ test norm:abn",
    )
    p.add_argument("--max-norm", type=int, default=0,
                   help="homologous: optional midlate global cap before per-view sample")
    p.add_argument("--crop-pad", type=float, default=0.12)
    p.add_argument("--val-patient-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--appearance-dim",
        type=int,
        default=0,
        help="Appearance embed dim (0=auto from cache; FetalCLIP=768, ResNet50=2048)",
    )
    p.add_argument(
        "--embed-load-only",
        action="store_true",
        help="Never call FetalCLIP builder; only load --fetalclip-cache (for ResNet etc.)",
    )
    p.add_argument(
        "--view-decomp",
        action="store_true",
        default=True,
        help="After fusion train: leave-one-view / single-view metrics (default on)",
    )
    p.add_argument(
        "--no-view-decomp",
        action="store_false",
        dest="view_decomp",
        help="Disable leave-one-view / single-view ablation",
    )
    p.add_argument(
        "--fusion",
        default="feature_transformer",
        choices=[
            "score_mlp",
            "score_attn",
            "feature_transformer",
            "graph_transformer",
            "anatomy_graph",
            "cross_plane_cls",
            "attention_mil",
            "view_bilstm",
            "view_mean",
            "view_max",
        ],
        help="feature_transformer/graph_transformer/anatomy_graph/cross_plane_cls/view_bilstm = view tokens; "
             "graph_transformer = cross-view imputation + Transformer; "
             "anatomy_graph (ALVG) = anatomy-masked imputation + Transformer; "
             "attention_mil = bag MIL; score_* = score fusion",
    )
    p.add_argument(
        "--fallback",
        default="drop",
        choices=["drop", "token"],
        help="drop=discard YOLO-miss frames (default); token=keep as fallback view + full-image CLIP",
    )
    p.add_argument("--mil-max-frames", type=int, default=32,
                   help="Attention-MIL: max frames per patient bag")
    p.add_argument(
        "--feat-model",
        default="m1",
        choices=["m0", "m1"],
        help="m0 = anatomy stats only; m1 = M0 ‖ FetalCLIP plane crop",
    )
    p.add_argument(
        "--m0-features",
        default="all",
        choices=sorted(M0_FEATURE_SUBSETS.keys()),
        help="M0 feature subset (see agcd/plane_features.py)",
    )
    p.add_argument(
        "--within-view-pool",
        default="rep_frame",
        choices=list(WITHIN_VIEW_POOLS),
        help="Within-view pool: rep_frame=quality B-mode frame [M0‖CLIP] (recommended); "
             "frame_attn=quality-weighted softmax mean; "
             "m0_mean_clip_rep=M0 mean + that frame CLIP; nanmax/mean/score_argmax/split_mean_max=legacy",
    )
    p.add_argument(
        "--extra-feats",
        default="none",
        choices=["none", "view_anat"],
        help="none=base M0/CLIP; view_anat=concat active-view anatomy stats+pos (3VT vessels etc.)",
    )
    p.add_argument(
        "--clip-norm",
        default="hybrid",
        choices=["hybrid", "joint"],
        help="hybrid=M0 StandardScaler + CLIP L2 only; joint=legacy full StandardScaler",
    )
    p.add_argument(
        "--norm-view-drop",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="Optional: randomly drop normal patients' views to match abn miss rates "
        "(default off; set 1/true to enable)",
    )
    p.add_argument(
        "--norm-view-drop-scale",
        type=float,
        default=1.0,
        help="Multiply abn per-view miss rates when dropping normal views (1.0=match)",
    )
    p.add_argument(
        "--no-view-embed",
        action="store_true",
        help="E5: disable view-identity embeddings in ViewTokenFusion",
    )
    p.add_argument(
        "--missing-fill",
        default="zero",
        choices=["learned", "zero"],
        help="Missing-view fill — zero (Round3 main default) or learned token (E5 ablation)",
    )
    p.add_argument(
        "--pool-mask",
        default="present",
        choices=["present", "uniform"],
        help="E5: present-masked mean (default) vs uniform 1/V pooling (includes missing slots)",
    )
    p.add_argument(
        "--graph-layers",
        type=int,
        default=1,
        help="graph_transformer: cross-attn imputation layers",
    )
    p.add_argument(
        "--dann",
        action="store_true",
        help="Enable domain-adversarial head (private vs CARDIUM) with GRL",
    )
    p.add_argument(
        "--lambda-d-max",
        type=float,
        default=0.3,
        help="DANN: max GRL strength λ_max",
    )
    p.add_argument(
        "--mvp",
        action="store_true",
        help="Enable masked-view prediction auxiliary loss (graph_transformer)",
    )
    p.add_argument(
        "--no-mvp",
        action="store_true",
        help="Disable MVP even for graph/anatomy_graph fusion",
    )
    p.add_argument(
        "--no-anatomy-encoder-mask",
        action="store_true",
        help="ALVG: anatomy mask on imputation only; encoder uses full attention",
    )
    p.add_argument(
        "--graph-adj",
        default="auto",
        choices=["auto", "anatomy", "random", "full", "mean",
                 "fixed", "fully_connected", "none", "no_graph"],
        help="Graph ablation for anatomy_graph / graph_transformer: "
             "auto=anatomy for ALVG else full-present; "
             "random=permute anatomy node labels (seeded); "
             "full=all-to-all among present; "
             "mean=mean-pool present slots (no graph attention)",
    )
    p.add_argument(
        "--lambda-mvp",
        type=float,
        default=0.1,
        help="Weight for masked-view prediction loss",
    )
    p.add_argument(
        "--include-empty-view-patients",
        action="store_true",
        help="Eval: include every manifest test patient (zero-fill if no 4-view frames)",
    )
    p.add_argument(
        "--coverage-report",
        type=Path,
        default=None,
        help="Write real multi-view coverage JSON (real_joint / zy_real)",
    )
    p.add_argument(
        "--eval-norm-swap",
        action="store_true",
        help="E1: after homologous train, also eval on held-out midlate normals + test abnormals "
             "(same-machine vs new-device normal swap)",
    )
    p.add_argument(
        "--eval-cardium",
        action="store_true",
        help="After train, also eval fusion on CARDIUM official fold holdout (no CARDIUM train merge)",
    )
    p.add_argument(
        "--eval-zy",
        action="store_true",
        help="After train, also eval on private zy (norm_4ac) val/test bags (OOD negatives stress)",
    )
    p.add_argument(
        "--zy-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "study_screening" / "manifest.jsonl",
        help="Private manifest used for --eval-zy (zy=norm_4ac normals + private CHD)",
    )
    p.add_argument(
        "--zy-tags",
        type=Path,
        default=PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_private.jsonl",
        help="YOLO tags for --eval-zy",
    )
    p.add_argument(
        "--midlate-eval-n",
        type=int,
        default=0,
        help="E1: held-out midlate normal patients (0=match # of new-device test normals)",
    )
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=None,
                   help="Transformer dropout; default 0.2 (v1) / 0.1 (CLS)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--max-patients", type=int, default=0,
                   help="Smoke: keep at most N patients total (0=all)")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "masvf_view_token")
    p.add_argument("--feat-backend", default="m1_crop", choices=["m0", "m1_crop"])
    p.add_argument("--features-only", action="store_true",
                   help="Require complete M0+FetalCLIP caches (skip YOLO load)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Re-run YOLO for all frames and append to feature cache")
    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--export-badcase-dir",
        type=Path,
        default=None,
        help="If set, export test misclassifications + viewer.html to this directory",
    )
    p.add_argument(
        "--badcase-threshold-mode",
        default="val_f1_tuned",
        choices=["val_f1_tuned", "fixed_0.5", "youden"],
        help="Threshold mode used for FP/FN labeling in badcase export",
    )
    args = p.parse_args()
    if args.fetalclip_cache is None:
        if args.lora_adapter is not None:
            args.fetalclip_cache = (
                PRIVATE_CROP_LORA_CACHE if args.cohort == "private" else CARDIUM_CROP_LORA_CACHE
            )
        else:
            args.fetalclip_cache = PRIVATE_CROP_CACHE if args.cohort == "private" else CARDIUM_CROP_CACHE
    if args.dropout is None:
        if args.fusion == "cross_plane_cls":
            args.dropout = 0.1
        elif args.fusion == "attention_mil":
            args.dropout = 0.2
        elif args.fusion == "view_bilstm":
            args.dropout = 0.2
        else:
            args.dropout = 0.2
    if args.feat_model == "m0":
        args.feat_backend = "m0"
    else:
        args.feat_backend = "m1_crop"
    return args


def _file_fingerprint(path: Path | str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime_unix": int(st.st_mtime),
        "sha256": h.hexdigest(),
    }


def _build_repro_block(args: argparse.Namespace) -> dict:
    """Seed + weight/cache fingerprints for exact reruns."""
    app_weights = None
    if getattr(args, "embed_load_only", False):
        # Infer ImageNet weights from appearance dim / cache name when possible
        cache = str(getattr(args, "fetalclip_cache", "") or "")
        for name, (fname, dim) in (
            ("resnet50", ("resnet50_imagenet_v2.pth", 2048)),
            ("efficientnet_b0", ("efficientnet_b0_imagenet_v1.pth", 1280)),
            ("vit_b_16", ("vit_b_16_imagenet_v1.pth", 768)),
        ):
            if name in cache or int(getattr(args, "appearance_dim", 0) or 0) == dim:
                app_weights = _file_fingerprint(
                    PROJECT_ROOT / "weights" / "torchvision" / fname
                )
                break
    return {
        "seed": int(getattr(args, "seed", 42) or 42),
        "train_protocol": getattr(args, "train_protocol", "private"),
        "fusion": args.fusion,
        "graph_adj": str(getattr(args, "graph_adj", "auto") or "auto"),
        "feat_model": args.feat_model,
        "m0_features": args.m0_features,
        "within_view_pool": args.within_view_pool,
        "fallback": args.fallback,
        "clip_norm": getattr(args, "clip_norm", "hybrid"),
        "appearance_dim": int(getattr(args, "appearance_dim", 0) or 0),
        "embed_load_only": bool(getattr(args, "embed_load_only", False)),
        "lora_adapter": _file_fingerprint(getattr(args, "lora_adapter", None)),
        "fetalclip_cache": _file_fingerprint(getattr(args, "fetalclip_cache", None)),
        "feature_cache": _file_fingerprint(getattr(args, "feature_cache", None)),
        "fetalclip_weights": _file_fingerprint(
            PROJECT_ROOT / "fetalclip" / "FetalCLIP_weights.pt"
        ),
        "appearance_imagenet_weights": app_weights,
        "repro_config": _file_fingerprint(
            EXPERIMENTS_DIR / "configs" / "homologous_miccai_repro.json"
        ),
    }


def feat_dim(args: argparse.Namespace) -> int:
    n_m0 = len(resolve_m0_feature_names(args.m0_features))
    if args.feat_model == "m0":
        base = n_m0
    else:
        app_dim = int(getattr(args, "appearance_dim", 0) or 0)
        if app_dim <= 0:
            app_dim = FETALCLIP_EMBED_DIM
        base = n_m0 + app_dim
    if getattr(args, "extra_feats", "none") == "view_anat":
        return base + view_anat_frame_dim()
    return base


def _infer_appearance_dim(embed_cache: dict | None, args: argparse.Namespace) -> int:
    forced = int(getattr(args, "appearance_dim", 0) or 0)
    dim_counts: dict[int, int] = {}
    if embed_cache:
        # Deduplicate by object id: cache stores sample_id + image_path aliases
        seen: set[int] = set()
        for vec in embed_cache.values():
            oid = id(vec)
            if oid in seen:
                continue
            seen.add(oid)
            d = int(np.asarray(vec).ravel().shape[0])
            dim_counts[d] = dim_counts.get(d, 0) + 1
    if len(dim_counts) > 1:
        raise RuntimeError(
            f"appearance cache has mixed dims {dim_counts} "
            f"(file={getattr(args, 'fetalclip_cache', None)}). "
            "Likely FetalCLIP 768-d rows were appended into an ImageNet cache. "
            "Fix: rm the file and rebuild with "
            "`python -u experiments/build_imagenet_backbone_embed_cache.py --backbones ... --rebuild`, "
            "and always use --embed-load-only for backbone runs."
        )
    cache_dim = next(iter(dim_counts), None)
    if forced > 0:
        if cache_dim is not None and cache_dim != forced:
            raise RuntimeError(
                f"appearance_dim={forced} but cache vectors are {cache_dim}-d "
                f"(cache={getattr(args, 'fetalclip_cache', None)}). "
                "Unset APPEARANCE_DIM for FetalCLIP/LoRA runs, or pass the matching "
                "ImageNet cache. Also: `unset APPEARANCE_DIM EMBED_LOAD_ONLY FETALCLIP_CACHE`."
            )
        return forced
    if cache_dim is not None:
        return cache_dim
    return FETALCLIP_EMBED_DIM


def _guard_imagenet_cache_not_polluted_by_fetalclip(args: argparse.Namespace) -> None:
    """Refuse FetalCLIP encode into a non-768 / ImageNet-named cache path."""
    if getattr(args, "embed_load_only", False):
        return
    app_dim = int(getattr(args, "appearance_dim", 0) or 0)
    cache = str(getattr(args, "fetalclip_cache", "") or "")
    imagenet_hint = any(
        x in cache for x in ("resnet50", "efficientnet_b0", "vit_b_16", "_resnet", "_vit_")
    )
    if app_dim > 0 and app_dim != FETALCLIP_EMBED_DIM:
        raise SystemExit(
            f"ERROR: appearance_dim={app_dim} requires --embed-load-only "
            f"(refusing FetalCLIP encode into {cache!r})."
        )
    if imagenet_hint:
        raise SystemExit(
            f"ERROR: cache path looks like ImageNet ({cache!r}) but --embed-load-only "
            "was not set; FetalCLIP would append 768-d vectors and corrupt the file."
        )



def n_m0_dims(args: argparse.Namespace) -> int:
    return len(resolve_m0_feature_names(args.m0_features))


def n_views_for(args: argparse.Namespace) -> int:
    return len(active_views(args.fallback))


def _device(arg: str) -> torch.device:
    if arg.isdigit():
        return torch.device(f"cuda:{arg}" if torch.cuda.is_available() else "cpu")
    if arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(arg)
    return torch.device("cpu")


def _is_color_doppler_row(r) -> bool:
    """True if frame looks like color Doppler (血流) rather than B-mode structure."""
    meta = getattr(r, "meta", None) or {}
    if meta.get("is_color_doppler") or meta.get("doppler"):
        return True
    path = str(getattr(r, "image_path", "") or "")
    low = path.lower()
    if any(m in path or m in low for m in _DOPPLER_PATH_MARKERS):
        return True
    return False


def _frame_quality_key(r) -> tuple:
    """Higher = better structural representative. Prefer non-Doppler, then plane quality."""
    meta = getattr(r, "meta", None) or {}
    conf = float(getattr(r, "plane_conf", 0.0) or meta.get("plane_conf", 0.0) or 0.0)
    area = float(meta.get("plane_area_norm", 0.0) or 0.0)
    comp = float(meta.get("key_anatomy_completeness", 0.0) or 0.0)
    non_doppler = 0.0 if _is_color_doppler_row(r) else 1.0
    return (non_doppler, conf, area, comp)


def _pick_rep_index(rows_in_view: list) -> int:
    """Index of quality-best frame; prefers B-mode over color Doppler when both exist."""
    return int(max(range(len(rows_in_view)), key=lambda i: _frame_quality_key(rows_in_view[i])))


def _pool_view_matrix(
    arr: np.ndarray,
    pool: str,
    n_m0: int,
    scores: np.ndarray | None,
    rows_in_view: list | None = None,
) -> np.ndarray:
    """Pool [N,D] frames within one view → [D]."""
    if pool in ("rep_frame", "m0_mean_clip_rep"):
        if not rows_in_view or len(rows_in_view) != len(arr):
            raise ValueError(f"{pool} requires rows_in_view aligned with arr")
        # Prefer structural (non-Doppler) subset when available — 4CH B-mode ≠ 4CH Doppler.
        struct_idxs = [i for i, r in enumerate(rows_in_view) if not _is_color_doppler_row(r)]
        use_idxs = struct_idxs if struct_idxs else list(range(len(rows_in_view)))
        use_rows = [rows_in_view[i] for i in use_idxs]
        use_arr = arr[use_idxs]
        rep_local = _pick_rep_index(use_rows)
        if pool == "rep_frame":
            # 同源：该代表帧的完整 [M0‖CLIP]
            return use_arr[rep_local]
        # M0 = mean over (prefer structural) frames; CLIP = same rep frame
        if n_m0 <= 0 or n_m0 >= use_arr.shape[1]:
            return use_arr[rep_local]
        with np.errstate(all="ignore"):
            m0 = np.nanmean(use_arr[:, :n_m0], axis=0)
        clip = use_arr[rep_local, n_m0:]
        return np.concatenate([m0, clip], axis=0)
    if pool == "frame_attn":
        if not rows_in_view or len(rows_in_view) != len(arr):
            raise ValueError("frame_attn requires rows_in_view aligned with arr")
        # Softmax over quality logits (B-mode preferred); no learned params.
        logits = []
        for r in rows_in_view:
            non_d, conf, area, comp = _frame_quality_key(r)
            logits.append(3.0 * float(non_d) + 2.0 * float(conf) + float(area) + float(comp))
        logits_a = np.asarray(logits, dtype=np.float64)
        logits_a = logits_a - np.max(logits_a)
        w = np.exp(logits_a)
        w = w / max(float(w.sum()), 1e-12)
        return np.nan_to_num((arr * w[:, None]).sum(axis=0), nan=0.0)
    if pool == "score_argmax":
        if scores is None or len(scores) != len(arr):
            raise ValueError("score_argmax requires per-frame scores aligned with arr")
        return arr[int(np.argmax(scores))]
    if pool == "mean":
        with np.errstate(all="ignore"):
            return np.nanmean(arr, axis=0)
    if pool == "split_mean_max":
        if n_m0 <= 0 or n_m0 >= arr.shape[1]:
            # no CLIP part → fall back to mean
            with np.errstate(all="ignore"):
                return np.nanmean(arr, axis=0)
        with np.errstate(all="ignore"):
            m0 = np.nanmean(arr[:, :n_m0], axis=0)
            clip = np.nanmax(arr[:, n_m0:], axis=0)
        return np.concatenate([m0, clip], axis=0)
    # nanmax (default legacy)
    with np.errstate(all="ignore"):
        return np.nanmax(arr, axis=0)


def load_manifest_test_patient_meta(manifest: Path) -> dict[str, tuple[int, str]]:
    """patient_id -> (label_binary, disease_name)."""
    out: dict[str, tuple[int, str]] = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s.get("split") != "test":
                continue
            pid = str(s["patient_id"])
            lab = int(s.get("label_binary", 0))
            dis = str(
                s.get("disease_folder")
                or s.get("label_disease_cn")
                or ("正常" if lab == 0 else "未知")
            )
            out[pid] = (lab, dis)
    return out


def augment_manifest_test_patients(
    tok,
    pres,
    y,
    pids,
    *,
    manifest: Path,
    d_in: int,
    n_views: int = 0,
    max_frames: int = 0,
) -> tuple:
    """Append zero tokens for manifest test patients missing from eval set."""
    meta = load_manifest_test_patient_meta(manifest)
    if not meta:
        return tok, pres, y, pids
    existing = {str(p) for p in pids}
    add = [pid for pid in sorted(meta) if pid not in existing]
    if not add:
        return tok, pres, y, pids
    labels_add = [meta[pid][0] for pid in add]
    if tok is None or len(pids) == 0:
        if max_frames > 0:
            tok_new = np.zeros((len(add), max_frames, d_in), dtype=np.float32)
            pres_new = np.zeros((len(add), max_frames), dtype=np.bool_)
        else:
            nv = n_views or len(VIEWS)
            tok_new = np.zeros((len(add), nv, d_in), dtype=np.float32)
            pres_new = np.zeros((len(add), nv), dtype=np.bool_)
        y_new = np.asarray(labels_add, dtype=np.int64)
        print(
            f"  unified test augment: +{len(add)} manifest patients "
            f"(total test={len(add)}; pos={sum(labels_add)})",
            flush=True,
        )
        return tok_new, pres_new, y_new, add
    if max_frames > 0:
        z_tok = np.zeros((len(add), max_frames, d_in), dtype=np.float32)
        z_pres = np.zeros((len(add), max_frames), dtype=np.bool_)
    else:
        nv = tok.shape[1]
        z_tok = np.zeros((len(add), nv, d_in), dtype=np.float32)
        z_pres = np.zeros((len(add), nv), dtype=np.bool_)
    y_new = np.concatenate([np.asarray(y, dtype=np.int64), np.asarray(labels_add, dtype=np.int64)])
    tok_out = np.concatenate([np.asarray(tok), z_tok], axis=0)
    pres_out = np.concatenate([np.asarray(pres), z_pres], axis=0)
    pids_out = list(pids) + add
    print(
        f"  unified test augment: +{len(add)} manifest patients "
        f"({len(pids)} → {len(pids_out)}; +pos={sum(labels_add)})",
        flush=True,
    )
    return tok_out, pres_out, y_new, pids_out


def patient_view_feat_tokens(
    rows,
    X: np.ndarray,
    *,
    pool: str = "nanmax",
    n_m0: int = 0,
    scores: np.ndarray | None = None,
    views: tuple[str, ...] = VIEWS,
):
    """[V,D] mid-level tokens by within-view pooling."""
    if pool not in WITHIN_VIEW_POOLS:
        raise ValueError(f"unknown within-view pool {pool!r}")
    by_pid: dict[str, list] = defaultdict(list)
    score_by_pid: dict[str, list] = defaultdict(list)
    for i, (r, x) in enumerate(zip(rows, X)):
        pid = str(r.patient_id)
        by_pid[pid].append((r, x))
        if scores is not None:
            score_by_pid[pid].append(float(scores[i]))

    tokens_list, present_list, labels, pids = [], [], [], []
    for pid, items in by_pid.items():
        label = int(items[0][0].label)
        tok = np.zeros((len(views), X.shape[1]), dtype=np.float32)
        present = np.zeros(len(views), dtype=np.bool_)
        pid_scores = score_by_pid.get(pid)
        for vi, view in enumerate(views):
            idxs = [j for j, (r, _) in enumerate(items) if r.cardium_view == view]
            if not idxs:
                continue
            arr = np.stack([items[j][1] for j in idxs], axis=0)
            rows_in_view = [items[j][0] for j in idxs]
            sc = None
            if pid_scores is not None:
                sc = np.asarray([pid_scores[j] for j in idxs], dtype=np.float64)
            pooled = _pool_view_matrix(
                arr, pool, n_m0, sc, rows_in_view=rows_in_view,
            )
            if np.all(np.isnan(pooled)):
                continue
            tok[vi] = np.nan_to_num(pooled, nan=0.0)
            present[vi] = True
        if not present.any():
            continue
        tokens_list.append(tok)
        present_list.append(present)
        labels.append(label)
        pids.append(pid)
    return (
        np.stack(tokens_list),
        np.stack(present_list),
        np.asarray(labels, dtype=np.int64),
        pids,
    )


def _row_domain_id(row) -> int:
    meta = getattr(row, "meta", None) or {}
    dom = str(meta.get("domain") or DOMAIN_PRIVATE).lower()
    return 1 if dom == DOMAIN_CARDIUM else 0


def patient_domain_ids(rows, pids: list[str]) -> np.ndarray:
    pid_dom: dict[str, int] = {}
    for r in rows:
        pid_dom[str(r.patient_id)] = _row_domain_id(r)
    return np.asarray([pid_dom.get(str(p), 0) for p in pids], dtype=np.int64)


def patient_view_scores(rows, probs: np.ndarray, views: tuple[str, ...] = VIEWS):
    """Per-patient [V] view scores = max frame score within view."""
    by_pid: dict[str, list] = defaultdict(list)
    for r, p in zip(rows, probs):
        by_pid[str(r.patient_id)].append((r, float(p)))

    scores_list, present_list, labels, pids = [], [], [], []
    for pid, items in by_pid.items():
        label = int(items[0][0].label)
        sc = np.zeros(len(views), dtype=np.float32)
        present = np.zeros(len(views), dtype=np.bool_)
        for vi, view in enumerate(views):
            ps = [p for r, p in items if r.cardium_view == view]
            if not ps:
                continue
            sc[vi] = max(ps)
            present[vi] = True
        if not present.any():
            continue
        scores_list.append(sc)
        present_list.append(present)
        labels.append(label)
        pids.append(pid)
    return (
        np.stack(scores_list),
        np.stack(present_list),
        np.asarray(labels, dtype=np.int64),
        pids,
    )


def patient_frame_bags(
    rows,
    X: np.ndarray,
    *,
    max_frames: int = 32,
    scores: np.ndarray | None = None,
):
    """Attention-MIL bags: [N, max_frames, D] + present mask (score-ranked truncate)."""
    by_pid: dict[str, list] = defaultdict(list)
    for i, (r, x) in enumerate(zip(rows, X)):
        sc = float(scores[i]) if scores is not None else 0.0
        by_pid[str(r.patient_id)].append((sc, x, int(r.label)))

    bags, present, labels, pids = [], [], [], []
    d = X.shape[1]
    for pid, items in by_pid.items():
        items = sorted(items, key=lambda t: t[0], reverse=True)[:max_frames]
        bag = np.zeros((max_frames, d), dtype=np.float32)
        mask = np.zeros(max_frames, dtype=np.bool_)
        for i, (_, x, _) in enumerate(items):
            bag[i] = np.nan_to_num(x, nan=0.0)
            mask[i] = True
        if not mask.any():
            continue
        bags.append(bag)
        present.append(mask)
        labels.append(items[0][2])
        pids.append(pid)
    return (
        np.stack(bags),
        np.stack(present),
        np.asarray(labels, dtype=np.int64),
        pids,
    )


def train_fusion(
    model: nn.Module,
    tr_tok, tr_pres, tr_y,
    va_tok, va_pres, va_y,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    seed: int,
    tr_domain: np.ndarray | None = None,
    va_domain: np.ndarray | None = None,
    domain_classifier: nn.Module | None = None,
    lambda_d_max: float = 0.0,
    lambda_mvp: float = 0.0,
    use_mvp: bool = False,
) -> nn.Module:
    torch.manual_seed(seed)
    tr_dom_t = None
    if tr_domain is not None:
        tr_dom_t = torch.from_numpy(np.asarray(tr_domain, dtype=np.int64))
    ds = PatientTokenDS(
        torch.from_numpy(tr_tok.astype(np.float32)),
        torch.from_numpy(tr_pres),
        torch.from_numpy(tr_y.astype(np.float32)),
        domains=tr_dom_t,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if domain_classifier is not None:
        opt.add_param_group({"params": domain_classifier.parameters(), "lr": lr})
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    va_t = torch.from_numpy(va_tok.astype(np.float32)).to(device)
    va_p = torch.from_numpy(va_pres).to(device)
    grl = GradientReversal(alpha=1.0) if domain_classifier is not None else None
    best_state, best_f1, stale = None, -1.0, 0
    n_steps = max(1, epochs * max(1, len(loader)))

    for ep in range(epochs):
        model.train()
        if domain_classifier is not None:
            domain_classifier.train()
        step_base = ep * max(1, len(loader))
        lam = 0.0
        for step_i, batch in enumerate(loader):
            if tr_dom_t is not None:
                tok, present, y, dom = batch
                dom = dom.to(device)
            else:
                tok, present, y = batch
                dom = None
            tok, present, y = tok.to(device), present.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(tok, present)
            loss = crit(logits, y)
            if use_mvp and lambda_mvp > 0 and hasattr(model, "mvp_loss"):
                loss = loss + float(lambda_mvp) * model.mvp_loss(tok, present)
            if domain_classifier is not None and dom is not None and lambda_d_max > 0:
                progress = (step_base + step_i) / float(n_steps)
                lam = dann_lambda(progress, lambda_d_max)
                grl.alpha = lam
                if hasattr(model, "pooled_features"):
                    feat = model.pooled_features(tok, present)
                else:
                    feat = tok.mean(dim=1)
                dom_logits = domain_classifier(grl(feat))
                loss = loss + domain_loss(dom_logits, dom)
            loss.backward()
            opt.step()
        model.eval()
        if domain_classifier is not None:
            domain_classifier.eval()
        with torch.no_grad():
            va_prob = torch.sigmoid(model(va_t, va_p)).cpu().numpy()
            if domain_classifier is not None and tr_dom_t is not None and lambda_d_max > 0:
                tr_t = torch.from_numpy(tr_tok.astype(np.float32)).to(device)
                tr_p = torch.from_numpy(tr_pres).to(device)
                if hasattr(model, "pooled_features"):
                    feat = model.pooled_features(tr_t, tr_p)
                else:
                    feat = tr_t.mean(dim=1)
                dom_logits = domain_classifier(grl(feat))
                dacc = domain_accuracy(dom_logits, tr_dom_t.to(device))
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(f"    domain_acc(train)={dacc:.3f} lambda_d={lam:.3f}")
        auc = float(roc_auc_score(va_y, va_prob)) if len(np.unique(va_y)) > 1 else float("nan")
        thr, m = find_best_binary_threshold(va_y, va_prob)
        f1 = float(m["f1"])
        improved = f1 > best_f1 + 1e-4
        if improved:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if (ep + 1) % 10 == 0 or ep == 0 or improved:
            print(
                f"    ep {ep+1:03d} val_f1={f1:.4f} thr={thr:.2f} "
                f"auc={auc:.4f} best_f1={best_f1:.4f}"
            )
        if patience > 0 and stale >= patience:
            print(f"    early stop @ ep {ep+1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    m = binary_metrics_at_threshold(y, p, thr)
    return {k: v for k, v in m.items() if k != "confusion_matrix"}


def _fmt_ops(m: dict) -> str:
    return (
        f"F1={_fmt_metric(m.get('f1'))} Sens={_fmt_metric(m.get('sensitivity'))} "
        f"Spec={_fmt_metric(m.get('specificity'))} AUC={_fmt_metric(m.get('auc'))}"
    )


def _print_threshold_modes(label: str, modes: dict[str, dict]) -> None:
    for name in ("val_f1_tuned", "fixed_0.5", "youden", "sens_at_spec_0.90"):
        if name not in modes:
            continue
        m = modes[name]
        extra = ""
        if name == "sens_at_spec_0.90":
            extra = f" (target Spec≥0.90, val_Spec={_fmt_metric(m.get('val_specificity'))})"
        print(f"  {label} [{name}] {_fmt_ops(m)} thr={m['threshold']:.2f}{extra}")


def _mean_mode_metric(folds: list[dict], who: str, mode: str, key: str) -> float:
    vals = []
    for r in folds:
        block = r.get(who) or {}
        m = block.get(mode)
        if not isinstance(m, dict) or key not in m or m[key] is None:
            continue
        try:
            vals.append(float(m[key]))
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else float("nan")


def _modes_present_in_folds(folds: list[dict], who: str = "fusion") -> list[str]:
    """Modes that appear in ≥1 fold (union), in canonical order."""
    preferred = ("val_f1_tuned", "fixed_0.5", "youden", "sens_at_spec_0.90")
    if not folds:
        return list(preferred)
    found: set[str] = set()
    for r in folds:
        found |= set((r.get(who) or {}).keys())
    return [m for m in preferred if m in found]


def _threshold_at_min_specificity_local(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    target_spec: float = 0.90,
) -> float:
    """Val thr with Spec≥target maximizing Sens (no dependency on metrics.py version)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if int((y_true == 0).sum()) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_t = 1.0
    best_sens = -1.0
    for t in np.linspace(0.99, 0.01, 99):
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


def _ensure_sens_at_spec_mode(
    modes: dict[str, dict],
    y_val: np.ndarray,
    prob_val: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
    *,
    target_spec: float = 0.90,
    label: str = "",
) -> dict[str, dict]:
    """Always (re)compute Sens@Spec≥0.90 so SUMMARY never depends on metrics.py sync."""
    from chd_baseline.metrics import binary_metrics_at_threshold

    thr = _threshold_at_min_specificity_local(y_val, prob_val, target_spec=target_spec)
    m = binary_metrics_at_threshold(y_test, prob_test, float(thr))
    m_val = binary_metrics_at_threshold(y_val, prob_val, float(thr))
    out = dict(modes)
    out["sens_at_spec_0.90"] = {
        **{k: v for k, v in m.items() if k != "confusion_matrix"},
        "threshold": float(thr),
        "target_specificity": float(target_spec),
        "val_specificity": m_val.get("specificity"),
    }
    tag = f"{label} " if label else ""
    print(
        f"  {tag}[sens_at_spec_0.90] "
        f"Sens={out['sens_at_spec_0.90'].get('sensitivity')} "
        f"Spec={out['sens_at_spec_0.90'].get('specificity')} "
        f"thr={thr:.3f} val_Spec={m_val.get('specificity')}"
    )
    return out


LR_GATE_TAUS = (0.50, 0.70, 0.85, 0.90, 0.95, 0.99, 1.01)  # 1.01 → never fire (= pure fusion)
LR_GATE_FORCED_TAU = 0.95  # always report (even when selected τ disables gate)
LR_GATE_F1_SLACK = 0.01  # primary τ may drop ≤ this vs τ=1.01 F1


def apply_lr_max_gate(
    p_fus: np.ndarray,
    p_lr: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Local-evidence gate: if LR_max ≥ τ, use max(fusion, LR); else fusion.

    Soft scores so existing threshold modes / Sens@90%Spec still apply.
    τ≥1.01 disables the gate (identical to pure fusion).
    """
    p_fus = np.asarray(p_fus, dtype=np.float64)
    p_lr = np.asarray(p_lr, dtype=np.float64)
    if float(tau) >= 1.0 + 1e-9:
        return p_fus.copy()
    return np.where(p_lr >= float(tau), np.maximum(p_fus, p_lr), p_fus)


def _align_scores_by_pid(
    pids_src: list[str],
    scores_src: np.ndarray,
    pids_dst: list[str],
    *,
    fill_missing: float | None = None,
) -> np.ndarray:
    """Align LR scores from pids_src order to pids_dst order.

    When --include-empty-view-patients is on, te_pids can contain manifest
    patients that had no frames in the embed cache (zero-fill bags), so they
    never appear in te_pids_max.  Pass fill_missing=0.0 to tolerate this.
    If fill_missing is None (default) the strict check is kept.
    """
    m = {str(p): float(s) for p, s in zip(pids_src, scores_src)}
    missing = [str(p) for p in pids_dst if str(p) not in m]
    if missing:
        if fill_missing is None:
            raise RuntimeError(
                f"LR gate align failed: {len(missing)} pids missing in LR map "
                f"(e.g. {missing[:3]})"
            )
        if len(missing) <= 5 or len(missing) < 0.1 * len(pids_dst):
            print(
                f"  LR align: {len(missing)} pids missing (zero-fill bags from "
                f"--include-empty-view-patients), filling with {fill_missing:.3f}"
            )
        else:
            print(
                f"  WARN: LR align large gap {len(missing)}/{len(pids_dst)} pids missing, "
                f"filling with {fill_missing:.3f} — check embed coverage"
            )
        for p in missing:
            m[p] = fill_missing
    return np.asarray([m[str(p)] for p in pids_dst], dtype=np.float64)


def select_lr_gate_tau(
    y_val: np.ndarray,
    p_fus_val: np.ndarray,
    p_lr_val: np.ndarray,
    *,
    taus: tuple[float, ...] = LR_GATE_TAUS,
    f1_slack: float = LR_GATE_F1_SLACK,
) -> dict:
    """Grid-search τ on val.

    - tau_f1: max val F1 (ties → higher τ; often 1.01 = off)
    - tau_sens90: max Sens@90%Spec (ties → higher τ)
    - tau_primary: max Sens@90% among τ with F1 ≥ F1(τ=1.01)−slack
      (clinical OP first; F1 floor avoids wrecking the main metric)
    """
    y_val = np.asarray(y_val, dtype=np.int64)
    rows = []
    best_f1, best_t_f1 = -1.0, 1.01
    best_sens90, best_t_s90 = -1.0, 1.01
    for t in taus:
        p = apply_lr_max_gate(p_fus_val, p_lr_val, t)
        thr_f1, m_f1 = find_best_binary_threshold(y_val, p)
        f1 = float(m_f1.get("f1") or 0.0)
        thr90 = _threshold_at_min_specificity_local(y_val, p, target_spec=0.90)
        m90 = binary_metrics_at_threshold(y_val, p, float(thr90))
        sens90 = float(m90.get("sensitivity") or 0.0)
        rows.append({
            "tau": float(t),
            "val_f1": f1,
            "val_f1_thr": float(thr_f1),
            "val_sens_at_spec_0.90": sens90,
            "val_spec90_thr": float(thr90),
            "val_spec_at_sens_op": m90.get("specificity"),
            "n_val_gate_fire": int((np.asarray(p_lr_val) >= float(t)).sum()) if t < 1.0 else 0,
        })
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and t > best_t_f1):
            best_f1, best_t_f1 = f1, float(t)
        if sens90 > best_sens90 + 1e-12 or (
            abs(sens90 - best_sens90) <= 1e-12 and t > best_t_s90
        ):
            best_sens90, best_t_s90 = sens90, float(t)

    base = next((r for r in rows if r["tau"] >= 1.0), None)
    base_f1 = float(base["val_f1"]) if base is not None else best_f1
    eligible = [r for r in rows if r["val_f1"] + 1e-12 >= base_f1 - float(f1_slack)]
    if not eligible:
        eligible = list(rows)
    # Primary: max Sens@90 under F1 floor; ties → higher τ (safer / closer to off)
    best_pri = max(
        eligible,
        key=lambda r: (float(r["val_sens_at_spec_0.90"]), float(r["tau"])),
    )
    tau_primary = float(best_pri["tau"])
    return {
        "tau_primary": tau_primary,
        "tau_f1": best_t_f1,
        "tau_sens90": best_t_s90,
        "val_f1_at_tau_primary": float(best_pri["val_f1"]),
        "val_sens90_at_tau_primary": float(best_pri["val_sens_at_spec_0.90"]),
        "val_f1_at_tau_f1": best_f1,
        "val_sens90_at_tau_sens90": best_sens90,
        "f1_slack": float(f1_slack),
        "baseline_f1_tau_off": base_f1,
        "grid": rows,
        # aliases for older SUMMARY keys
        "alpha_f1": tau_primary,
        "alpha_sens90": best_t_s90,
    }


@torch.no_grad()
def predict_fusion(model, tokens, present, device) -> np.ndarray:
    model.eval()
    logit = model(
        torch.from_numpy(tokens.astype(np.float32)).to(device),
        torch.from_numpy(present).to(device),
    ).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logit))


def evaluate_view_decomposition(
    model,
    va_tok,
    va_pres,
    va_y,
    te_tok,
    te_pres,
    te_y,
    views: tuple[str, ...],
    device,
) -> dict:
    """Leave-one / single-view / keep-k combos (no retrain; mask present views)."""
    from itertools import combinations

    n = len(views)
    va_pres_b = np.asarray(va_pres, dtype=bool)
    te_pres_b = np.asarray(te_pres, dtype=bool)
    out: dict = {
        "leave_one_out": {},
        "single_view": {},
        "keep_k": {},
        "progressive": {},
    }

    def _eval_mask(va_p: np.ndarray, te_p: np.ndarray, label: str) -> dict:
        va_s = predict_fusion(model, va_tok, va_p, device)
        te_s = predict_fusion(model, te_tok, te_p, device)
        modes = patient_metrics_at_threshold_modes(va_y, va_s, te_y, te_s)
        return _ensure_sens_at_spec_mode(modes, va_y, va_s, te_y, te_s, label=label)

    def _metric_triplet(modes: dict) -> dict:
        vt = modes.get("val_f1_tuned") or {}
        s90 = modes.get("sens_at_spec_0.90") or {}
        return {
            "f1": vt.get("f1"),
            "auc": vt.get("auc"),
            "sens_at_spec_0.90": s90.get("sensitivity"),
            "ppv": vt.get("ppv"),
        }

    # Full bag (all originally present views kept as-is)
    full_modes = _eval_mask(va_pres_b, te_pres_b, label="keep_all")
    out["keep_k"][str(n)] = {"all": full_modes}
    _print_threshold_modes("keep_all", full_modes)

    for i, vname in enumerate(views):
        va_p = va_pres_b.copy()
        te_p = te_pres_b.copy()
        va_p[:, i] = False
        te_p[:, i] = False
        modes = _eval_mask(va_p, te_p, label=f"leave_out_{vname}")
        out["leave_one_out"][vname] = modes
        _print_threshold_modes(f"leave_out_{vname}", modes)

    for i, vname in enumerate(views):
        va_p = np.zeros_like(va_pres_b, dtype=bool)
        te_p = np.zeros_like(te_pres_b, dtype=bool)
        va_p[:, i] = va_pres_b[:, i]
        te_p[:, i] = te_pres_b[:, i]
        modes = _eval_mask(va_p, te_p, label=f"only_{vname}")
        out["single_view"][vname] = modes
        _print_threshold_modes(f"only_{vname}", modes)

    # keep-k for k=2..n-1 (k=n and k=1 covered above)
    for k in range(2, n):
        combo_modes: dict[str, dict] = {}
        for idxs in combinations(range(n), k):
            name = "+".join(views[i] for i in idxs)
            va_p = np.zeros_like(va_pres_b, dtype=bool)
            te_p = np.zeros_like(te_pres_b, dtype=bool)
            for i in idxs:
                va_p[:, i] = va_pres_b[:, i]
                te_p[:, i] = te_pres_b[:, i]
            modes = _eval_mask(va_p, te_p, label=f"keep{k}_{name}")
            combo_modes[name] = modes
            _print_threshold_modes(f"keep{k}_{name}", modes)
        out["keep_k"][str(k)] = combo_modes

    out["keep_k"]["1"] = {v: out["single_view"][v] for v in views}

    # Progressive summary: mean F1/AUC over combos at each k
    for k_str, combos in out["keep_k"].items():
        f1s, aucs, sens = [], [], []
        for modes in combos.values():
            t = _metric_triplet(modes)
            if t["f1"] is not None:
                f1s.append(float(t["f1"]))
            if t["auc"] is not None:
                aucs.append(float(t["auc"]))
            if t["sens_at_spec_0.90"] is not None:
                sens.append(float(t["sens_at_spec_0.90"]))
        out["progressive"][k_str] = {
            "n_combos": len(combos),
            "f1_mean": float(np.mean(f1s)) if f1s else None,
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "sens90_mean": float(np.mean(sens)) if sens else None,
        }
    return out


def patient_agg_from_frames(rows, probs, how: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    by: dict[str, list] = defaultdict(list)
    for r, p in zip(rows, probs):
        by[str(r.patient_id)].append((int(r.label), float(p)))
    y, s, pids = [], [], []
    for pid in sorted(by):
        items = by[pid]
        pids.append(pid)
        y.append(items[0][0])
        vals = [p for _, p in items]
        s.append(max(vals) if how == "max" else float(np.mean(vals)))
    return np.asarray(y), np.asarray(s), pids


_NEW_DEVICE_NORM_COHORTS = frozenset({
    "norm_4ac", "zy", "new_norm_zy", "zy_norm",
})
_NEW_DEVICE_NORM_PREFIXES = ("norm_4ac", "zy_norm", "zy")
_SAME_SOURCE_NORM_COHORTS = frozenset({
    "tertiary_norm",  # 三级筛查阴性 — homologous_real in-domain normals
    "midlate",
})


def _infer_source_cohort(pid: str, meta: dict) -> str:
    cohort = str(meta.get("source_cohort") or "")
    if not cohort and "|" in pid:
        cohort = pid.split("|", 1)[0]
    if meta.get("homologous_midlate") or pid.startswith("midlate|"):
        return "midlate"
    if pid.startswith("tertiary_norm|"):
        return "tertiary_norm"
    return cohort


def _is_new_device_norm(label: int, cohort: str, pid: str) -> bool:
    """Held-out site/cohort normals (zy / norm_4ac), not same-source tertiary/midlate."""
    if label != 0:
        return False
    if cohort in _SAME_SOURCE_NORM_COHORTS or pid.startswith("tertiary_norm|") or pid.startswith("midlate|"):
        return False
    if cohort in _NEW_DEVICE_NORM_COHORTS:
        return True
    prefix = pid.split("|", 1)[0] if "|" in pid else pid
    if prefix in _NEW_DEVICE_NORM_PREFIXES:
        return True
    # default: treat unknown normals as held-out (legacy homologous→zy eval)
    return cohort not in ("abnorm_new", "")


def _is_homologous_abnorm(label: int, cohort: str, pid: str) -> bool:
    if label != 1:
        return False
    if cohort == "midlate" or pid.startswith("midlate|"):
        return False
    return True


def _build_patient_info(rows) -> dict[str, dict]:
    """One entry per patient_id from frame rows (test/val)."""
    info: dict[str, dict] = {}
    for r in rows:
        pid = str(r.patient_id)
        if pid in info:
            continue
        meta = getattr(r, "meta", None) or {}
        cohort = _infer_source_cohort(pid, meta)
        label = int(r.label)
        info[pid] = {
            "label": label,
            "label_name": str(getattr(r, "label_name", "") or ""),
            "source_cohort": cohort,
            "is_new_device_norm": _is_new_device_norm(label, cohort, pid),
            "is_homologous_abnorm": _is_homologous_abnorm(label, cohort, pid),
            "split": str(getattr(r, "split", "") or ""),
        }
    return info


def _cohort_counts(patient_info: dict[str, dict]) -> dict[str, int]:
    from collections import Counter
    c: Counter[str] = Counter()
    for v in patient_info.values():
        key = f"{'abn' if v['label'] else 'norm'}:{v.get('source_cohort') or 'unknown'}"
        c[key] += 1
    return dict(sorted(c.items()))


def _fmt_metric(v, fmt: str = ".3f") -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)
        if np.isnan(fv):
            return "N/A"
        return format(fv, fmt)
    except (TypeError, ValueError):
        return "N/A"


def _print_stratified_test_report(
    label: str,
    stratified: dict[str, dict[str, dict]],
    *,
    primary_mode: str = "val_f1_tuned",
) -> None:
    print(f"\n========== STRATIFIED TEST ({label}) ==========")
    strata_order = (
        ("all", "全部 test"),
        ("screening_ood", "新机正常 + 同源异常 (OOD screening)"),
        ("abnorm_homologous", "仅同源异常 → Sens"),
        ("norm_new_device", "仅新机正常 → Spec/FPR"),
        ("norm_all", "全部正常"),
    )
    for mode in ("val_f1_tuned", "fixed_0.5", "youden", "sens_at_spec_0.90"):
        if mode not in stratified.get("all", {}):
            continue
        thr = stratified.get("all", {}).get(mode, {}).get("threshold", float("nan"))
        print(f"  --- [{mode}] thr={_fmt_metric(thr, '.2f')} ---")
        for key, title in strata_order:
            m = stratified.get(key, {}).get(mode, {})
            n = m.get("n_patients", m.get("n", 0))
            if not n:
                print(f"    {title:28s} n=0")
                continue
            parts = [f"n={n}"]
            if m.get("auc") is not None:
                parts.append(f"AUC={_fmt_metric(m['auc'])}")
            if m.get("f1") is not None:
                parts.append(f"F1={_fmt_metric(m['f1'])}")
            if m.get("sensitivity") is not None:
                parts.append(f"Sens={_fmt_metric(m['sensitivity'])}")
            if m.get("specificity") is not None:
                parts.append(f"Spec={_fmt_metric(m['specificity'])}")
            if m.get("fpr") is not None:
                parts.append(f"FPR={_fmt_metric(m['fpr'])}")
            print(f"    {title:28s} " + "  ".join(parts))
        if mode == primary_mode:
            ood = stratified.get("screening_ood", {}).get(mode, {})
            nn = stratified.get("norm_new_device", {}).get(mode, {})
            print(
                f"    >> OOD 解读: screening_ood AUC={_fmt_metric(ood.get('auc'))} | "
                f"新机正常 Spec={_fmt_metric(nn.get('specificity'))} "
                f"FPR={_fmt_metric(nn.get('fpr'))}"
            )


def _is_graph_fusion(fusion: str) -> bool:
    return fusion in ("graph_transformer", "anatomy_graph")


def make_fusion_model(args: argparse.Namespace, d_in: int | None = None) -> nn.Module:
    d_in = feat_dim(args) if d_in is None else d_in
    nv = n_views_for(args)
    n_m0 = n_m0_dims(args)
    if _is_graph_fusion(args.fusion):
        adj_arg = str(getattr(args, "graph_adj", "auto") or "auto")
        if adj_arg == "auto":
            adj_arg = "anatomy" if args.fusion == "anatomy_graph" else "full"
        spec = resolve_graph_adjacency(
            adj_arg,
            seed=int(getattr(args, "seed", 42) or 42),
        )
        args._resolved_graph_adj = spec["mode"]
        args._resolved_graph_perm = spec.get("perm")
        # graph_transformer historical default: no adjacency tensor (all present).
        adj = spec["adjacency"]
        if args.fusion == "graph_transformer" and spec["mode"] == "full":
            adj = None
        use_adj = adj is not None
        return ViewGraphFusion(
            d_in=d_in,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            n_views=nv,
            n_m0=n_m0,
            use_view_embed=not bool(getattr(args, "no_view_embed", False)),
            pool_mask=str(getattr(args, "pool_mask", "present") or "present"),
            graph_layers=int(getattr(args, "graph_layers", 1) or 1),
            use_anatomy_adj=use_adj,
            adjacency=adj,
            impute_mode=spec["impute_mode"],
            encoder_anatomy_mask=(
                use_adj and not bool(getattr(args, "no_anatomy_encoder_mask", False))
            ),
        )
    if args.fusion == "feature_transformer":
        return ViewTokenFusion(
            d_in=d_in, d_model=args.d_model, n_layers=args.n_layers,
            dropout=args.dropout, n_views=nv,
            use_view_embed=not bool(getattr(args, "no_view_embed", False)),
            missing_fill=str(getattr(args, "missing_fill", "learned") or "learned"),
            pool_mask=str(getattr(args, "pool_mask", "present") or "present"),
        )
    if args.fusion == "cross_plane_cls":
        return ViewTokenFusionCLS(
            d_in=d_in, d_model=args.d_model, n_layers=args.n_layers,
            dropout=args.dropout, n_views=nv,
        )
    if args.fusion == "view_bilstm":
        return ViewBiLSTMFusion(
            d_in=d_in, d_model=args.d_model, dropout=args.dropout, n_views=nv,
        )
    if args.fusion == "attention_mil":
        return AttentionMIL(d_in=d_in, d_model=args.d_model, dropout=args.dropout)
    if args.fusion in ("view_mean", "view_max"):
        return ViewStatPoolFusion(
            d_in=d_in,
            d_model=args.d_model,
            dropout=args.dropout,
            mode="max" if args.fusion == "view_max" else "mean",
        )
    if args.fusion == "score_mlp":
        return ScoreMLPFusion(n_views=nv)
    return ScoreAttnFusion(n_views=nv)


def _is_feature_fusion(fusion: str) -> bool:
    return fusion in (
        "feature_transformer", "graph_transformer", "anatomy_graph",
            "cross_plane_cls", "view_bilstm", "view_mean", "view_max",
        )


def _is_mil_fusion(fusion: str) -> bool:
    return fusion == "attention_mil"


def _rows_X(rows, args: argparse.Namespace, embed_cache: dict | None) -> np.ndarray:
    import inspect

    app_dim = int(getattr(args, "appearance_dim", 0) or 0) or None
    kwargs = dict(
        model=args.feat_model,
        embed_cache=embed_cache,
        m0_features=args.m0_features,
    )
    # Server may have newer fusion.py but stale masvf_m0_screening.py
    if "appearance_dim" in inspect.signature(rows_to_X).parameters:
        kwargs["appearance_dim"] = app_dim
    elif app_dim is not None and app_dim != FETALCLIP_EMBED_DIM:
        raise RuntimeError(
            f"rows_to_X() missing appearance_dim= (need dim {app_dim}). "
            "Sync experiments/masvf_m0_screening.py to the server."
        )
    X = rows_to_X(rows, **kwargs)
    if getattr(args, "extra_feats", "none") != "view_anat":
        return X
    tags = getattr(args, "_tags", None) or {}
    extras = []
    for r in rows:
        tag = tags.get(r.sample_id) or tags.get(str(r.sample_id)) or {}
        # prefer plane_area from row.meta when tag is sparse
        if r.meta and "plane_area_norm" in r.meta and "plane_area_norm" not in tag:
            tag = {**tag, "plane_area_norm": r.meta["plane_area_norm"]}
        extras.append(view_anat_frame_from_tag(tag, r.cardium_view))
    return np.hstack([X, np.stack(extras).astype(np.float64)])


def subsample_rows_by_patient(rows, max_patients: int, seed: int):
    """Cap patients for smoke runs.

    Stratify by (train vs eval) and label so homologous midlate normals are not
    wiped out by random draw over a much larger abn+eval pool (which caused
    fit_lr 'only one class' crashes).
    """
    if max_patients <= 0:
        return rows
    by_pid: dict[str, list] = {}
    meta: dict[str, tuple[str, int]] = {}
    for r in rows:
        pid = str(r.patient_id)
        by_pid.setdefault(pid, []).append(r)
        if pid not in meta:
            meta[pid] = (str(getattr(r, "split", "") or ""), int(r.label))

    train_neg = [p for p, (sp, y) in meta.items() if sp == "train" and y == 0]
    train_pos = [p for p, (sp, y) in meta.items() if sp == "train" and y == 1]
    eval_pids = [p for p, (sp, _) in meta.items() if sp != "train"]
    rng = np.random.default_rng(seed)

    def _take(pool: list[str], n: int) -> list[str]:
        if n <= 0 or not pool:
            return []
        n = min(int(n), len(pool))
        return [str(x) for x in rng.choice(pool, size=n, replace=False).tolist()]

    # Half budget for train (must keep both labels when available), half for eval.
    n_train_budget = max(2, max_patients // 2)
    n_eval_budget = max(0, max_patients - n_train_budget)
    n_neg = min(len(train_neg), max(1, n_train_budget // 2)) if train_neg else 0
    n_pos = min(len(train_pos), max(1, n_train_budget - n_neg)) if train_pos else 0
    if train_neg and train_pos and n_neg + n_pos > n_train_budget:
        n_pos = max(1, n_train_budget - n_neg)
    keep = set(_take(train_neg, n_neg) + _take(train_pos, n_pos) + _take(eval_pids, n_eval_budget))
    leftover = [p for p in meta if p not in keep]
    need = max_patients - len(keep)
    if need > 0 and leftover:
        keep.update(_take(leftover, need))

    out = [r for r in rows if str(r.patient_id) in keep]
    n_tr0 = sum(1 for p in keep if meta[p][0] == "train" and meta[p][1] == 0)
    n_tr1 = sum(1 for p in keep if meta[p][0] == "train" and meta[p][1] == 1)
    print(
        f"  smoke subsample patients {len(meta)} → {len(keep)} "
        f"(train norm={n_tr0} abn={n_tr1}; frames {len(rows)} → {len(out)})"
    )
    if n_tr0 == 0 or n_tr1 == 0:
        print(
            "  WARN: smoke subsample left a single train class — "
            "raise MAX_PATIENTS or set MAX_PATIENTS=0"
        )
    return out


def _assert_binary_labels(y: np.ndarray, *, where: str) -> None:
    labs = sorted({int(v) for v in np.asarray(y).ravel().tolist()})
    if len(labs) < 2:
        raise SystemExit(
            f"ERROR: {where} has a single class {labs} — cannot fit frame LR. "
            f"Common cause: MAX_PATIENTS smoke subsample dropped all midlate normals. "
            f"Fix: MAX_PATIENTS=0 or use stratified subsample (already on); "
            f"check train patients norm/abn counts above."
        )


def drop_norm_views_to_match_abn(
    tokens: np.ndarray,
    present: np.ndarray,
    labels: np.ndarray,
    *,
    scale: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Randomly mask normal patients' views using abn per-view miss rates (train only)."""
    tokens_in = np.asarray(tokens)
    present_in = np.asarray(present)
    tokens = tokens_in.copy()
    present = present_in.copy()
    labels = np.asarray(labels)
    abn = labels == 1
    norm = ~abn
    meta = {"applied": False, "p_miss": [], "mean_views_before": 0.0, "mean_views_after": 0.0, "n_slots_dropped": 0}
    if not abn.any() or not norm.any():
        return tokens, present, meta
    p_miss = np.clip(
        (1.0 - present_in[abn].mean(axis=0).astype(np.float64)) * float(scale),
        0.0,
        0.95,
    )
    rng = np.random.default_rng(seed)
    before = float(present_in[norm].sum(axis=1).mean())
    n_drop = 0
    for i in np.where(norm)[0]:
        on = np.where(present_in[i])[0]
        if len(on) == 0:
            continue
        keep = []
        for v in on.tolist():
            if len(on) == 1 or rng.random() >= p_miss[v]:
                keep.append(v)
            else:
                n_drop += 1
        if not keep:
            keep = [int(rng.choice(on))]
            n_drop = max(0, n_drop - 1)
        present[i] = False
        tokens[i] = 0.0
        for v in keep:
            present[i, v] = True
            tokens[i, v] = tokens_in[i, v]
    after = float(present[norm].sum(axis=1).mean())
    meta.update({
        "applied": True,
        "p_miss": [float(x) for x in p_miss],
        "mean_views_before": before,
        "mean_views_after": after,
        "n_slots_dropped": int(n_drop),
    })
    return tokens, present, meta


def run_eval_fold(
    *,
    fold: int,
    fit_rows,
    val_rows,
    test_rows,
    embed_cache: dict | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    X_fit = _rows_X(fit_rows, args, embed_cache)
    n_m0 = n_m0_dims(args)
    clip_norm = str(getattr(args, "clip_norm", "hybrid") or "hybrid")
    # m0-only: hybrid degenerates to joint on M0 block
    lr_n_m0 = n_m0 if args.feat_model == "m1" else 0
    y_fit = np.array([r.label for r in fit_rows], dtype=np.int64)
    _assert_binary_labels(y_fit, where="fit_rows for frame LR")
    clf, fill = fit_lr(
        X_fit,
        y_fit,
        n_m0=lr_n_m0,
        clip_norm=clip_norm,
    )
    m0_names = resolve_m0_feature_names(args.m0_features)
    coef_n_m0 = n_m0  # m0-only: all dims; m1: M0 block length (may be 0 for none)
    lr_coef_summary = summarize_frame_lr_coefs(
        clf, m0_names=m0_names, n_m0=coef_n_m0,
    )
    print_frame_lr_coef_summary(lr_coef_summary)
    fit_prob = predict_lr(clf, fill, X_fit)
    val_prob = predict_lr(clf, fill, _rows_X(val_rows, args, embed_cache))
    te_prob = predict_lr(clf, fill, _rows_X(test_rows, args, embed_cache))
    print(f"  clip_norm={clip_norm} n_m0={lr_n_m0}")

    y_max, s_max, te_pids_max = patient_agg_from_frames(test_rows, te_prob, "max")
    y_mean, s_mean, te_pids_mean = patient_agg_from_frames(test_rows, te_prob, "mean")
    va_ymax, va_smax, va_pids_max = patient_agg_from_frames(val_rows, val_prob, "max")
    va_ymean, va_smean, va_pids_mean = patient_agg_from_frames(val_rows, val_prob, "mean")
    c2b_max_modes = patient_metrics_at_threshold_modes(va_ymax, va_smax, y_max, s_max)
    c2b_mean_modes = patient_metrics_at_threshold_modes(va_ymean, va_smean, y_mean, s_mean)
    c2b_max_modes = _ensure_sens_at_spec_mode(
        c2b_max_modes, va_ymax, va_smax, y_max, s_max, label="C2-b max",
    )
    c2b_mean_modes = _ensure_sens_at_spec_mode(
        c2b_mean_modes, va_ymean, va_smean, y_mean, s_mean, label="C2-b mean",
    )
    _print_threshold_modes("C2-b patient-max", c2b_max_modes)
    _print_threshold_modes("C2-b patient-mean", c2b_mean_modes)

    test_patient_info = _build_patient_info(test_rows)
    print(f"  test cohort counts: {_cohort_counts(test_patient_info)}")

    d_in = feat_dim(args)
    views = active_views(args.fallback)
    if _is_feature_fusion(args.fusion):
        X_fit_i, _ = impute_features(X_fit, fill)
        X_val_i, _ = impute_features(_rows_X(val_rows, args, embed_cache), fill)
        X_te_i, _ = impute_features(_rows_X(test_rows, args, embed_cache), fill)
        X_fit_i, X_val_i, X_te_i = standardize_feature_blocks(
            X_fit_i, X_val_i, X_te_i, n_m0=lr_n_m0, clip_norm=clip_norm,
        )
        pool = args.within_view_pool
        use_scores = pool == "score_argmax"
        tr_tok, tr_pres, tr_y, tr_pids = patient_view_feat_tokens(
            fit_rows, X_fit_i, pool=pool, n_m0=n_m0, views=views,
            scores=fit_prob if use_scores else None,
        )
        va_tok, va_pres, va_y, va_pids = patient_view_feat_tokens(
            val_rows, X_val_i, pool=pool, n_m0=n_m0, views=views,
            scores=val_prob if use_scores else None,
        )
        te_tok, te_pres, te_y, te_pids = patient_view_feat_tokens(
            test_rows, X_te_i, pool=pool, n_m0=n_m0, views=views,
            scores=te_prob if use_scores else None,
        )
        if getattr(args, "include_empty_view_patients", False):
            te_tok, te_pres, te_y, te_pids = augment_manifest_test_patients(
                te_tok, te_pres, te_y, te_pids,
                manifest=args.manifest,
                d_in=d_in,
                n_views=len(views),
            )
        tr_dom = patient_domain_ids(fit_rows, tr_pids)
        va_dom = patient_domain_ids(val_rows, va_pids)
        model = make_fusion_model(args, d_in=d_in).to(device)
    elif _is_mil_fusion(args.fusion):
        X_fit_i, _ = impute_features(X_fit, fill)
        X_val_i, _ = impute_features(_rows_X(val_rows, args, embed_cache), fill)
        X_te_i, _ = impute_features(_rows_X(test_rows, args, embed_cache), fill)
        X_fit_i, X_val_i, X_te_i = standardize_feature_blocks(
            X_fit_i, X_val_i, X_te_i, n_m0=lr_n_m0, clip_norm=clip_norm,
        )
        mf = args.mil_max_frames
        tr_tok, tr_pres, tr_y, _ = patient_frame_bags(
            fit_rows, X_fit_i, max_frames=mf, scores=fit_prob,
        )
        va_tok, va_pres, va_y, va_pids = patient_frame_bags(
            val_rows, X_val_i, max_frames=mf, scores=val_prob,
        )
        te_tok, te_pres, te_y, te_pids = patient_frame_bags(
            test_rows, X_te_i, max_frames=mf, scores=te_prob,
        )
        if getattr(args, "include_empty_view_patients", False):
            te_tok, te_pres, te_y, te_pids = augment_manifest_test_patients(
                te_tok, te_pres, te_y, te_pids,
                manifest=args.manifest,
                d_in=d_in,
                max_frames=mf,
            )
        model = make_fusion_model(args, d_in=d_in).to(device)
    else:
        tr_tok, tr_pres, tr_y, _ = patient_view_scores(fit_rows, fit_prob, views=views)
        va_tok, va_pres, va_y, va_pids = patient_view_scores(val_rows, val_prob, views=views)
        te_tok, te_pres, te_y, te_pids = patient_view_scores(test_rows, te_prob, views=views)
        if getattr(args, "include_empty_view_patients", False):
            te_tok, te_pres, te_y, te_pids = augment_manifest_test_patients(
                te_tok, te_pres, te_y, te_pids,
                manifest=args.manifest,
                d_in=d_in,
                n_views=len(views),
            )
        model = make_fusion_model(args, d_in=d_in).to(device)

    if getattr(args, "norm_view_drop", False) and not _is_mil_fusion(args.fusion):
        tr_tok, tr_pres, drop_meta = drop_norm_views_to_match_abn(
            tr_tok, tr_pres, tr_y,
            scale=float(getattr(args, "norm_view_drop_scale", 1.0) or 1.0),
            seed=int(args.seed) + fold,
        )
        if drop_meta.get("applied"):
            print(
                f"  norm view-drop: mean views {drop_meta['mean_views_before']:.2f}"
                f"→{drop_meta['mean_views_after']:.2f} "
                f"slots_dropped={drop_meta['n_slots_dropped']} "
                f"p_miss={['%.2f' % p for p in drop_meta['p_miss']]}"
            )
        else:
            print("  norm view-drop: skipped (need both norm and abn in train)")
    elif getattr(args, "norm_view_drop", False) and _is_mil_fusion(args.fusion):
        print("  norm view-drop: skipped for attention_mil (frame bags, not view slots)")

    print(
        f"  patients fit/val/te={len(tr_y)}/{len(va_y)}/{len(te_y)} "
        f"feat={args.feat_model}/{args.m0_features} pool={args.within_view_pool} "
        f"fallback={args.fallback} fusion={args.fusion} d_in={d_in} n_views={len(views)} "
        f"clip_norm={clip_norm}"
    )
    domain_classifier = None
    lambda_d_max = float(getattr(args, "lambda_d_max", 0.0) or 0.0)
    use_dann = bool(getattr(args, "dann", False)) and lambda_d_max > 0
    use_mvp = (
        not bool(getattr(args, "no_mvp", False))
        and (bool(getattr(args, "mvp", False)) or _is_graph_fusion(args.fusion))
    )
    lambda_mvp = float(getattr(args, "lambda_mvp", 0.0) or 0.0)
    if use_dann and _is_feature_fusion(args.fusion):
        pool_dim = int(getattr(args, "d_model", 64))
        domain_classifier = DomainClassifier(pool_dim, hidden=128).to(device)
    tr_dom = locals().get("tr_dom")
    va_dom = locals().get("va_dom")
    model = train_fusion(
        model, tr_tok, tr_pres, tr_y, va_tok, va_pres, va_y,
        device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        pos_weight=args.pos_weight, seed=args.seed + fold,
        tr_domain=tr_dom, va_domain=va_dom,
        domain_classifier=domain_classifier,
        lambda_d_max=lambda_d_max if use_dann else 0.0,
        lambda_mvp=lambda_mvp,
        use_mvp=use_mvp and _is_graph_fusion(args.fusion),
    )
    va_s = predict_fusion(model, va_tok, va_pres, device)
    te_s = predict_fusion(model, te_tok, te_pres, device)
    fusion_modes = patient_metrics_at_threshold_modes(va_y, va_s, te_y, te_s)
    fusion_modes = _ensure_sens_at_spec_mode(
        fusion_modes, va_y, va_s, te_y, te_s, label=str(args.fusion),
    )
    _print_threshold_modes(args.fusion, fusion_modes)

    view_decomp = None
    if getattr(args, "view_decomp", True) and not _is_mil_fusion(args.fusion):
        print("  --- view decomposition (leave-one / single-view) ---")
        view_decomp = evaluate_view_decomposition(
            model, va_tok, va_pres, va_y, te_tok, te_pres, te_y, views, device,
        )
    elif getattr(args, "view_decomp", True):
        print("  view decomp: skipped for attention_mil")

    fusion_thresholds = {
        mode: float(fusion_modes[mode]["threshold"])
        for mode in fusion_modes
    }
    stratified_fusion = stratified_patient_screening_metrics(
        te_y, te_s, te_pids, test_patient_info, thresholds=fusion_thresholds,
    )
    _print_stratified_test_report(args.fusion, stratified_fusion)

    c2b_thresholds = {
        mode: float(c2b_max_modes[mode]["threshold"])
        for mode in c2b_max_modes
    }
    stratified_c2b_max = stratified_patient_screening_metrics(
        y_max, s_max, te_pids_max, test_patient_info, thresholds=c2b_thresholds,
    )
    _print_stratified_test_report("C2-b patient-max", stratified_c2b_max)

    # --- LR max-gate: if LR_max ≥ τ → score = max(fusion, LR); else fusion ---
    _fill = 0.0 if getattr(args, "include_empty_view_patients", False) else None
    va_lr_aligned = _align_scores_by_pid(va_pids_max, va_smax, va_pids, fill_missing=_fill)
    te_lr_aligned = _align_scores_by_pid(te_pids_max, s_max, te_pids, fill_missing=_fill)
    bypass_sel = select_lr_gate_tau(va_y, va_s, va_lr_aligned)
    tau_primary = float(bypass_sel.get("tau_primary", bypass_sel["tau_f1"]))
    tau_f1 = float(bypass_sel["tau_f1"])
    tau_s90 = float(bypass_sel["tau_sens90"])
    va_s_bypass = apply_lr_max_gate(va_s, va_lr_aligned, tau_primary)
    te_s_bypass = apply_lr_max_gate(te_s, te_lr_aligned, tau_primary)
    bypass_modes = patient_metrics_at_threshold_modes(va_y, va_s_bypass, te_y, te_s_bypass)
    bypass_modes = _ensure_sens_at_spec_mode(
        bypass_modes, va_y, va_s_bypass, te_y, te_s_bypass,
        label=f"fusion_lr_gate(τ={tau_primary:g})",
    )
    _print_threshold_modes(f"fusion_lr_gate τ={tau_primary:g}", bypass_modes)
    print(
        f"  lr_gate select: τ_primary={tau_primary:g} "
        f"(val_f1={bypass_sel.get('val_f1_at_tau_primary', float('nan')):.4f}, "
        f"val_sens90={bypass_sel.get('val_sens90_at_tau_primary', float('nan')):.4f}) | "
        f"τ_f1={tau_f1:g} | τ_sens90={tau_s90:g}"
    )
    bypass_modes_s90 = None
    stratified_bypass_s90 = None
    if abs(tau_s90 - tau_primary) > 1e-12:
        va_s_b2 = apply_lr_max_gate(va_s, va_lr_aligned, tau_s90)
        te_s_b2 = apply_lr_max_gate(te_s, te_lr_aligned, tau_s90)
        bypass_modes_s90 = patient_metrics_at_threshold_modes(va_y, va_s_b2, te_y, te_s_b2)
        bypass_modes_s90 = _ensure_sens_at_spec_mode(
            bypass_modes_s90, va_y, va_s_b2, te_y, te_s_b2,
            label=f"fusion_lr_gate_sens90(τ={tau_s90:g})",
        )
        _print_threshold_modes(f"fusion_lr_gate_sens90 τ={tau_s90:g}", bypass_modes_s90)
        thr_s90 = {
            mode: float(bypass_modes_s90[mode]["threshold"])
            for mode in bypass_modes_s90
        }
        stratified_bypass_s90 = stratified_patient_screening_metrics(
            te_y, te_s_b2, te_pids, test_patient_info, thresholds=thr_s90,
        )

    # Always report forced τ=0.95 so SUMMARY is not identical-to-fusion when primary=off
    bypass_modes_forced = None
    stratified_bypass_forced = None
    tau_forced = float(LR_GATE_FORCED_TAU)
    if abs(tau_forced - tau_primary) > 1e-12:
        va_s_bf = apply_lr_max_gate(va_s, va_lr_aligned, tau_forced)
        te_s_bf = apply_lr_max_gate(te_s, te_lr_aligned, tau_forced)
        bypass_modes_forced = patient_metrics_at_threshold_modes(va_y, va_s_bf, te_y, te_s_bf)
        bypass_modes_forced = _ensure_sens_at_spec_mode(
            bypass_modes_forced, va_y, va_s_bf, te_y, te_s_bf,
            label=f"fusion_lr_gate_forced(τ={tau_forced:g})",
        )
        _print_threshold_modes(f"fusion_lr_gate_forced τ={tau_forced:g}", bypass_modes_forced)
        thr_f = {
            mode: float(bypass_modes_forced[mode]["threshold"])
            for mode in bypass_modes_forced
        }
        stratified_bypass_forced = stratified_patient_screening_metrics(
            te_y, te_s_bf, te_pids, test_patient_info, thresholds=thr_f,
        )

    bypass_thresholds = {
        mode: float(bypass_modes[mode]["threshold"])
        for mode in bypass_modes
    }
    stratified_bypass = stratified_patient_screening_metrics(
        te_y, te_s_bypass, te_pids, test_patient_info, thresholds=bypass_thresholds,
    )
    _print_stratified_test_report(f"fusion_lr_gate τ={tau_primary:g}", stratified_bypass)

    if getattr(args, "export_badcase_dir", None):
        from fusion_badcase_export import collect_patient_cases, export_badcase_bundle

        bc_mode = str(getattr(args, "badcase_threshold_mode", "val_f1_tuned"))
        bc_thr = float(fusion_modes[bc_mode]["threshold"])
        bc_thr_bypass = float(bypass_modes[bc_mode]["threshold"])
        view_score_map: dict[str, list[float]] = {}
        if not _is_feature_fusion(args.fusion) and not _is_mil_fusion(args.fusion):
            for pid, sc, pres in zip(te_pids, te_tok, te_pres):
                view_score_map[str(pid)] = [
                    float(sc[i]) if bool(pres[i]) else float("nan")
                    for i in range(len(views))
                ]
        lr_max_map = {str(p): float(s) for p, s in zip(te_pids, te_lr_aligned)}
        bypass_map = {str(p): float(s) for p, s in zip(te_pids, te_s_bypass)}
        cases = collect_patient_cases(
            test_rows,
            te_prob,
            te_s,
            te_pids,
            test_patient_info,
            threshold=bc_thr,
            threshold_mode=bc_mode,
            project_root=PROJECT_ROOT,
            fallback=str(getattr(args, "fallback", "drop")),
            view_scores=view_score_map or None,
            lr_max_scores=lr_max_map,
            bypass_scores=bypass_map,
            bypass_threshold=bc_thr_bypass,
            bypass_alpha=tau_primary,  # gate τ (kept as bypass_alpha key for HTML compat)
            bypass_tau=tau_primary,
        )
        html_path = export_badcase_bundle(
            cases,
            Path(args.export_badcase_dir),
            meta={
                "train_protocol": getattr(args, "train_protocol", "private"),
                "fusion": args.fusion,
                "feat_model": args.feat_model,
                "m0_features": args.m0_features,
                "m0_feature_names": resolve_m0_feature_names(args.m0_features),
                "within_view_pool": args.within_view_pool,
                "fallback": getattr(args, "fallback", "drop"),
                "clip_norm": getattr(args, "clip_norm", "hybrid"),
                "norm_view_drop": bool(getattr(args, "norm_view_drop", False)),
                "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
                "threshold_mode": bc_mode,
                "threshold": bc_thr,
                "lr_gate_tau": tau_primary,
                "lr_bypass_alpha": tau_primary,  # legacy alias (= gate τ)
                "lr_bypass_threshold": bc_thr_bypass,
                "lr_bypass_select": bypass_sel,
                "fold": fold,
                "c2b_max": {
                    mode: {
                        "threshold": float(c2b_max_modes[mode]["threshold"]),
                        "f1": float(c2b_max_modes[mode]["f1"]),
                        "auc": float(c2b_max_modes[mode]["auc"]),
                    }
                    for mode in c2b_max_modes
                },
                "fusion_modes": {
                    mode: {
                        "threshold": float(fusion_modes[mode]["threshold"]),
                        "f1": float(fusion_modes[mode]["f1"]),
                        "auc": float(fusion_modes[mode]["auc"]),
                        "sensitivity": float(fusion_modes[mode]["sensitivity"]),
                        "specificity": float(fusion_modes[mode]["specificity"]),
                    }
                    for mode in fusion_modes
                },
                "fusion_lr_bypass_modes": {
                    mode: {
                        "threshold": float(bypass_modes[mode]["threshold"]),
                        "f1": float(bypass_modes[mode]["f1"]),
                        "auc": float(bypass_modes[mode]["auc"]),
                        "sensitivity": float(bypass_modes[mode]["sensitivity"]),
                        "specificity": float(bypass_modes[mode]["specificity"]),
                    }
                    for mode in bypass_modes
                },
                "lr_coef_summary": lr_coef_summary,
            },
            project_root=PROJECT_ROOT,
        )
        n_err = sum(1 for c in cases if c["error"] != "OK")
        print(f"  badcase export: {n_err} errors → {html_path}")

    m_c2b_max = c2b_max_modes["val_f1_tuned"]
    m_c2b_mean = c2b_mean_modes["val_f1_tuned"]
    m_fusion = fusion_modes["val_f1_tuned"]
    m_bypass = bypass_modes["val_f1_tuned"]
    if "sens_at_spec_0.90" not in fusion_modes:
        raise RuntimeError(
            "fusion modes missing sens_at_spec_0.90 — sync masvf_view_token_fusion.py "
            "(_ensure_sens_at_spec_mode) to the machine running this job"
        )

    eval_norm_swap = None
    hold_rows = list(getattr(args, "_same_machine_eval_rows", None) or [])
    if bool(getattr(args, "eval_norm_swap", False)) and hold_rows and _is_feature_fusion(args.fusion):
        # Same trained model; swap only test normals (held-out midlate) while keeping test abnormals.
        abn_te = [r for r in test_rows if int(r.label) == 1]
        swap_rows = hold_rows + abn_te
        X_sw_raw = _rows_X(swap_rows, args, embed_cache)
        X_sw_i, _ = impute_features(X_sw_raw, fill)
        # Reuse train→val→te standardization fit; append swap as 4th block.
        X_fit_i2, _ = impute_features(X_fit, fill)
        X_val_i2, _ = impute_features(_rows_X(val_rows, args, embed_cache), fill)
        X_te_i2, _ = impute_features(_rows_X(test_rows, args, embed_cache), fill)
        X_fit_i2, X_val_i2, X_te_i2, X_sw_i = standardize_feature_blocks(
            X_fit_i2, X_val_i2, X_te_i2, X_sw_i, n_m0=lr_n_m0, clip_norm=clip_norm,
        )
        pool = args.within_view_pool
        use_scores = pool == "score_argmax"
        sw_prob = predict_lr(clf, fill, X_sw_raw)
        sw_tok, sw_pres, sw_y, sw_pids = patient_view_feat_tokens(
            swap_rows, X_sw_i, pool=pool, n_m0=n_m0, views=views,
            scores=sw_prob if use_scores else None,
        )
        sw_s = predict_fusion(model, sw_tok, sw_pres, device)
        # Reuse val thresholds from new-device eval for fair comparison of discrimination.
        swap_modes = {}
        yb = np.asarray(sw_y).ravel().astype(np.int64)
        pb = np.asarray(sw_s).ravel().astype(np.float64)
        for mode, md in fusion_modes.items():
            thr = float(md["threshold"])
            m = binary_metrics_at_threshold(yb, pb, thr)
            swap_modes[mode] = {
                "threshold": thr,
                "f1": float(m.get("f1", float("nan"))),
                "auc": float(m.get("auc", float("nan"))),
                "sensitivity": float(m.get("sensitivity", float("nan"))),
                "specificity": float(m.get("specificity", float("nan"))),
                "ppv": float(m.get("ppv", float("nan"))),
            }
        # Also retune on val for swap test (optional secondary)
        swap_retune = patient_metrics_at_threshold_modes(va_y, va_s, sw_y, sw_s)
        swap_retune = _ensure_sens_at_spec_mode(
            swap_retune, va_y, va_s, sw_y, sw_s, label="eval_norm_swap",
        )
        n_sw_norm = len({r.patient_id for r in hold_rows})
        n_sw_abn = len({r.patient_id for r in abn_te})
        eval_norm_swap = {
            "same_machine_midlate": {
                "n_norm_patients": n_sw_norm,
                "n_abn_patients": n_sw_abn,
                "modes_fixed_thr_from_new_device_val": swap_modes,
                "modes_val_retuned": {
                    mode: {
                        "threshold": float(swap_retune[mode]["threshold"]),
                        "f1": float(swap_retune[mode]["f1"]),
                        "auc": float(swap_retune[mode]["auc"]),
                        "sensitivity": float(swap_retune[mode]["sensitivity"]),
                        "specificity": float(swap_retune[mode]["specificity"]),
                    }
                    for mode in swap_retune
                },
            },
            "new_device": {
                "n_norm_patients": int(sum(1 for y in te_y if int(y) == 0)),
                "n_abn_patients": int(sum(1 for y in te_y if int(y) == 1)),
                "modes": {
                    mode: {
                        "threshold": float(fusion_modes[mode]["threshold"]),
                        "f1": float(fusion_modes[mode]["f1"]),
                        "auc": float(fusion_modes[mode]["auc"]),
                        "sensitivity": float(fusion_modes[mode]["sensitivity"]),
                        "specificity": float(fusion_modes[mode]["specificity"]),
                    }
                    for mode in fusion_modes
                },
            },
        }
        sm = swap_retune.get("val_f1_tuned", {})
        nd = fusion_modes.get("val_f1_tuned", {})
        print(
            "  E1 eval-norm-swap (val-retuned): "
            f"same-machine AUC={_fmt_metric(sm.get('auc'))} F1={_fmt_metric(sm.get('f1'))} | "
            f"new-device AUC={_fmt_metric(nd.get('auc'))} F1={_fmt_metric(nd.get('f1'))}"
        )
    elif bool(getattr(args, "eval_norm_swap", False)):
        print("  E1 eval-norm-swap: skipped (need holdout rows + feature fusion)")

    cardium_external = None
    cardium_rows = list(getattr(args, "_cardium_test_rows", None) or [])
    if cardium_rows and (_is_feature_fusion(args.fusion) or _is_mil_fusion(args.fusion)):
        print("  --- CARDIUM external test (official fold holdout) ---")
        X_card_raw = _rows_X(cardium_rows, args, embed_cache)
        X_card_i, _ = impute_features(X_card_raw, fill)
        X_fit_i2, _ = impute_features(X_fit, fill)
        X_val_i2, _ = impute_features(_rows_X(val_rows, args, embed_cache), fill)
        X_te_i2, _ = impute_features(_rows_X(test_rows, args, embed_cache), fill)
        X_fit_i2, X_val_i2, X_te_i2, X_card_i = standardize_feature_blocks(
            X_fit_i2, X_val_i2, X_te_i2, X_card_i, n_m0=lr_n_m0, clip_norm=clip_norm,
        )
        pool = args.within_view_pool
        use_scores = pool == "score_argmax"
        card_prob = predict_lr(clf, fill, X_card_raw)
        if _is_mil_fusion(args.fusion):
            mf = args.mil_max_frames
            card_tok, card_pres, card_y, _ = patient_frame_bags(
                cardium_rows, X_card_i, max_frames=mf, scores=card_prob,
            )
        else:
            card_tok, card_pres, card_y, _ = patient_view_feat_tokens(
                cardium_rows, X_card_i, pool=pool, n_m0=n_m0, views=views,
                scores=card_prob if use_scores else None,
            )
        card_s = predict_fusion(model, card_tok, card_pres, device)
        # (A) thresholds transferred from private val (strict transfer)
        card_modes = patient_metrics_at_threshold_modes(va_y, va_s, card_y, card_s)
        card_modes = _ensure_sens_at_spec_mode(
            card_modes, va_y, va_s, card_y, card_s, label="CARDIUM external",
        )
        _print_threshold_modes("CARDIUM external (private-val thr)", card_modes)
        m_card = card_modes.get("val_f1_tuned", {})
        # (B) label-light: tune on stratified 25% of CARDIUM test, eval on 75%
        card_ll = None
        card_oracle = None
        try:
            y_c = np.asarray(card_y, dtype=np.int64).ravel()
            p_c = np.asarray(card_s, dtype=np.float64).ravel()
            rng = np.random.RandomState(int(args.seed) + 17)
            cal_idx, te_idx = [], []
            for lab in (0, 1):
                idx = np.where(y_c == lab)[0]
                rng.shuffle(idx)
                n_cal = max(1, int(round(0.25 * len(idx)))) if len(idx) else 0
                if len(idx) > 1:
                    n_cal = min(n_cal, len(idx) - 1)
                cal_idx.extend(idx[:n_cal].tolist())
                te_idx.extend(idx[n_cal:].tolist())
            cal_idx = np.asarray(cal_idx, dtype=np.int64)
            te_idx = np.asarray(te_idx, dtype=np.int64)
            if len(cal_idx) and len(te_idx):
                card_ll = patient_metrics_at_threshold_modes(
                    y_c[cal_idx], p_c[cal_idx], y_c[te_idx], p_c[te_idx],
                )
                card_ll = _ensure_sens_at_spec_mode(
                    card_ll, y_c[cal_idx], p_c[cal_idx], y_c[te_idx], p_c[te_idx],
                    label="CARDIUM label-light",
                )
                _print_threshold_modes(
                    f"CARDIUM label-light (cal={len(cal_idx)} eval={len(te_idx)})",
                    card_ll,
                )
            card_oracle = patient_metrics_at_threshold_modes(y_c, p_c, y_c, p_c)
            card_oracle = _ensure_sens_at_spec_mode(
                card_oracle, y_c, p_c, y_c, p_c, label="CARDIUM oracle",
            )
        except Exception as exc:
            print(f"  WARN: CARDIUM label-light retune failed ({exc})")
        cardium_external = {
            "n_patients": int(len(card_y)),
            "fusion": card_modes,
            "auc": float(m_card.get("auc", float("nan"))),
            "f1": float(m_card.get("f1", float("nan"))),
            "label_light": card_ll,
            "oracle_full": card_oracle,
            "patient_scores": {
                "y": [int(x) for x in np.asarray(card_y).ravel()],
                "p": [float(x) for x in np.asarray(card_s).ravel()],
            },
        }
        ll_f1 = (card_ll or {}).get("val_f1_tuned", {}).get("f1")
        print(
            f"  CARDIUM external AUC={_fmt_metric(cardium_external.get('auc'))} "
            f"F1_transfer={_fmt_metric(cardium_external.get('f1'))} "
            f"F1_label_light={_fmt_metric(ll_f1)}"
        )

    zy_external = None
    zy_rows = list(getattr(args, "_zy_test_rows", None) or [])
    if zy_rows and _is_feature_fusion(args.fusion):
        print("  --- zy (private norm_4ac) external test ---")
        X_zy_raw = _rows_X(zy_rows, args, embed_cache)
        X_zy_i, _ = impute_features(X_zy_raw, fill)
        X_fit_i2, _ = impute_features(X_fit, fill)
        X_val_i2, _ = impute_features(_rows_X(val_rows, args, embed_cache), fill)
        X_te_i2, _ = impute_features(_rows_X(test_rows, args, embed_cache), fill)
        X_fit_i2, X_val_i2, X_te_i2, X_zy_i = standardize_feature_blocks(
            X_fit_i2, X_val_i2, X_te_i2, X_zy_i, n_m0=lr_n_m0, clip_norm=clip_norm,
        )
        pool = args.within_view_pool
        use_scores = pool == "score_argmax"
        zy_prob = predict_lr(clf, fill, X_zy_raw)
        zy_tok, zy_pres, zy_y, _ = patient_view_feat_tokens(
            zy_rows, X_zy_i, pool=pool, n_m0=n_m0, views=views,
            scores=zy_prob if use_scores else None,
        )
        zy_s = predict_fusion(model, zy_tok, zy_pres, device)
        zy_modes = patient_metrics_at_threshold_modes(va_y, va_s, zy_y, zy_s)
        zy_modes = _ensure_sens_at_spec_mode(
            zy_modes, va_y, va_s, zy_y, zy_s, label="zy external",
        )
        _print_threshold_modes("zy external (train-val thr)", zy_modes)
        m_zy = zy_modes.get("val_f1_tuned", {})
        zy_external = {
            "n_patients": int(len(zy_y)),
            "n_norm": int(sum(1 for y in np.asarray(zy_y).ravel() if int(y) == 0)),
            "n_abn": int(sum(1 for y in np.asarray(zy_y).ravel() if int(y) == 1)),
            "fusion": zy_modes,
            "auc": float(m_zy.get("auc", float("nan"))),
            "f1": float(m_zy.get("f1", float("nan"))),
            "patient_scores": {
                "y": [int(x) for x in np.asarray(zy_y).ravel()],
                "p": [float(x) for x in np.asarray(zy_s).ravel()],
            },
        }
        print(
            f"  zy external AUC={_fmt_metric(zy_external.get('auc'))} "
            f"F1={_fmt_metric(zy_external.get('f1'))} "
            f"n={zy_external['n_patients']} (norm={zy_external['n_norm']} abn={zy_external['n_abn']})"
        )

    return {
        "fold": fold,
        "lr_coef_summary": lr_coef_summary,
        "c2b_max": c2b_max_modes,
        "c2b_mean": c2b_mean_modes,
        "fusion": fusion_modes,
        "fusion_lr_bypass": bypass_modes,  # keyed as bypass for SUMMARY compat (= LR gate)
        "lr_gate_tau": tau_primary,
        "lr_gate_tau_f1": tau_f1,
        "lr_gate_tau_sens90": tau_s90,
        "lr_bypass_alpha": tau_primary,  # legacy alias (= gate τ)
        "lr_bypass_alpha_sens90": tau_s90,
        "lr_bypass_select": bypass_sel,
        "fusion_lr_bypass_sens90": bypass_modes_s90,
        "stratified_fusion_lr_bypass_sens90": stratified_bypass_s90,
        "fusion_lr_bypass_forced": bypass_modes_forced,
        "lr_gate_tau_forced": tau_forced if bypass_modes_forced is not None else None,
        "stratified_fusion_lr_bypass_forced": stratified_bypass_forced,
        "patient_scores": {
            "pids_val": [str(p) for p in va_pids],
            "pids_test": [str(p) for p in te_pids],
            "fusion": {
                "y_val": [int(x) for x in np.asarray(va_y).ravel()],
                "p_val": [float(x) for x in np.asarray(va_s).ravel()],
                "y_test": [int(x) for x in np.asarray(te_y).ravel()],
                "p_test": [float(x) for x in np.asarray(te_s).ravel()],
            },
            "fusion_lr_bypass": {
                "y_val": [int(x) for x in np.asarray(va_y).ravel()],
                "p_val": [float(x) for x in np.asarray(va_s_bypass).ravel()],
                "y_test": [int(x) for x in np.asarray(te_y).ravel()],
                "p_test": [float(x) for x in np.asarray(te_s_bypass).ravel()],
                "p_lr_val": [float(x) for x in np.asarray(va_lr_aligned).ravel()],
                "p_lr_test": [float(x) for x in np.asarray(te_lr_aligned).ravel()],
                "tau": tau_primary,
                "alpha": tau_primary,  # legacy
            },
            "c2b_max": {
                "y_val": [int(x) for x in np.asarray(va_ymax).ravel()],
                "p_val": [float(x) for x in np.asarray(va_smax).ravel()],
                "y_test": [int(x) for x in np.asarray(y_max).ravel()],
                "p_test": [float(x) for x in np.asarray(s_max).ravel()],
            },
            "c2b_mean": {
                "y_val": [int(x) for x in np.asarray(va_ymean).ravel()],
                "p_val": [float(x) for x in np.asarray(va_smean).ravel()],
                "y_test": [int(x) for x in np.asarray(y_mean).ravel()],
                "p_test": [float(x) for x in np.asarray(s_mean).ravel()],
            },
        },
        "stratified_fusion": stratified_fusion,
        "stratified_fusion_lr_bypass": stratified_bypass,
        "stratified_c2b_max": stratified_c2b_max,
        "test_cohort_counts": _cohort_counts(test_patient_info),
        "auc_c2b_max": m_c2b_max["auc"],
        "auc_c2b_mean": m_c2b_mean["auc"],
        "auc_fusion": m_fusion["auc"],
        "auc_fusion_lr_bypass": m_bypass["auc"],
        "auc_fusion_screening_ood": stratified_fusion["screening_ood"]["val_f1_tuned"].get("auc"),
        "spec_norm_new_device_fixed_0_5": stratified_fusion["norm_new_device"]["fixed_0.5"].get(
            "specificity"
        ),
        "f1_c2b_max": m_c2b_max["f1"],
        "f1_c2b_mean": m_c2b_mean["f1"],
        "f1_fusion": m_fusion["f1"],
        "f1_fusion_lr_bypass": m_bypass["f1"],
        "n_test_patients": int(len(te_y)),
        "view_decomp": view_decomp,
        "eval_norm_swap": eval_norm_swap,
        "cardium_external": cardium_external,
        "zy_external": zy_external,
        "seed": int(getattr(args, "seed", 42) or 42),
        "appearance_dim": int(getattr(args, "appearance_dim", 0) or 0),
    }


def _build_screening_ns(args: argparse.Namespace, tags_path: Path, feat: Path) -> argparse.Namespace:
    return argparse.Namespace(
        device=args.device, fetalclip_device=args.device, batch_size=32,
        yolo_conf=0.25, yolo_imgsz=640, min_plane_conf=0.15,
        frame_select="fetalclip_diverse_gate", k_per_view=0, view_mode="4view",
        model=args.feat_model, clip_crop="plane", crop_pad=args.crop_pad,
        rebuild_cache=args.rebuild_cache, rebuild_fetalclip_cache=False, cohort=args.cohort,
        folds=args.folds, cardium_processed=args.cardium_processed,
        feature_cache=feat, fetalclip_cache=args.fetalclip_cache,
        image_tags=tags_path, val_patient_ratio=args.val_patient_ratio,
        seed=args.seed, yolo_weights=args.yolo_weights, max_studies=0,
        manifest=args.manifest, data_root=args.data_root, m0_features=args.m0_features,
        features_only=args.features_only,
        _unreadable_skip_total=0,
    )


def _frames_need_yolo(args: argparse.Namespace, cache: dict) -> bool:
    if args.rebuild_cache:
        return True
    if args.cohort == "private":
        studies = load_private_studies(args.manifest, "")
        if getattr(args, "train_protocol", "private") in ("homologous", "real_joint"):
            # Need M0 for abnorm-train + val/test; include zy train normals for zy_unused/real_joint
            include_train_zy = _include_private_zy_train_normals(args)
            studies = [
                s for s in studies
                if s.get("split") in ("val", "test")
                or (
                    s.get("split") == "train"
                    and (int(s.get("label_binary", 0)) == 1 or include_train_zy)
                )
            ]
        # Do NOT call Path.is_file() per frame — on NFS over ~600k tertiary
        # frames this stalls for tens of minutes with zero GPU use before YOLO loads.
        need_ids = {
            f"{s['study_id']}|{rel}"
            for s in studies
            for rel in (s.get("frame_paths") or [])
        }
        if not need_ids:
            return False
        if any(sid not in cache for sid in need_ids):
            return True
        if getattr(args, "train_protocol", "private") in ("homologous", "real_joint"):
            # Probe whether midlate sample_ids are missing (use same synth as train)
            try:
                from midlate_crop_fetalclip_linear import (
                    collect_normals_midlate,
                    sample_midlate_as_multiview_synth_patients,
                    sample_midlate_as_synth_patients,
                )
                from masvf_m0_screening import MIDLATE_CN_TO_CARDIUM

                def _pid(s: dict) -> str:
                    return str(s.get("patient_id") or s.get("study_id") or "")

                abn_train_n = len({
                    _pid(s) for s in studies
                    if s.get("split") == "train" and int(s.get("label_binary", 0)) == 1
                })
                test_studies = [s for s in studies if s.get("split") == "test"] or [
                    s for s in studies if s.get("split") in ("val", "test")
                ]
                n_eval_norm = len({
                    _pid(s) for s in test_studies
                    if int(s.get("label_binary", 0)) == 0
                })
                n_eval_abn = len({
                    _pid(s) for s in test_studies
                    if int(s.get("label_binary", 0)) == 1
                })
                n_size, is_mv = _homologous_synth_size(
                    abn_train_n, n_eval_norm, n_eval_abn, args
                )
                raw = collect_normals_midlate(
                    Path(args.norm_corpus), max_n=int(getattr(args, "max_norm", 0) or 0)
                )
                if is_mv:
                    dicts = sample_midlate_as_multiview_synth_patients(
                        raw, n_size, seed=args.seed, frames_per_view=1
                    )
                else:
                    dicts = sample_midlate_as_synth_patients(raw, n_size, seed=args.seed)
                for d in dicts:
                    view_cn = str(d.get("view") or "")
                    if view_cn not in MIDLATE_CN_TO_CARDIUM and not d.get("cardium_view"):
                        continue
                    sid = f"midlate|{view_cn}|{Path(d['path']).stem}"
                    if sid not in cache:
                        return True
            except Exception as exc:
                print(f"[warn] midlate cache probe failed ({exc}); will load YOLO")
                return True
        return False
    folds = [x.strip() for x in args.folds.split(",") if x.strip()]
    for fold in folds:
        for sp in ("train", "test"):
            for rec in load_fold_split(fold, sp, args.cardium_processed):
                if rec.sample_id not in cache:
                    return True
    return False


def _load_yolo_if_needed(args: argparse.Namespace, cache: dict):
    if not _frames_need_yolo(args, cache):
        print("Feature cache hit → skip YOLO")
        return None
    if args.features_only:
        raise SystemExit(
            "ERROR: --features-only but M0 feature cache incomplete. "
            "Run once without --features-only (or run run_masvf_m0_private.sh) to backfill cache."
        )
    weights = resolve_weights(args.yolo_weights)
    print(f"Loading YOLO to backfill M0 feature cache: {weights}")
    return load_yolo_model(weights)


def _load_tags_and_cache(args: argparse.Namespace) -> tuple[Path, dict, Path]:
    if args.image_tags is None:
        screen = PROJECT_ROOT / "data" / "study_screening"
        if args.cohort == "private":
            tags_path = screen / "yolo_image_tags_private.jsonl"
        else:
            tags_path = screen / "yolo_image_tags_cardium_anatomy.jsonl"
            if not tags_path.is_file():
                tags_path = screen / "yolo_image_tags_cardium_xyxy.jsonl"
    else:
        tags_path = args.image_tags
    tags = load_tags(resolve_tags(tags_path))
    pair = str(getattr(args, "swap_views", "") or "").strip()
    if pair:
        a, b = [x.strip() for x in pair.split(",")]
        n = 0
        for sid, row in tags.items():
            if sid == "_by_rel" or not isinstance(row, dict):
                continue
            v = row.get("cardium_view")
            if v == a:
                row["cardium_view"] = b
                n += 1
            elif v == b:
                row["cardium_view"] = a
                n += 1
        rel = tags.get("_by_rel") or {}
        for row in rel.values():
            if not isinstance(row, dict):
                continue
            v = row.get("cardium_view")
            if v == a:
                row["cardium_view"] = b
            elif v == b:
                row["cardium_view"] = a
        print(f"  SWAP_VIEWS {a}<->{b} retagged_sample_ids={n}", flush=True)
    feat = args.feature_cache
    if feat is None:
        feat = PRIVATE_FEATURE_CACHE if args.cohort == "private" else CARDIUM_FEATURE_CACHE
    feat = Path(feat)
    if not feat.is_file() and args.cohort == "cardium":
        legacy = PROJECT_ROOT / "data" / "study_screening" / "cardium_det_feature_cache.jsonl"
        if legacy.is_file():
            feat = legacy
    return tags_path, tags, feat


def _apply_fallback_policy(all_rows: list, fallback: str) -> list:
    """drop=4view only; token=keep YOLO-miss as fallback view."""
    if fallback == "drop":
        return filter_view_mode(all_rows, "4view")
    kept = []
    n_fb = 0
    for r in all_rows:
        if r.cardium_view in VIEWS:
            kept.append(r)
        elif not r.cardium_view:
            r.cardium_view = FALLBACK_VIEW
            meta = dict(r.meta or {})
            meta["fallback"] = True
            r.meta = meta
            kept.append(r)
            n_fb += 1
        # non-standard named views still dropped
    print(f"  fallback=token kept_standard+fallback frames={len(kept)} (fallback={n_fb})")
    return kept


def _eval_norm_abn_patients(rows) -> tuple[int, int]:
    """Prefer test split for ratio; fall back to all eval rows."""
    test = [r for r in rows if getattr(r, "split", "") == "test"]
    use = test if test else list(rows)
    n_norm = len({r.patient_id for r in use if int(r.label) == 0})
    n_abn = len({r.patient_id for r in use if int(r.label) == 1})
    return n_norm, n_abn


def _homologous_synth_size(
    n_abn_train: int,
    n_eval_norm: int,
    n_eval_abn: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Return (n_synth_patients or frames_per_view, mode_flag).

    multiview: returns (n_patients, 1) where 1 = frames_per_view within patient.
    frame: returns (frames_per_view, 0) legacy.
    """
    mode = str(getattr(args, "midlate_synth_mode", "multiview") or "multiview").lower()
    fixed = int(getattr(args, "frames_per_view_norm", 0) or 0)
    balance = bool(getattr(args, "balance_norm_to_eval", True))

    if mode == "frame":
        if fixed > 0:
            print(f"  midlate mode=frame frames_per_view={fixed} (explicit)")
            return fixed, 0
        if not balance:
            print("  midlate mode=frame frames_per_view=4 (fallback)")
            return 4, 0
        from masvf_m0_screening import MIDLATE_CN_TO_CARDIUM
        n_views = max(1, len(MIDLATE_CN_TO_CARDIUM))
        target_norm = max(
            n_views,
            int(round(n_abn_train / max(n_eval_abn, 1) * n_eval_norm)),
        )
        fpv = max(1, (target_norm + n_views - 1) // n_views)
        print(
            f"  midlate mode=frame balance → target_patients≈{target_norm} "
            f"frames_per_view={fpv} (1 frame = 1 patient; NOT for fusion)"
        )
        return fpv, 0

    # multiview: target is number of multi-view synth patients
    if fixed > 0 and not balance:
        print(f"  midlate mode=multiview n_patients={fixed} (explicit)")
        return fixed, 1
    if fixed > 0 and balance:
        # explicit overrides auto
        print(f"  midlate mode=multiview n_patients={fixed} (explicit)")
        return fixed, 1
    if not balance:
        print("  midlate mode=multiview n_patients=64 (fallback)")
        return 64, 1

    target_norm = max(
        1,
        int(round(n_abn_train / max(n_eval_abn, 1) * n_eval_norm)),
    )
    print(
        f"  balance midlate→test prevalence (MULTIVIEW bags):\n"
        f"    test norm:abn = {n_eval_norm}:{n_eval_abn}\n"
        f"    train abn = {n_abn_train}\n"
        f"    target_norm_patients = {n_abn_train}/{n_eval_abn}*{n_eval_norm} "
        f"= {target_norm}\n"
        f"    → each synth patient = 1 frame × {{4CH,LVOT,RVOT,VVT}} "
        f"(total frames≈{target_norm * 4})"
    )
    return target_norm, 1


def _homologous_frames_per_view(
    n_abn_train: int,
    n_eval_norm: int,
    n_eval_abn: int,
    args: argparse.Namespace,
) -> int:
    """Back-compat wrapper: returns primary size (n_patients or frames_per_view)."""
    size, _ = _homologous_synth_size(n_abn_train, n_eval_norm, n_eval_abn, args)
    return size


def _resolved_train_norm_src(args: argparse.Namespace, train_protocol: str) -> str:
    src = str(getattr(args, "homologous_train_norm", "midlate") or "midlate").lower()
    if src == "zy_real":
        src = "zy_unused"
    if train_protocol == "real_joint" and src not in ("cardium_only",):
        src = "zy_unused"
    return src


def _include_private_zy_train_normals(args: argparse.Namespace) -> bool:
    proto = getattr(args, "train_protocol", "private")
    if proto not in ("homologous", "real_joint"):
        return False
    src = _resolved_train_norm_src(args, proto)
    return src in ("zy_unused", "zy_plus_1view")


def _select_zy_train_rows(
    private_rows: list,
    *,
    n_size: int,
    seed: int,
) -> tuple[list, set[str], list[str]]:
    """Pick train-split zy normal frames (patient subsample if n_size>0)."""
    zy_train_all = [r for r in private_rows if r.split == "train" and int(r.label) == 0]
    zy_pids = sorted({str(r.patient_id) for r in zy_train_all})
    if not zy_pids:
        raise SystemExit(
            "ERROR: need train-split normal patients in manifest "
            "(NORMAL_SOURCE=zy / source_cohort=norm_4ac)."
        )
    n_target = int(n_size) if int(n_size) > 0 else len(zy_pids)
    n_target = min(n_target, len(zy_pids))
    rng = random.Random(int(seed))
    pick = set(rng.sample(zy_pids, n_target)) if n_target < len(zy_pids) else set(zy_pids)
    train_norm_rows = [r for r in zy_train_all if str(r.patient_id) in pick]
    return train_norm_rows, pick, zy_pids


def _attach_cardium_joint_rows(
    args: argparse.Namespace,
    all_rows: list,
    *,
    yolo_model,
    ns: argparse.Namespace,
    title: str = "real_joint",
) -> list:
    """Append CARDIUM fold train/test rows for real_joint protocols."""
    fold = int(getattr(args, "joint_cardium_fold", 1) or 1)
    cardium_tags_path = (
        PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_cardium.jsonl"
    )
    cardium_tags = load_tags(cardium_tags_path)
    cardium_cache = load_feature_cache(CARDIUM_FEATURE_CACHE)
    cardium_ns = _build_screening_ns(args, cardium_tags_path, CARDIUM_FEATURE_CACHE)
    cardium_train_rows: list = []
    cardium_test_rows: list = []
    for sp, out_split in (("train", "train"), ("test", "test_cardium")):
        records = load_fold_split(str(fold), sp, args.cardium_processed)
        cr = build_cardium_frame_rows(
            records,
            tags=cardium_tags,
            yolo_model=yolo_model,
            args=cardium_ns,
            cache=cardium_cache,
        )
        for r in cr:
            tag_row_domain(r, DOMAIN_CARDIUM)
            prefix_patient_id(r, "cardium|")
            r.split = out_split
            r.fold = fold
        if sp == "train":
            cardium_train_rows = cr
        else:
            cardium_test_rows = cr
    merged = all_rows + cardium_train_rows + cardium_test_rows
    args._cardium_test_rows = cardium_test_rows
    cov = coverage_report(merged)
    print_coverage_report(cov, title=f"{title} coverage (fold {fold})")
    if getattr(args, "coverage_report", None):
        write_coverage_json(cov, Path(args.coverage_report))
    n_card_tr = len({r.patient_id for r in cardium_train_rows})
    n_card_te = len({r.patient_id for r in cardium_test_rows})
    n_card_tr_norm = len({r.patient_id for r in cardium_train_rows if int(r.label) == 0})
    n_card_tr_abn = len({r.patient_id for r in cardium_train_rows if int(r.label) == 1})
    print(
        f"  +CARDIUM fold{fold}: train_frames={len(cardium_train_rows)} "
        f"(patients={n_card_tr}; norm={n_card_tr_norm} abn={n_card_tr_abn}) "
        f"test_frames={len(cardium_test_rows)} (patients={n_card_te})"
    )
    return merged


def _attach_eval_cardium_holdout(
    args: argparse.Namespace,
    all_rows: list,
    *,
    yolo_model,
    ns: argparse.Namespace,
) -> list:
    """Eval-only: CARDIUM fold test → args._cardium_test_rows (no train merge)."""
    fold = int(getattr(args, "joint_cardium_fold", 1) or 1)
    cardium_tags_path = (
        PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_cardium.jsonl"
    )
    if not cardium_tags_path.is_file():
        print(f"  WARN: --eval-cardium skipped (missing {cardium_tags_path})")
        args._cardium_test_rows = []
        return all_rows
    cardium_tags = load_tags(cardium_tags_path)
    cardium_cache = load_feature_cache(CARDIUM_FEATURE_CACHE)
    cardium_ns = _build_screening_ns(args, cardium_tags_path, CARDIUM_FEATURE_CACHE)
    cardium_ns.allow_missing_cache = True
    records = load_fold_split(str(fold), "test", args.cardium_processed)
    cr = build_cardium_frame_rows(
        records,
        tags=cardium_tags,
        yolo_model=yolo_model,
        args=cardium_ns,
        cache=cardium_cache,
    )
    for r in cr:
        tag_row_domain(r, DOMAIN_CARDIUM)
        prefix_patient_id(r, "cardium|")
        r.split = "test_cardium"
        r.fold = fold
    args._cardium_test_rows = cr
    n_pat = len({r.patient_id for r in cr})
    print(f"  +eval-cardium fold{fold}: frames={len(cr)} patients={n_pat} (holdout only)")
    return all_rows + cr


def _attach_eval_zy_holdout(
    args: argparse.Namespace,
    all_rows: list,
    *,
    yolo_model,
    ns: argparse.Namespace,
) -> list:
    """Eval-only: private zy (norm_4ac) val/test + private CHD val/test."""
    zy_manifest = Path(
        getattr(args, "zy_manifest", None)
        or (PROJECT_ROOT / "data" / "study_screening" / "manifest.jsonl")
    )
    zy_tags_path = Path(
        getattr(args, "zy_tags", None)
        or (PROJECT_ROOT / "data" / "study_screening" / "yolo_image_tags_private.jsonl")
    )
    if not zy_manifest.is_file():
        print(f"  WARN: --eval-zy skipped (missing {zy_manifest})")
        args._zy_test_rows = []
        return all_rows
    zy_tags = load_tags(zy_tags_path) if zy_tags_path.is_file() else {}
    zy_cache = load_feature_cache(PRIVATE_FEATURE_CACHE)
    zy_ns = _build_screening_ns(args, zy_tags_path, PRIVATE_FEATURE_CACHE)
    zy_ns.allow_missing_cache = True
    studies = [
        s for s in load_private_studies(zy_manifest, "")
        if s.get("split") in ("val", "test")
    ]
    cr = build_private_frame_rows(
        studies,
        data_root=args.data_root,
        tags=zy_tags,
        yolo_model=yolo_model,
        args=zy_ns,
        cache=zy_cache,
    )
    for r in cr:
        tag_row_domain(r, DOMAIN_PRIVATE)
        prefix_patient_id(r, "zy|")
        r.split = "test_zy"
    args._zy_test_rows = cr
    n_pat = len({r.patient_id for r in cr})
    n_norm = len({r.patient_id for r in cr if int(r.label) == 0})
    n_abn = len({r.patient_id for r in cr if int(r.label) == 1})
    print(
        f"  +eval-zy: frames={len(cr)} patients={n_pat} "
        f"(norm={n_norm} abn={n_abn}; private val+test)"
    )
    return all_rows + cr


def _build_rows_and_embed(
    args: argparse.Namespace,
    tags: dict,
    cache: dict,
    ns: argparse.Namespace,
    yolo_model,
) -> tuple[list, dict | None]:
    train_protocol = getattr(args, "train_protocol", "private")
    if args.cohort == "private" and train_protocol in ("homologous", "real_joint"):
        train_norm_src = _resolved_train_norm_src(args, train_protocol)
        norm_desc = {
            "zy_unused": "zy train-split real patient bags",
            "cardium_only": "CARDIUM fold-train normals only (no zy train normals)",
            "midlate": "Frankenstein midlate synth bags",
        }.get(train_norm_src, train_norm_src)
        print(
            "=== Homologous / real_joint train protocol ===\n"
            f"  protocol = {train_protocol}\n"
            f"  train_norm_source = {train_norm_src} ({norm_desc})\n"
            "  train = train-normals + abnorm train (+ CARDIUM train if real_joint)\n"
            "  eval  = private val/test (held-out zy norm + homologous abnorm)"
        )
        studies_all = load_private_studies(args.manifest, "")
        include_train_zy = train_norm_src in ("zy_unused", "zy_plus_1view")
        studies = [
            s for s in studies_all
            if s.get("split") in ("val", "test")
            or (
                s.get("split") == "train"
                and (int(s.get("label_binary", 0)) == 1 or include_train_zy)
            )
        ]
        private_rows = build_private_frame_rows(
            studies, data_root=args.data_root, tags=tags, yolo_model=yolo_model, args=ns, cache=cache,
        )
        for r in private_rows:
            tag_row_domain(r, DOMAIN_PRIVATE)
        abn_train = [r for r in private_rows if r.split == "train" and int(r.label) == 1]
        eval_rows = [r for r in private_rows if r.split in ("val", "test")]
        n_abn_pat = len({r.patient_id for r in abn_train})
        n_eval_norm, n_eval_abn = _eval_norm_abn_patients(eval_rows)
        n_size, is_mv = _homologous_synth_size(n_abn_pat, n_eval_norm, n_eval_abn, args)
        mode = str(getattr(args, "midlate_synth_mode", "multiview") or "multiview").lower()
        args._resolved_frames_per_view_norm = n_size
        args._resolved_midlate_synth_mode = mode
        args._resolved_homologous_train_norm = train_norm_src

        if train_norm_src == "cardium_only":
            if train_protocol != "real_joint":
                raise SystemExit(
                    "ERROR: --homologous-train-norm=cardium_only requires --train-protocol real_joint"
                )
            args._same_machine_eval_rows = []
            all_rows = abn_train + eval_rows
            n_abn_pat = len({r.patient_id for r in abn_train})
            print(
                "  cardium_only: private train = abnorm train ONLY "
                f"(patients={n_abn_pat}); normals from CARDIUM train fold"
            )
            print(
                f"  eval held-out zy: norm={n_eval_norm} abn={n_eval_abn} "
                "(patient-disjoint from all train sources)"
            )
            all_rows = _attach_cardium_joint_rows(
                args, all_rows, yolo_model=yolo_model, ns=ns, title="cardium_only",
            )
        elif train_norm_src == "zy_unused":
            # Frankenstein control: previously discarded manifest train-split zy
            # real patient-level normals (patient-disjoint from val/test zy).
            if bool(getattr(args, "eval_norm_swap", False)):
                print(
                    "  WARN: --eval-norm-swap ignored under homologous-train-norm=zy_unused "
                    "(E1 midlate holdout is midlate-only)"
                )
            args._same_machine_eval_rows = []
            train_norm_rows, pick, zy_pids = _select_zy_train_rows(
                private_rows, n_size=int(n_size), seed=int(args.seed),
            )
            all_rows = train_norm_rows + abn_train + eval_rows
            n_trn = len(pick)
            from collections import Counter
            views_per = Counter()
            by_p: dict[str, set] = {}
            for r in train_norm_rows:
                by_p.setdefault(str(r.patient_id), set()).add(r.cardium_view)
            for vs in by_p.values():
                views_per[len(vs)] += 1
            print(
                f"  Frankenstein CTRL (zy_unused): available_train_zy={len(zy_pids)} "
                f"selected={n_trn} (target≈{n_size}; leftover_unused={len(zy_pids) - n_trn}) "
                f"| patient-disjoint from val/test zy"
            )
            print(
                f"  homologous frames: zy_train={len(train_norm_rows)} (patients={n_trn}) "
                f"abn_train={len(abn_train)} (patients={n_abn_pat}) eval={len(eval_rows)} "
                f"| train norm:abn patients = {n_trn}:{n_abn_pat} "
                f"(eval test ratio {n_eval_norm}:{n_eval_abn})"
            )
            print(
                f"  zy_unused view_coverage_hist={dict(sorted(views_per.items()))} "
                f"(real patient bags; expect multi-view)"
            )
            print(
                "  NOTE: same frozen LoRA as main; isolates fusion train-norm construction "
                "(Frankenstein midlate vs real zy bags). Eval remains val/test zy → "
                "same-machine train/test normals (Frankenstein vs real bags; "
                "not a held-out-machine claim)."
            )
            if train_protocol == "real_joint":
                all_rows = _attach_cardium_joint_rows(
                    args, all_rows, yolo_model=yolo_model, ns=ns, title="real_joint",
                )
        elif train_norm_src == "zy_plus_1view":
            # Main experimental recipe (no CARDIUM train, no Frankenstein 4-view stitch):
            #   train normals = zy train bags + midlate single-plane as 1-view patients
            #   train pos     = private CHD train
            #   eval          = private val/test (held-out zy + CHD)
            #   external      = optional --eval-cardium holdout
            if train_protocol == "real_joint":
                raise SystemExit(
                    "ERROR: zy_plus_1view is homologous-only (CARDIUM must stay eval-only). "
                    "Use --train-protocol homologous --eval-cardium instead of real_joint."
                )
            if bool(getattr(args, "eval_norm_swap", False)):
                print("  WARN: --eval-norm-swap ignored under zy_plus_1view")
            args._same_machine_eval_rows = []
            # Prefer all zy train bags (n_size=0); optional cap via frames-per-view-norm if set
            # and BALANCE only when user explicitly sets frames-per-view-norm > 0.
            zy_cap = int(getattr(args, "frames_per_view_norm", 0) or 0)
            # For zy selection: 0 = all. Don't use auto-balance n_size (that was for midlate).
            train_norm_rows, pick, zy_pids = _select_zy_train_rows(
                private_rows, n_size=zy_cap, seed=int(args.seed),
            )
            n_trn = len(pick)

            ov = int(getattr(args, "oneview_per_folder", -1))
            if ov < 0:
                # default: generous 1-view pool; 0 in frames_per_view_norm means "all planes"
                per_folder = 0 if zy_cap == 0 else max(1, zy_cap)
            else:
                per_folder = int(ov)

            args._resolved_midlate_synth_mode = "frame"
            oneview_rows = build_midlate_frame_rows(
                norm_corpus=Path(args.norm_corpus),
                frames_per_view=per_folder,
                seed=int(args.seed) + 17,
                yolo_model=yolo_model,
                args=ns,
                cache=cache,
                max_norm=int(getattr(args, "max_norm", 0) or 0),
                synth_mode="frame",
                n_synth_patients=0,
            )
            for r in oneview_rows:
                tag_row_domain(r, DOMAIN_PRIVATE)
                r.split = "train"
                r.label = 0
                meta = dict(r.meta or {})
                meta["oneview_missing3"] = True
                meta["homologous_midlate"] = True
                r.meta = meta

            all_rows = train_norm_rows + oneview_rows + abn_train + eval_rows
            n_1v = len({r.patient_id for r in oneview_rows})
            from collections import Counter
            views_per = Counter()
            by_p: dict[str, set] = {}
            for r in train_norm_rows:
                by_p.setdefault(str(r.patient_id), set()).add(r.cardium_view)
            for vs in by_p.values():
                views_per[len(vs)] += 1
            ov_views = Counter()
            for r in oneview_rows:
                ov_views[str(r.cardium_view or "?")] += 1
            print(
                f"  PROTOCOL zy_plus_1view (no CARDIUM train, no Frankenstein stitch)\n"
                f"    zy_train patients={n_trn}/{len(zy_pids)} frames={len(train_norm_rows)} "
                f"view_hist={dict(sorted(views_per.items()))}\n"
                f"    midlate_1view patients={n_1v} frames={len(oneview_rows)} "
                f"per_folder={per_folder if per_folder > 0 else 'all'} "
                f"slots={dict(ov_views)}\n"
                f"    abn_train patients={n_abn_pat} frames={len(abn_train)}\n"
                f"    eval val+test frames={len(eval_rows)} "
                f"(held-out zy norm={n_eval_norm} abn={n_eval_abn})\n"
                f"    1-view bags → present mask has 1 True / 3 False (missing_fill)"
            )
        else:
            n_hold = 0
            if bool(getattr(args, "eval_norm_swap", False)):
                n_hold = int(getattr(args, "midlate_eval_n", 0) or 0)
                if n_hold <= 0:
                    n_hold = int(n_eval_norm)
                print(
                    f"  E1 eval-norm-swap: hold out {n_hold} midlate patients "
                    f"(same-machine normals) for dual eval"
                )
            n_build = int(n_size) + int(n_hold)
            if is_mv:
                midlate_rows = build_midlate_frame_rows(
                    norm_corpus=Path(args.norm_corpus),
                    frames_per_view=1,
                    seed=args.seed,
                    yolo_model=yolo_model,
                    args=ns,
                    cache=cache,
                    max_norm=int(getattr(args, "max_norm", 0) or 0),
                    synth_mode="multiview",
                    n_synth_patients=n_build,
                )
            else:
                midlate_rows = build_midlate_frame_rows(
                    norm_corpus=Path(args.norm_corpus),
                    frames_per_view=n_build,
                    seed=args.seed,
                    yolo_model=yolo_model,
                    args=ns,
                    cache=cache,
                    max_norm=int(getattr(args, "max_norm", 0) or 0),
                    synth_mode="frame",
                    n_synth_patients=0,
                )
            hold_rows: list = []
            if n_hold > 0:
                # Consecutive synth IDs: first n_size train, remainder same-machine eval normals.
                pids_ordered = sorted({str(r.patient_id) for r in midlate_rows})
                # Prefer numeric order of midlate|mv|##### when present
                def _pid_key(p: str):
                    if "|mv|" in p:
                        try:
                            return (0, int(p.rsplit("|", 1)[-1]))
                        except ValueError:
                            return (1, p)
                    return (1, p)
                pids_ordered = sorted(pids_ordered, key=_pid_key)
                train_pids = set(pids_ordered[: int(n_size)])
                hold_pids = set(pids_ordered[int(n_size): int(n_size) + int(n_hold)])
                train_mid = []
                for r in midlate_rows:
                    pid = str(r.patient_id)
                    if pid in hold_pids:
                        r.split = "test_same_machine"
                        hold_rows.append(r)
                    elif pid in train_pids or not hold_pids:
                        train_mid.append(r)
                midlate_rows = train_mid
                args._same_machine_eval_rows = hold_rows
                print(
                    f"  E1 split midlate: train_patients={len(train_pids)} "
                    f"same_machine_eval_patients={len(hold_pids)} "
                    f"frames_hold={len(hold_rows)}"
                )
            else:
                args._same_machine_eval_rows = []
            all_rows = midlate_rows + hold_rows + abn_train + eval_rows
            n_mid_pat = len({r.patient_id for r in midlate_rows})
            # view coverage of synth norms
            from collections import Counter
            views_per = Counter()
            by_p: dict[str, set] = {}
            for r in midlate_rows:
                by_p.setdefault(str(r.patient_id), set()).add(r.cardium_view)
            for vs in by_p.values():
                views_per[len(vs)] += 1
            print(
                f"  homologous frames: midlate={len(midlate_rows)} (patients={n_mid_pat}) "
                f"abn_train={len(abn_train)} (patients={n_abn_pat}) eval={len(eval_rows)} "
                f"| train norm:abn patients = {n_mid_pat}:{n_abn_pat} "
                f"(eval test ratio {n_eval_norm}:{n_eval_abn})"
            )
            print(
                f"  midlate synth_mode={mode} | "
                f"frames/patient≈{len(midlate_rows) / max(n_mid_pat, 1):.1f} | "
                f"view_coverage_hist={dict(sorted(views_per.items()))} "
                f"(expect ~4 views/patient in multiview)"
            )
            abn_fp = len(abn_train) / max(n_abn_pat, 1)
            print(
                f"  compare density: midlate frames/patient≈"
                f"{len(midlate_rows) / max(n_mid_pat, 1):.1f} vs "
                f"abn frames/patient≈{abn_fp:.1f} "
                f"(fusion pools to 1 token/view, so 4-view coverage matters more than raw frames)"
            )
    elif args.cohort == "private":
        studies = load_private_studies(args.manifest, "")
        all_rows = build_private_frame_rows(
            studies, data_root=args.data_root, tags=tags, yolo_model=yolo_model, args=ns, cache=cache,
        )
    else:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
        all_records = []
        for fold in folds:
            for sp in ("train", "test"):
                all_records.extend(load_fold_split(str(fold), sp, args.cardium_processed))
        all_rows = build_cardium_frame_rows(
            all_records, tags=tags, yolo_model=yolo_model, args=ns, cache=cache,
        )

    # OOD eval bags (embed with train data; never enter train/val/test splits)
    if bool(getattr(args, "eval_cardium", False)):
        if getattr(args, "_cardium_test_rows", None) is None:
            all_rows = _attach_eval_cardium_holdout(
                args, all_rows, yolo_model=yolo_model, ns=ns,
            )
    if bool(getattr(args, "eval_zy", False)):
        if getattr(args, "_zy_test_rows", None) is None:
            all_rows = _attach_eval_zy_holdout(
                args, all_rows, yolo_model=yolo_model, ns=ns,
            )

    all_rows = _apply_fallback_policy(all_rows, args.fallback)

    # Embed items from actual rows (covers midlate paths not in manifest)
    items: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for r in all_rows:
        if r.sample_id in seen:
            continue
        seen.add(r.sample_id)
        items.append((r.sample_id, Path(r.image_path)))

    crop_boxes = collect_crop_boxes(tags, items)
    # Midlate / private meta GT boxes override missing tags
    for r in all_rows:
        xy = (r.meta or {}).get("plane_xyxy")
        if isinstance(xy, (list, tuple)) and len(xy) == 4:
            try:
                crop_boxes[r.sample_id] = (float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3]))
            except (TypeError, ValueError):
                pass
    n_force = 0
    for r in all_rows:
        if r.cardium_view == FALLBACK_VIEW:
            crop_boxes[r.sample_id] = None
            n_force += 1
    if n_force:
        print(f"  forced full-image CLIP for {n_force} fallback frames")
    _guard_imagenet_cache_not_polluted_by_fetalclip(args)
    if getattr(args, "embed_load_only", False):
        from agcd.fetalclip_embed import load_embed_cache

        if not Path(args.fetalclip_cache).is_file():
            raise SystemExit(
                f"ERROR: --embed-load-only requires existing cache: {args.fetalclip_cache}"
            )
        embed_cache = load_embed_cache(Path(args.fetalclip_cache))
        print(f"  embed_load_only: loaded {len(embed_cache)} vectors from {args.fetalclip_cache}")
    else:
        embed_cache = build_or_load_embeddings(
            items,
            cache_path=args.fetalclip_cache,
            device=str(args.device),
            batch_size=32,
            crop_boxes=crop_boxes,
            crop_pad=args.crop_pad,
            lora_adapter=args.lora_adapter,
        )
    all_rows = apply_frame_selection(all_rows, tags, ns, embed_cache)
    if args.feat_model == "m0":
        return all_rows, None
    return all_rows, embed_cache


def _result_stem(args: argparse.Namespace) -> str:
    extra = getattr(args, "extra_feats", "none") or "none"
    stem = (
        f"{args.fusion}"
        f"__feat-{args.feat_model}"
        f"__m0-{args.m0_features}"
        f"__pool-{args.within_view_pool}"
        f"__fb-{args.fallback}"
    )
    if extra != "none":
        stem += f"__xa-{extra}"
    if getattr(args, "lora_adapter", None) is not None:
        stem += "__clip-homologous_lora"
    app_dim = int(getattr(args, "appearance_dim", 0) or 0)
    if app_dim > 0 and app_dim != FETALCLIP_EMBED_DIM:
        stem += f"__adim-{app_dim}"
    if getattr(args, "embed_load_only", False):
        stem += "__loadonly"
    cn = str(getattr(args, "clip_norm", "hybrid") or "hybrid")
    stem += f"__cn-{cn}"
    if getattr(args, "norm_view_drop", False):
        stem += "__nvdrop"
    if getattr(args, "train_protocol", "private") == "homologous":
        stem += "__tp-homologous"
        tn = str(getattr(args, "_resolved_homologous_train_norm", None)
                 or getattr(args, "homologous_train_norm", "midlate") or "midlate")
        if tn not in ("midlate",):
            stem += f"__tn-{tn}"
        mode = str(getattr(args, "_resolved_midlate_synth_mode", None)
                   or getattr(args, "midlate_synth_mode", "multiview") or "multiview")
        stem += f"__ms-{mode}"
        fpv = int(getattr(args, "_resolved_frames_per_view_norm", 0) or 0)
        if fpv <= 0:
            fpv = int(getattr(args, "frames_per_view_norm", 0) or 0)
        stem += f"__np-{fpv if fpv > 0 else 'auto'}"
    if getattr(args, "train_protocol", "private") == "real_joint":
        stem += "__tp-real_joint"
        tn = str(
            getattr(args, "_resolved_homologous_train_norm", None)
            or getattr(args, "homologous_train_norm", "")
            or ""
        )
        if tn == "cardium_only":
            stem += "__tn-cardium_only"
        stem += f"__cjf-{int(getattr(args, 'joint_cardium_fold', 1) or 1)}"
    if bool(getattr(args, "dann", False)):
        stem += "__dann"
    if args.fusion == "graph_transformer":
        stem += "__graph"
    if args.fusion == "anatomy_graph":
        stem += "__alvg"
    gadj = str(getattr(args, "_resolved_graph_adj", None)
               or getattr(args, "graph_adj", "auto") or "auto")
    if gadj not in ("auto", "") and not (
        args.fusion == "anatomy_graph" and gadj == "anatomy"
    ) and not (
        args.fusion == "graph_transformer" and gadj in ("auto", "full")
    ):
        stem += f"__gadj-{gadj}"
    if bool(getattr(args, "no_anatomy_encoder_mask", False)):
        stem += "__alvg-noencmask"
    if bool(getattr(args, "no_mvp", False)):
        stem += "__nomvp"
    elif bool(getattr(args, "mvp", False)) or _is_graph_fusion(args.fusion):
        stem += "__mvp"
    if bool(getattr(args, "no_view_embed", False)):
        stem += "__noviewid"
    # Always pass fill so stem gets __miss-zero when non-learned; learned stays untagged for E5 baseline
    mf = str(getattr(args, "missing_fill", "zero") or "zero")
    if mf != "learned":
        stem += f"__miss-{mf}"
    pm = str(getattr(args, "pool_mask", "present") or "present")
    if pm != "present":
        stem += f"__poolmask-{pm}"
    if bool(getattr(args, "eval_norm_swap", False)):
        stem += "__normswap"
    if bool(getattr(args, "eval_cardium", False)):
        stem += "__evalcardium"
    if bool(getattr(args, "eval_zy", False)):
        stem += "__evalzy"
    seed = int(getattr(args, "seed", 42) or 42)
    if seed != 42:
        stem += f"__seed-{seed}"
    return stem


def _print_disease_split_audit(train_rows, test_rows) -> None:
    """Warn if any abnorm disease is train-only or test-only (patient-level)."""
    from collections import Counter

    def _disease(r) -> str:
        name = str(getattr(r, "label_name", "") or "")
        if name and name not in ("midlate_normal", "normal", ""):
            return name
        meta = getattr(r, "meta", None) or {}
        return str(meta.get("label_disease") or meta.get("disease") or "unk")

    def _abn_disease_patients(rows):
        out: dict[str, set[str]] = {}
        for r in rows:
            if int(r.label) != 1:
                continue
            d = _disease(r)
            out.setdefault(d, set()).add(str(r.patient_id))
        return out

    tr = _abn_disease_patients(train_rows)
    te = _abn_disease_patients(test_rows)
    all_d = sorted(set(tr) | set(te))
    if not all_d:
        return
    print("  disease × split (abn patients):")
    only_tr, only_te = [], []
    for d in all_d:
        ntr, nte = len(tr.get(d, ())), len(te.get(d, ()))
        flag = ""
        if ntr == 0:
            flag = "  *** TEST-ONLY"
            only_te.append(d)
        elif nte == 0:
            flag = "  *** TRAIN-ONLY"
            only_tr.append(d)
        tot = ntr + nte
        ptr = 100.0 * ntr / tot if tot else 0.0
        pte = 100.0 * nte / tot if tot else 0.0
        print(f"    {d:16s} train={ntr:4d} ({ptr:4.1f}%)  test={nte:4d} ({pte:4.1f}%){flag}")
    if only_tr or only_te:
        print(
            f"  WARN: disease not in both splits — "
            f"train_only={only_tr} test_only={only_te}. "
            f"Rebuild manifest with --stratify-disease 1."
        )
    else:
        print("  disease audit OK: every abn disease appears in train and test")


def main() -> None:
    args = parse_args()
    d_in = feat_dim(args)
    nv = n_views_for(args)
    if args.dry_run:
        m = make_fusion_model(args, d_in=d_in)
        if _is_mil_fusion(args.fusion):
            t = torch.randn(2, args.mil_max_frames, d_in)
            p = torch.zeros(2, args.mil_max_frames, dtype=torch.bool)
            p[:, :4] = True
        elif _is_feature_fusion(args.fusion):
            t = torch.randn(2, nv, d_in)
            p = torch.zeros(2, nv, dtype=torch.bool)
            p[:, : min(4, nv)] = True
        else:
            t = torch.rand(2, nv)
            p = torch.zeros(2, nv, dtype=torch.bool)
            p[:, : min(4, nv)] = True
        print("dry-run", args.fusion, "d_in", d_in, "n_views", nv, m(t, p).shape)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(str(args.device))
    xa = getattr(args, "extra_feats", "none")
    xa_note = f" + view_anat {view_anat_frame_dim()}" if xa == "view_anat" else ""
    lora_note = f" lora={args.lora_adapter.name}" if args.lora_adapter else ""
    app_note = ""
    if args.feat_model != "m0":
        ad = int(getattr(args, "appearance_dim", 0) or FETALCLIP_EMBED_DIM)
        app_note = f" + app {ad}"
    print(
        f"cohort={args.cohort} device={device} fusion={args.fusion} fallback={args.fallback} "
        f"feat={args.feat_model} m0={args.m0_features} pool={args.within_view_pool} "
        f"extra={xa} d_in={d_in} n_views={nv} (=M0 {n_m0_dims(args)}"
        f"{app_note}"
        f"{xa_note}){lora_note}"
    )

    tags_path, tags, feat = _load_tags_and_cache(args)
    args._tags = tags
    if not feat.is_file():
        feat.parent.mkdir(parents=True, exist_ok=True)
        feat.write_text("", encoding="utf-8")
        print(f"WARN: created empty feature cache {feat} (will YOLO-backfill)")
    if not args.fetalclip_cache.is_file():
        args.fetalclip_cache.parent.mkdir(parents=True, exist_ok=True)
        args.fetalclip_cache.write_text("", encoding="utf-8")
        print(
            f"WARN: created empty FetalCLIP cache {args.fetalclip_cache} "
            "(will encode on first run)"
        )
    cache = load_feature_cache(feat)
    ns = _build_screening_ns(args, tags_path, feat)
    yolo_model = _load_yolo_if_needed(args, cache)
    if yolo_model is not None:
        print("YOLO loaded for M0 backfill — GPU should show memory now", flush=True)

    print("=== Build C2-b frames ===")
    all_rows, embed_cache = _build_rows_and_embed(args, tags, cache, ns, yolo_model)
    args.appearance_dim = _infer_appearance_dim(embed_cache, args)
    d_in = feat_dim(args)
    print(
        f"  appearance_dim={args.appearance_dim} d_in={d_in} "
        f"(embed_load_only={bool(getattr(args, 'embed_load_only', False))})"
    )
    if args.max_patients > 0:
        all_rows = subsample_rows_by_patient(all_rows, args.max_patients, args.seed)
    print(f"  frames={len(all_rows)}")

    fold_results = []
    if args.cohort == "private":
        train_rows = [r for r in all_rows if r.split == "train"]
        val_rows_manifest = [r for r in all_rows if r.split == "val"]
        test_rows = [r for r in all_rows if r.split == "test"]
        fit_rows, holdout = split_val_patients(
            train_rows, args.val_patient_ratio, args.seed, lambda r: r.patient_id,
        )
        val_rows = val_rows_manifest if val_rows_manifest else holdout
        proto = getattr(args, "train_protocol", "private")
        n_tr_pos = len({r.patient_id for r in train_rows if r.label == 1})
        n_tr_neg = len({r.patient_id for r in train_rows if r.label == 0})
        n_te_pos = len({r.patient_id for r in test_rows if r.label == 1})
        n_te_neg = len({r.patient_id for r in test_rows if r.label == 0})
        print(
            f"\n=== Private protocol={proto} "
            f"fit={len(fit_rows)} val={len(val_rows)} test={len(test_rows)} ==="
        )
        print(
            f"  train patients: norm={n_tr_neg} abn={n_tr_pos} | "
            f"test patients: norm={n_te_neg} abn={n_te_pos}"
        )
        _print_disease_split_audit(train_rows, test_rows)
        fold_results.append(run_eval_fold(
            fold=0, fit_rows=fit_rows, val_rows=val_rows, test_rows=test_rows,
            embed_cache=embed_cache, args=args, device=device,
        ))
    else:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]
        for fold in folds:
            print(f"\n=== Fold {fold} ===")
            fold_rows = [r for r in all_rows if r.fold == fold]
            train_rows = [r for r in fold_rows if r.split == "train"]
            test_rows = [r for r in fold_rows if r.split == "test"]
            fit_rows, val_rows = split_val_patients(
                train_rows, args.val_patient_ratio, args.seed + fold, lambda r: r.patient_id,
            )
            fold_results.append(run_eval_fold(
                fold=fold, fit_rows=fit_rows, val_rows=val_rows, test_rows=test_rows,
                embed_cache=embed_cache, args=args, device=device,
            ))

    mean_f1_c2b = float(np.mean([r["f1_c2b_max"] for r in fold_results]))
    mean_f1_mean = float(np.mean([r["f1_c2b_mean"] for r in fold_results]))
    mean_f1_vt = float(np.mean([r["f1_fusion"] for r in fold_results]))
    mean_c2b = float(np.mean([r["auc_c2b_max"] for r in fold_results]))
    mean_mean = float(np.mean([r["auc_c2b_mean"] for r in fold_results]))
    mean_vt = float(np.mean([r["auc_fusion"] for r in fold_results]))
    summary_modes = {}
    mode_names = _modes_present_in_folds(fold_results, "fusion")
    # Prefer union with c2b / bypass keys so Sens@90%Spec appears if any head has it
    for who in ("c2b_max", "c2b_mean", "fusion", "fusion_lr_bypass", "fusion_lr_bypass_sens90", "fusion_lr_bypass_forced"):
        for m in _modes_present_in_folds(fold_results, who):
            if m not in mode_names:
                mode_names.append(m)
    for who, prefix in (
        ("c2b_max", "C2-b max"),
        ("c2b_mean", "C2-b mean"),
        ("fusion", "fusion"),
        ("fusion_lr_bypass", "fusion_lr_bypass"),
        ("fusion_lr_bypass_sens90", "fusion_lr_bypass_sens90"),
        ("fusion_lr_bypass_forced", "fusion_lr_bypass_forced"),
    ):
        summary_modes[who] = {}
        for mode in mode_names:
            # Skip if this head never produced the mode (avoid nan-only clutter)
            if not any(mode in (r.get(who) or {}) for r in fold_results):
                continue
            summary_modes[who][mode] = {
                "f1": _mean_mode_metric(fold_results, who, mode, "f1"),
                "sensitivity": _mean_mode_metric(fold_results, who, mode, "sensitivity"),
                "specificity": _mean_mode_metric(fold_results, who, mode, "specificity"),
                "auc": _mean_mode_metric(fold_results, who, mode, "auc"),
                "ppv": _mean_mode_metric(fold_results, who, mode, "ppv"),
            }
    taus = [
        float(r.get("lr_gate_tau", r.get("lr_bypass_alpha")))
        for r in fold_results
        if r.get("lr_gate_tau") is not None or r.get("lr_bypass_alpha") is not None
    ]
    mean_tau = float(np.mean(taus)) if taus else None
    mean_alpha = mean_tau  # legacy alias
    mean_f1_bypass = float(np.mean([r["f1_fusion_lr_bypass"] for r in fold_results if "f1_fusion_lr_bypass" in r])) if any(
        "f1_fusion_lr_bypass" in r for r in fold_results
    ) else None
    mean_auc_bypass = float(np.mean([r["auc_fusion_lr_bypass"] for r in fold_results if "auc_fusion_lr_bypass" in r])) if any(
        "auc_fusion_lr_bypass" in r for r in fold_results
    ) else None
    stem = _result_stem(args)
    out = {
        "cohort": args.cohort,
        "method": args.fusion,
        "graph_adj": str(getattr(args, "_resolved_graph_adj", None)
                         or getattr(args, "graph_adj", "auto") or "auto"),
        "graph_adj_perm": getattr(args, "_resolved_graph_perm", None),
        "feat_model": args.feat_model,
        "m0_features": args.m0_features,
        "m0_feature_dim": n_m0_dims(args),
        "within_view_pool": args.within_view_pool,
        "fallback": args.fallback,
        "clip_norm": getattr(args, "clip_norm", "hybrid"),
        "norm_view_drop": bool(getattr(args, "norm_view_drop", False)),
        "norm_view_drop_scale": float(getattr(args, "norm_view_drop_scale", 1.0) or 1.0),
        "extra_feats": getattr(args, "extra_feats", "none"),
        "lora_adapter": str(args.lora_adapter) if args.lora_adapter else None,
        "seed": int(getattr(args, "seed", 42) or 42),
        "repro": _build_repro_block(args),
        "train_protocol": getattr(args, "train_protocol", "private"),
        "frames_per_view_norm": int(getattr(args, "frames_per_view_norm", 0) or 0),
        "resolved_frames_per_view_norm": int(
            getattr(args, "_resolved_frames_per_view_norm", 0) or 0
        ),
        "balance_norm_to_eval": bool(getattr(args, "balance_norm_to_eval", True)),
        "mil_max_frames": args.mil_max_frames,
        "n_views": n_views_for(args),
        "d_in": d_in,
        "max_patients": args.max_patients,
        "primary_metric": "patient_f1",
        "threshold_modes": list(mode_names),
        "feat_backend": args.feat_backend,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "folds": fold_results,
        "summary_by_mode": summary_modes,
        "mean_f1_c2b_max": mean_f1_c2b,
        "mean_f1_c2b_mean": mean_f1_mean,
        "mean_f1_fusion": mean_f1_vt,
        "mean_f1_fusion_lr_bypass": mean_f1_bypass,
        "delta_f1_vs_c2b_max": mean_f1_vt - mean_f1_c2b,
        "mean_auc_c2b_max": mean_c2b,
        "mean_auc_c2b_mean": mean_mean,
        "mean_auc_fusion": mean_vt,
        "mean_auc_fusion_lr_bypass": mean_auc_bypass,
        "delta_auc_vs_c2b_max": mean_vt - mean_c2b,
        "mean_lr_gate_tau": mean_tau,
        "mean_lr_bypass_alpha": mean_alpha,  # legacy (= mean τ)
    }
    card_ext = [r.get("cardium_external") for r in fold_results if r.get("cardium_external")]
    if card_ext:
        out["mean_auc_cardium_external"] = float(np.mean([c["auc"] for c in card_ext if c.get("auc") == c.get("auc")]))
        out["mean_f1_cardium_external"] = float(np.mean([c["f1"] for c in card_ext if c.get("f1") == c.get("f1")]))
        out["cardium_external"] = card_ext[0]
    zy_ext = [r.get("zy_external") for r in fold_results if r.get("zy_external")]
    if zy_ext:
        out["mean_auc_zy_external"] = float(np.mean([c["auc"] for c in zy_ext if c.get("auc") == c.get("auc")]))
        out["mean_f1_zy_external"] = float(np.mean([c["f1"] for c in zy_ext if c.get("f1") == c.get("f1")]))
        out["zy_external"] = zy_ext[0]
    if getattr(args, "train_protocol", None) == "real_joint":
        out["train_protocol"] = "real_joint"
        out["joint_cardium_fold"] = int(getattr(args, "joint_cardium_fold", 1) or 1)
        out["dann"] = bool(getattr(args, "dann", False))
        out["mvp"] = (
            not bool(getattr(args, "no_mvp", False))
            and (bool(getattr(args, "mvp", False)) or _is_graph_fusion(args.fusion))
        )
    if fold_results and "stratified_fusion" in fold_results[0]:
        sf0 = fold_results[0]["stratified_fusion"]
        out["summary_stratified_fusion"] = {
            mode: {
                stratum: sf0[stratum][mode]
                for stratum in sf0
                if mode in sf0[stratum]
            }
            for mode in ("val_f1_tuned", "fixed_0.5", "youden", "sens_at_spec_0.90")
        }
        out["summary_stratified_fusion"] = {
            mode: block for mode, block in out["summary_stratified_fusion"].items() if block
        }
        out["primary_ood_screening_auc"] = sf0["screening_ood"]["val_f1_tuned"].get("auc")
        out["primary_new_norm_spec_fixed_0_5"] = sf0["norm_new_device"]["fixed_0.5"].get(
            "specificity"
        )
        s90 = fold_results[0].get("fusion", {}).get("sens_at_spec_0.90", {})
        out["primary_sens_at_spec_0_90"] = s90.get("sensitivity")
        out["primary_spec_at_sens_op"] = s90.get("specificity")
        b90 = fold_results[0].get("fusion_lr_bypass", {}).get("sens_at_spec_0.90", {})
        out["primary_bypass_sens_at_spec_0_90"] = b90.get("sensitivity")
        out["primary_bypass_spec_at_sens_op"] = b90.get("specificity")
        out["primary_lr_gate_tau"] = fold_results[0].get(
            "lr_gate_tau", fold_results[0].get("lr_bypass_alpha")
        )
        out["primary_lr_bypass_alpha"] = out["primary_lr_gate_tau"]  # legacy
    elif fold_results:
        # Still expose primary Sens@90%Spec when stratified block is absent
        s90 = fold_results[0].get("fusion", {}).get("sens_at_spec_0.90", {})
        out["primary_sens_at_spec_0_90"] = s90.get("sensitivity")
        out["primary_spec_at_sens_op"] = s90.get("specificity")
        b90 = fold_results[0].get("fusion_lr_bypass", {}).get("sens_at_spec_0.90", {})
        out["primary_bypass_sens_at_spec_0_90"] = b90.get("sensitivity")
        out["primary_bypass_spec_at_sens_op"] = b90.get("specificity")
        out["primary_lr_gate_tau"] = fold_results[0].get(
            "lr_gate_tau", fold_results[0].get("lr_bypass_alpha")
        )
        out["primary_lr_bypass_alpha"] = out["primary_lr_gate_tau"]
    missing_s90 = [
        f"fold{r.get('fold')}"
        for r in fold_results
        if "sens_at_spec_0.90" not in (r.get("fusion") or {})
    ]
    if missing_s90:
        raise RuntimeError(
            f"JSON refuse: sens_at_spec_0.90 missing in {missing_s90}. "
            "Server code is stale — sync experiments/masvf_view_token_fusion.py "
            "and experiments/chd_baseline/metrics.py, then re-run."
        )
    if fold_results and fold_results[0].get("lr_coef_summary"):
        out["lr_coef_summary"] = fold_results[0]["lr_coef_summary"]
        coef_path = args.output_dir / f"lr_coef_summary_{stem}.json"
        coef_path.write_text(
            json.dumps(out["lr_coef_summary"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote LR coef summary → {coef_path}")
    path = args.output_dir / f"view_token_results_{stem}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # legacy alias for default Phase-A-like config
    if (
        args.fusion == "feature_transformer"
        and args.feat_model == "m1"
        and args.m0_features == "all"
        and args.within_view_pool == "nanmax"
        and args.fallback == "drop"
    ):
        (args.output_dir / "view_token_results_feature_transformer.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.fusion == "score_mlp":
        (args.output_dir / "view_token_results.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print("\n========== SUMMARY (primary = patient F1) ==========")
    for mode in mode_names:
        s = summary_modes
        if mode not in s.get("fusion", {}):
            continue
        if mode == "sens_at_spec_0.90":
            cm = s.get("c2b_max", {}).get(mode, {})
            fm = s["fusion"][mode]
            dsens = None
            if cm.get("sensitivity") is not None and fm.get("sensitivity") is not None:
                dsens = float(fm["sensitivity"]) - float(cm["sensitivity"])
            print(
                f"[{mode}] "
                f"C2-b max Sens={_fmt_metric(cm.get('sensitivity'), '.4f')} "
                f"Spec={_fmt_metric(cm.get('specificity'), '.4f')} | "
                f"{args.fusion} Sens={_fmt_metric(fm.get('sensitivity'), '.4f')} "
                f"Spec={_fmt_metric(fm.get('specificity'), '.4f')} "
                f"(ΔSens vs max={_fmt_metric(dsens, '+.4f')})"
            )
        else:
            cm = s.get("c2b_max", {}).get(mode, {})
            mn = s.get("c2b_mean", {}).get(mode, {})
            fm = s["fusion"][mode]
            df1 = None
            if cm.get("f1") is not None and fm.get("f1") is not None:
                df1 = float(fm["f1"]) - float(cm["f1"])
            print(
                f"[{mode}] "
                f"C2-b max F1={_fmt_metric(cm.get('f1'), '.4f')} "
                f"mean F1={_fmt_metric(mn.get('f1'), '.4f')} "
                f"{args.fusion} F1={_fmt_metric(fm.get('f1'), '.4f')} "
                f"(Δ vs max={_fmt_metric(df1, '+.4f')})"
            )
    dauc = None
    try:
        dauc = float(mean_vt) - float(mean_c2b)
    except (TypeError, ValueError):
        dauc = None
    print(
        f"val-F1-tuned AUC: C2-b max={_fmt_metric(mean_c2b, '.4f')}  "
        f"{args.fusion}={_fmt_metric(mean_vt, '.4f')}  "
        f"(Δ={_fmt_metric(dauc, '+.4f')})"
    )
    bp = summary_modes.get("fusion_lr_bypass") or {}
    if bp.get("val_f1_tuned"):
        print(
            f"lr_gate τ={_fmt_metric(mean_tau, '.3g')}: "
            f"F1={_fmt_metric(bp['val_f1_tuned'].get('f1'), '.4f')} "
            f"AUC={_fmt_metric(bp['val_f1_tuned'].get('auc'), '.4f')} | "
            f"Sens@90%Spec={_fmt_metric((bp.get('sens_at_spec_0.90') or {}).get('sensitivity'), '.4f')} "
            f"Spec={_fmt_metric((bp.get('sens_at_spec_0.90') or {}).get('specificity'), '.4f')}"
        )
    bpf = summary_modes.get("fusion_lr_bypass_forced") or {}
    if bpf.get("val_f1_tuned"):
        print(
            f"lr_gate forced τ={LR_GATE_FORCED_TAU:g}: "
            f"F1={_fmt_metric(bpf['val_f1_tuned'].get('f1'), '.4f')} "
            f"AUC={_fmt_metric(bpf['val_f1_tuned'].get('auc'), '.4f')} | "
            f"Sens@90%Spec={_fmt_metric((bpf.get('sens_at_spec_0.90') or {}).get('sensitivity'), '.4f')} "
            f"Spec={_fmt_metric((bpf.get('sens_at_spec_0.90') or {}).get('specificity'), '.4f')}"
        )
    if fold_results and "stratified_fusion" in fold_results[0] and args.cohort == "private":
        sf = fold_results[0]["stratified_fusion"]
        ood = (sf.get("screening_ood") or {}).get("val_f1_tuned") or {}
        nd = (sf.get("norm_new_device") or {}).get("fixed_0.5") or {}
        print(
            f"OOD screening AUC (new-norm vs hom-abn)={_fmt_metric(ood.get('auc'), '.4f')} | "
            f"new-norm Spec@0.5={_fmt_metric(nd.get('specificity'), '.4f')} "
            f"(n_norm={nd.get('n_patients', 0) or 0})"
        )
    print(f"Wrote {path}")
    print(
        "Target: fusion ≥ C2-b max on F1/AUC; "
        "clinical OP: Sens @ Spec≥0.90 (val-calibrated)"
    )


if __name__ == "__main__":
    main()
