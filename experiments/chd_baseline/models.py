"""Model builders for CHD baseline experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torchvision.models as tv_models

from .lora import inject_lora, lora_parameters


def _register_fetalclip(fetalclip_config: str) -> None:
    with open(fetalclip_config, encoding="utf-8") as f:
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = json.load(f)


def load_fetalclip_visual(
    fetalclip_config: str,
    fetalclip_weights: str,
    fetalclip_dir: str,
) -> tuple[nn.Module, object]:
    if fetalclip_dir not in sys.path:
        sys.path.insert(0, fetalclip_dir)
    _register_fetalclip(fetalclip_config)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=fetalclip_weights,
    )
    return model.visual, preprocess


def load_fetalclip_full(
    fetalclip_config: str,
    fetalclip_weights: str,
    fetalclip_dir: str,
) -> tuple[nn.Module, object, object]:
    if fetalclip_dir not in sys.path:
        sys.path.insert(0, fetalclip_dir)
    _register_fetalclip(fetalclip_config)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP", pretrained=fetalclip_weights,
    )
    tokenizer = open_clip.get_tokenizer("FetalCLIP")
    return model, preprocess, tokenizer


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return self.head(feats)


class FetalCLIPClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, feat_dim: int = 768) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)


def _load_state_dict_compatible(model: nn.Module, state: dict) -> None:
    """Load checkpoint keys whose tensor shapes match the model (skip e.g. ImageNet head)."""
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k in own and own[k].shape != state[k].shape]
    if skipped:
        print(f"Skipped {len(skipped)} mismatched keys (e.g. classifier head): {skipped[:3]}...")
    model.load_state_dict(compatible, strict=False)


def build_cnn(
    name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    weights_path: str | Path | None = None,
) -> nn.Module:
    """Build CNN classifier. Prefer local weights_path on offline servers."""
    name = name.lower()

    def _load_backbone(backbone: nn.Module) -> nn.Module:
        if pretrained:
            if weights_path is None:
                raise FileNotFoundError(
                    f"No local weights for {name}. Set models.{name}.weights in config "
                    f"or torchvision_weights_dir (offline server cannot download)."
                )
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing CNN weights: {path}")
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            backbone.load_state_dict(state)
        return backbone

    if name == "resnet50":
        backbone = _load_backbone(tv_models.resnet50(weights=None))
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return CNNClassifier(backbone, feat_dim, num_classes)
    if name == "resnet101":
        backbone = _load_backbone(tv_models.resnet101(weights=None))
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return CNNClassifier(backbone, feat_dim, num_classes)
    if name == "efficientnet_b0":
        backbone = _load_backbone(tv_models.efficientnet_b0(weights=None))
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return CNNClassifier(backbone, feat_dim, num_classes)
    if name == "densenet121":
        backbone = _load_backbone(tv_models.densenet121(weights=None))
        feat_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        return CNNClassifier(backbone, feat_dim, num_classes)
    if name == "convnext_large":
        backbone = _load_backbone(tv_models.convnext_large(weights=None))
        feat_dim = backbone.classifier[2].in_features
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        return CNNClassifier(backbone, feat_dim, num_classes)
    if name == "swin_large":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "swin_large requires timm: pip install timm"
            ) from exc
        model = timm.create_model(
            "swin_large_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )
        if pretrained:
            path = Path(weights_path) if weights_path else None
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"Missing Swin-L weights: {path}. "
                    "Run: python scripts/export_swin_large_timm.py"
                )
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(path, map_location="cpu")
            _load_state_dict_compatible(model, state)
        return model, cnn_transform(224)
    if name == "vit_b_16":
        backbone = _load_backbone(tv_models.vit_b_16(weights=None))
        feat_dim = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()
        return CNNClassifier(backbone, feat_dim, num_classes)
    raise ValueError(f"Unknown CNN: {name}")


def build_fetalclip_linear(
    num_classes: int,
    fetalclip_config: str,
    fetalclip_weights: str,
    fetalclip_dir: str,
    freeze_encoder: bool = True,
) -> tuple[FetalCLIPClassifier, object]:
    encoder, preprocess = load_fetalclip_visual(fetalclip_config, fetalclip_weights, fetalclip_dir)
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
    model = FetalCLIPClassifier(encoder, num_classes)
    return model, preprocess


def build_fetalclip_lora(
    num_classes: int,
    fetalclip_config: str,
    fetalclip_weights: str,
    fetalclip_dir: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target: str,
) -> tuple[FetalCLIPClassifier, object, list[str]]:
    encoder, preprocess = load_fetalclip_visual(fetalclip_config, fetalclip_weights, fetalclip_dir)
    for p in encoder.parameters():
        p.requires_grad = False
    replaced = inject_lora(encoder, target_pattern=lora_target, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = FetalCLIPClassifier(encoder, num_classes)
    return model, preprocess, replaced


def get_trainable_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    *,
    text_lr_scale: float = 0.3,
) -> list[dict]:
    """Separate LoRA / head / definition-text params from CNN backbone."""
    head_params = []
    text_params = []
    lora_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "text_features" in name:
            text_params.append(p)
        elif "lora_a" in name or "lora_b" in name:
            lora_params.append(p)
        elif "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr, "weight_decay": weight_decay})
    if text_params:
        groups.append({
            "params": text_params,
            "lr": lr * text_lr_scale,
            "weight_decay": 0.0,
        })
    extra = head_params + lora_params
    if extra:
        groups.append({"params": extra, "lr": lr * 10, "weight_decay": 0.0})
    return groups


def cnn_transform(image_size: int = 224):
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
