#!/usr/bin/env python3
"""ALVG figure: anatomy-adjacency prior + mean imputation over incomplete bags.

Paper figure is two panels (appendix cannot hold many images):
  (a) fixed YOLO-structure adjacency
  (b) mean softmax over all incomplete test bags (row = missing query)

Sync to the cloud box (same repo root as tertiary fusion):
  experiments/viz_alvg_attention.py
  experiments/run_viz_alvg_real_sample.sh

  cd /path/to/fetal-chd-gated-triage
  CUDA_VISIBLE_DEVICES=0 bash experiments/run_viz_alvg_real_sample.sh

Headline ALVG runs did not save best.pt; if no checkpoint is found we retrain
a *sampled* ALVG (--max-patients 80, --epochs 12) only for visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Cloud GPU boxes have no display.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "data" / "study_screening"
BSPC_FIG = ROOT / "docs" / "paper_bspc_overleaf" / "figures"

VIEW_LABELS = ("4CH", "LVOT", "RVOT", "3VT")
PRESENT = "#1B7F4A"
MISSING = "#B42318"
EDGE = "#1F4E79"
INK = "#1B1B1B"

DEFAULT_CKPT_GLOBS = (
    "outputs/tertiary_alvg_attn_sample/alvg_sample.pt",
    "outputs/tertiary_feat_ablation/anatomy_graph_seed42*/**/best.pt",
    "outputs/tertiary_feat_ablation/anatomy_graph_seed42*/**/*.pt",
    "outputs/tertiary_20241125_vs_chd_fusion_raw/anatomy_graph_seed42/**/*.pt",
    "outputs/tertiary_20241125_vs_chd_fusion_raw/anatomy_graph_seed42/*.pt",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ALVG anatomy attention")
    p.add_argument(
        "--out",
        type=Path,
        default=BSPC_FIG / "fig_alvg_attn.pdf",
    )
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument(
        "--real",
        action="store_true",
        help="Load real tokens and plot mean imputation over incomplete bags",
    )
    p.add_argument(
        "--paper-case",
        action="store_true",
        help="Local schematic only (no mean heatmap); prefer --real on the GPU box",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-patients", type=int, default=80)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--manifest",
        type=Path,
        default=SCREEN / "manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument(
        "--image-tags",
        type=Path,
        default=SCREEN / "yolo_image_tags_tertiary_20241125.jsonl",
    )
    p.add_argument(
        "--feature-cache",
        type=Path,
        default=SCREEN / "masvf_m0_tertiary_20241125_era2020_feature_cache.jsonl",
    )
    p.add_argument(
        "--fetalclip-cache",
        type=Path,
        default=SCREEN / "tertiary_20241125_vs_chd_raw_base_embeddings.jsonl",
    )
    return p.parse_args()


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def anatomy_adj() -> np.ndarray:
    sys.path.insert(0, str(ROOT / "experiments"))
    from view_graph_fusion import build_anatomy_adjacency

    return build_anatomy_adjacency().astype(float)


def _find_ckpt(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    for pat in DEFAULT_CKPT_GLOBS:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    return None


def imputation_attn_matrix(model, tokens: torch.Tensor, present: torch.Tensor) -> np.ndarray:
    """[4,4] softmax weights: missing query i → present adjacent key j; else NaN."""
    wmat = np.full((4, 4), np.nan, dtype=np.float64)
    imp = model.imputers[0]
    with torch.no_grad():
        x = model._embed(tokens, present)
        pres_idx = present[0].nonzero(as_tuple=False).squeeze(-1)
        miss_idx = (~present[0]).nonzero(as_tuple=False).squeeze(-1)
        if miss_idx.numel() == 0 or pres_idx.numel() == 0:
            return wmat
        for mi in miss_idx.tolist():
            pres_use = imp._filter_present(
                torch.as_tensor([mi], device=tokens.device),
                pres_idx,
            )
            if pres_use.numel() == 0:
                continue
            q = x[:, mi : mi + 1]
            kv = x[:, pres_use]
            _, w = imp.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
            ww = w.reshape(-1).detach().cpu().numpy()
            for k, pi in enumerate(pres_use.tolist()):
                wmat[mi, pi] = float(ww[k])
    return wmat


def mean_imputation_matrix(
    model, tokens: np.ndarray, present: np.ndarray, device
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Mean softmax [4,4] over bags with a missing slot; per-query bag counts."""
    acc = np.zeros((4, 4), dtype=np.float64)
    cnt = np.zeros((4, 4), dtype=np.float64)
    n_miss = np.zeros(4, dtype=np.int64)
    n_used = 0
    for i in range(present.shape[0]):
        n_on = int(present[i].sum())
        if n_on >= 4 or n_on < 1:
            continue
        tok = torch.from_numpy(tokens[i : i + 1].astype(np.float32)).to(device)
        pres = torch.from_numpy(present[i : i + 1]).to(device)
        w = imputation_attn_matrix(model, tok, pres)
        if not np.isfinite(w).any():
            continue
        n_used += 1
        miss = present[i] < 0.5
        n_miss += miss.astype(np.int64)
        for r in range(4):
            for c in range(4):
                if np.isfinite(w[r, c]):
                    acc[r, c] += float(w[r, c])
                    cnt[r, c] += 1.0
    mean = np.full((4, 4), np.nan, dtype=np.float64)
    np.divide(acc, cnt, out=mean, where=cnt > 0)
    print(
        f"mean imputation over n_incomplete={n_used} bags; "
        f"n_missing_query={n_miss.tolist()}"
    )
    return mean, cnt, n_used, n_miss


def _fusion_argv(args: argparse.Namespace, epochs: int, patience: int, max_patients: int) -> list[str]:
    return [
        "masvf_view_token_fusion.py",
        "--cohort", "private",
        "--train-protocol", "private",
        "--fusion", "anatomy_graph",
        "--feat-model", "m1",
        "--m0-features", "none",
        "--extra-feats", "view_anat",
        "--within-view-pool", "frame_attn",
        "--fallback", "drop",
        "--clip-norm", "hybrid",
        "--embed-load-only",
        "--mvp",
        "--lambda-mvp", "0.1",
        "--no-view-decomp",
        "--manifest", str(args.manifest),
        "--image-tags", str(args.image_tags),
        "--feature-cache", str(args.feature_cache),
        "--fetalclip-cache", str(args.fetalclip_cache),
        "--data-root", str(ROOT / "data"),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--max-patients", str(max_patients),
        "--output-dir", str(ROOT / "outputs" / "tertiary_alvg_attn_sample"),
    ]


def load_or_train_real(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(ROOT / "experiments"))
    from masvf_view_token_fusion import (  # noqa: E402
        _build_rows_and_embed,
        _build_screening_ns,
        _device,
        _load_tags_and_cache,
        _rows_X,
        feat_dim,
        fit_lr,
        impute_features,
        load_feature_cache,
        make_fusion_model,
        n_m0_dims,
        parse_args as fusion_parse_args,
        patient_view_feat_tokens,
        split_val_patients,
        standardize_feature_blocks,
        subsample_rows_by_patient,
        train_fusion,
    )

    bak = sys.argv
    sys.argv = _fusion_argv(args, args.epochs, args.patience, args.max_patients)
    try:
        fargs = fusion_parse_args()
    finally:
        sys.argv = bak

    for pth, lab in (
        (fargs.manifest, "manifest"),
        (fargs.image_tags, "image-tags"),
        (fargs.feature_cache, "feature-cache"),
        (fargs.fetalclip_cache, "fetalclip-cache"),
    ):
        if pth is None or not Path(pth).is_file():
            raise FileNotFoundError(f"missing {lab}: {pth}")

    device = _device(str(fargs.device))
    tags_path, tags, feat = _load_tags_and_cache(fargs)
    cache = load_feature_cache(feat)
    ns = _build_screening_ns(fargs, tags_path, feat)
    # Tertiary M0 cache never covers every manifest frame_path. Official ALVG
    # training does not use --features-only; viz must not load YOLO either.
    ns.allow_missing_cache = True
    print(
        f"viz: skip YOLO; allow_missing_cache "
        f"(M0 cache keys={len(cache)})"
    )
    all_rows, embed_cache = _build_rows_and_embed(fargs, tags, cache, ns, yolo_model=None)
    if fargs.max_patients > 0:
        all_rows = subsample_rows_by_patient(all_rows, fargs.max_patients, fargs.seed)

    train_rows = [r for r in all_rows if r.split == "train"]
    val_rows_m = [r for r in all_rows if r.split == "val"]
    test_rows = [r for r in all_rows if r.split == "test"]
    fit_rows, holdout = split_val_patients(
        train_rows, fargs.val_patient_ratio, fargs.seed, lambda r: r.patient_id
    )
    val_rows = val_rows_m if val_rows_m else holdout
    if not test_rows:
        print("WARN: no test split in subsample; using val rows for the figure case")
        test_rows = list(val_rows)
    if not fit_rows or not val_rows or not test_rows:
        raise RuntimeError(
            f"empty split after subsample: n_fit={len(fit_rows)} "
            f"n_val={len(val_rows)} n_te={len(test_rows)}"
        )

    n_m0 = n_m0_dims(fargs)
    clip_norm = "hybrid"
    X_fit = _rows_X(fit_rows, fargs, embed_cache)
    y_fit = np.array([r.label for r in fit_rows], dtype=np.int64)
    clf, fill = fit_lr(X_fit, y_fit, n_m0=n_m0 if fargs.feat_model == "m1" else 0, clip_norm=clip_norm)
    X_fit_i, _ = impute_features(X_fit, fill)
    X_val_i, _ = impute_features(_rows_X(val_rows, fargs, embed_cache), fill)
    X_te_i, _ = impute_features(_rows_X(test_rows, fargs, embed_cache), fill)
    X_fit_i, X_val_i, X_te_i = standardize_feature_blocks(
        X_fit_i, X_val_i, X_te_i, n_m0=n_m0 if fargs.feat_model == "m1" else 0, clip_norm=clip_norm
    )
    views = ("four_chamber", "lvot", "rvot", "vvt")
    tr_tok, tr_pres, tr_y, _ = patient_view_feat_tokens(
        fit_rows, X_fit_i, pool="frame_attn", n_m0=n_m0, views=views
    )
    va_tok, va_pres, va_y, _ = patient_view_feat_tokens(
        val_rows, X_val_i, pool="frame_attn", n_m0=n_m0, views=views
    )
    te_tok, te_pres, te_y, te_pids = patient_view_feat_tokens(
        test_rows, X_te_i, pool="frame_attn", n_m0=n_m0, views=views
    )
    d_in = int(feat_dim(fargs))
    model = make_fusion_model(fargs, d_in=d_in).to(device)

    ckpt = _find_ckpt(args.ckpt)
    ckpt_kind = "none"
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            model.load_state_dict(state, strict=False)
            ckpt_kind = f"loaded:{ckpt}"
            print(f"loaded ckpt {ckpt}")
        except Exception as exc:
            print(f"WARN: ckpt load failed ({exc}); training sample ALVG")
            ckpt = None
    if ckpt is None:
        print(
            f"training sampled ALVG n_fit={len(tr_y)} n_val={len(va_y)} "
            f"n_te={len(te_y)} epochs={fargs.epochs}"
        )
        model = train_fusion(
            model, tr_tok, tr_pres, tr_y, va_tok, va_pres, va_y,
            device=device, epochs=int(fargs.epochs), patience=int(fargs.patience),
            batch_size=int(fargs.batch_size), lr=float(fargs.lr),
            weight_decay=float(fargs.weight_decay), pos_weight=float(fargs.pos_weight),
            seed=int(fargs.seed), use_mvp=True, lambda_mvp=0.1,
        )
        save_dir = ROOT / "outputs" / "tertiary_alvg_attn_sample"
        save_dir.mkdir(parents=True, exist_ok=True)
        pt = save_dir / "alvg_sample.pt"
        torch.save(model.state_dict(), pt)
        ckpt_kind = f"trained_sample:{pt}"
        print(f"wrote {pt}")

    model.eval()
    wmat, cnt, n_used, n_miss = mean_imputation_matrix(model, te_tok, te_pres, device)
    meta = {
        "mode": "mean_over_incomplete",
        "ckpt": ckpt_kind,
        "n_test_in_sample": int(len(te_y)),
        "n_incomplete": int(n_used),
        "n_per_query": {VIEW_LABELS[i]: int(n_miss[i]) for i in range(4)},
        "max_patients": int(fargs.max_patients),
        "weights_mean": {
            VIEW_LABELS[i]: {
                VIEW_LABELS[j]: (None if np.isnan(wmat[i, j]) else round(float(wmat[i, j]), 4))
                for j in range(4)
            }
            for i in range(4)
        },
    }
    print("SAMPLE", json.dumps(meta, ensure_ascii=False))
    return {"wmat": wmat, "present": None, "meta": meta, "counts": cnt}


def plot_adjacency(ax, adj: np.ndarray) -> None:
    cmap = LinearSegmentedColormap.from_list("adj_blue", ["#F7F9FB", "#9BB7D4", "#1F4E79"])
    im = ax.imshow(adj, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    ax.set_yticklabels(VIEW_LABELS, fontsize=8)
    ax.tick_params(length=0)
    ax.set_xlabel("Target view", fontsize=7.5)
    ax.set_ylabel("Source view", fontsize=7.5)
    ax.set_title("Anatomy adjacency\n(shared YOLO structures)", fontsize=8.5, pad=6)
    for i in range(4):
        for j in range(4):
            val = int(adj[i, j] > 0.5)
            ax.text(
                j, i, str(val), ha="center", va="center", fontsize=8.5,
                color="white" if val else "#333333", fontweight="bold",
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=7)


def plot_mean_heatmap(ax, wmat: np.ndarray, counts: np.ndarray | None, meta: dict) -> None:
    """Grouped bars: one cluster per missing query that actually occurs."""
    nq = meta.get("n_per_query") or {}
    queries = [i for i in range(4) if int(nq.get(VIEW_LABELS[i], 0) or 0) > 0]
    if not queries:
        queries = [i for i in range(4) if np.isfinite(wmat[i]).any()]
    colors = {"4CH": "#1F4E79", "LVOT": "#2A9D8F", "RVOT": "#C45C26", "3VT": "#E9C46A"}
    x = np.arange(len(queries), dtype=float)
    width = 0.18
    offsets = (np.arange(4) - 1.5) * width
    for j, off in zip(range(4), offsets):
        ys = []
        for qi in queries:
            v = wmat[qi, j]
            ys.append(0.0 if not np.isfinite(v) else float(v))
        ax.bar(
            x + off, ys, width=width * 0.92,
            color=colors[VIEW_LABELS[j]], edgecolor="white", linewidth=0.4,
            label=VIEW_LABELS[j],
        )
        for xi, yi in zip(x + off, ys):
            if yi > 0:
                ax.text(xi, yi + 0.03, f"{yi:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{VIEW_LABELS[i]}\n(n={int(nq.get(VIEW_LABELS[i], 0))})" for i in queries],
        fontsize=7.5,
    )
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Mean attention", fontsize=7.5)
    ax.set_xlabel("Missing query", fontsize=7.5)
    ax.set_title(
        "Imputation mass on present neighbors\n(incomplete bags only; not a saliency map)",
        fontsize=8.5,
        pad=6,
    )
    ax.legend(title="Key", fontsize=6.5, title_fontsize=6.5, frameon=False, ncol=4, loc="upper right")
    ax.tick_params(length=0)
    n_inc = meta.get("n_incomplete")
    ax.text(
        0.5, -0.28,
        f"n_incomplete={n_inc}; 4CH never missing in this subsample",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#444444",
    )


def plot_drop_rvot_schematic(ax, adj: np.ndarray) -> None:
    """Single-case fallback: missing RVOT → adjacent present keys."""
    present = np.array([True, True, False, True])
    xy = np.array([(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)], dtype=float)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.48, 1.38)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Sampled missing slot (RVOT)\nimputation edges, not a saliency map",
        fontsize=8.5,
        pad=6,
    )
    miss = ~present
    for mi in np.where(miss)[0]:
        for pi in np.where(present)[0]:
            if adj[mi, pi] > 0.5:
                x0, y0 = xy[mi]
                x1, y1 = xy[pi]
                dx, dy = x1 - x0, y1 - y0
                L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                pad = 0.14
                ax.add_patch(
                    FancyArrowPatch(
                        (x0 + pad * dx / L, y0 + pad * dy / L),
                        (x1 - pad * dx / L, y1 - pad * dy / L),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        color=EDGE,
                        lw=1.4,
                        connectionstyle="arc3,rad=0.08",
                        zorder=1,
                    )
                )
    for i in range(4):
        on = bool(present[i])
        ax.add_patch(
            Circle(
                xy[i], 0.13,
                facecolor=PRESENT if on else "white",
                edgecolor="#0E3D24" if on else MISSING,
                lw=1.0 if on else 1.8,
                zorder=3,
            )
        )
        ax.text(
            xy[i, 0], xy[i, 1] - 0.24, VIEW_LABELS[i],
            ha="center", va="top", fontsize=8, fontweight="bold",
        )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PRESENT,
                   markeredgecolor="#0E3D24", markersize=8, label="Present"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor=MISSING, markeredgewidth=1.8, markersize=8, label="Missing"),
            Line2D([0], [0], color=EDGE, lw=1.4, label="Missing → present adj."),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.08),
    )


def plot_paper(out_path: Path, real: dict | None) -> None:
    _rc()
    adj = anatomy_adj()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.55), dpi=300)
    plot_adjacency(axes[0], adj)
    axes[0].text(-0.18, 1.08, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
    if real is None or real.get("wmat") is None:
        plot_drop_rvot_schematic(axes[1], adj)
    else:
        plot_mean_heatmap(axes[1], real["wmat"], real.get("counts"), real["meta"])
    axes[1].text(-0.18, 1.08, "b", transform=axes[1].transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, out_path)
    if real is not None:
        meta_path = out_path.with_name(out_path.stem + "_sample.json")
        meta_path.write_text(json.dumps(real["meta"], indent=2), encoding="utf-8")
        print("Wrote", meta_path)


def _save(fig, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=300)
    plt.close(fig)
    print("Wrote", out_path)
    print("Wrote", out_path.with_suffix(".png"))
    # keep BSPC + TMI copies in sync when writing the architecture-named file
    extras = []
    if out_path.name == "fig_alvg_attn.pdf":
        extras = [
            ROOT / "docs" / "paper_tertiary_overleaf" / "figures" / out_path.name,
            BSPC_FIG / out_path.name,
        ]
    for dest in extras:
        if dest.resolve() == out_path.resolve():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(out_path.read_bytes())
            png = dest.with_suffix(".png")
            png.write_bytes(out_path.with_suffix(".png").read_bytes())
            print("Copied", dest)
        except OSError:
            pass


def main() -> int:
    args = parse_args()
    if args.paper_case:
        plot_paper(args.out, None)
        return 0
    if args.real:
        real = load_or_train_real(args)
        plot_paper(args.out, real)
        return 0
    plot_paper(args.out, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
