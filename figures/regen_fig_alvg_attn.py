#!/usr/bin/env python3
"""Regenerate fig_alvg_attn.pdf from the saved subsample JSON (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
VIEW_LABELS = ("4CH", "LVOT", "RVOT", "3VT")
ADJ = np.array(
    [
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ],
    dtype=float,
)


def main() -> None:
    meta = json.loads((HERE / "fig_alvg_attn_sample.json").read_text())
    wmat = np.full((4, 4), np.nan)
    for i, qi in enumerate(VIEW_LABELS):
        row = meta["weights_mean"][qi]
        for j, kj in enumerate(VIEW_LABELS):
            v = row[kj]
            if v is not None:
                wmat[i, j] = float(v)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), dpi=300)
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("adj_blue", ["#F7F9FB", "#9BB7D4", "#1F4E79"])
    im = ax.imshow(ADJ, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(VIEW_LABELS, fontsize=10)
    ax.set_yticklabels(VIEW_LABELS, fontsize=10)
    ax.tick_params(length=0)
    ax.set_xlabel("Target view", fontsize=9)
    ax.set_ylabel("Source view", fontsize=9)
    ax.set_title("Anatomy adjacency\n(shared YOLO structures)", fontsize=10, pad=6)
    for i in range(4):
        for j in range(4):
            val = int(ADJ[i, j] > 0.5)
            ax.text(j, i, str(val), ha="center", va="center", fontsize=11,
                    color="white" if val else "#333333", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=7)
    ax.text(-0.18, 1.08, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

    ax = axes[1]
    nq = meta.get("n_per_query") or {}
    n_inc = int(meta.get("n_incomplete") or 0)
    colors = {"4CH": "#1F4E79", "LVOT": "#2A9D8F", "RVOT": "#C45C26", "3VT": "#E9C46A"}
    # RVOT is the only missing-query with n>1 and ≥2 present neighbors.
    # LVOT/3VT each occur in one bag where the other anatomical neighbor is
    # also absent, so softmax over a single key is identically 1 — omit those.
    qi = VIEW_LABELS.index("RVOT")
    keys, vals = [], []
    for j, kname in enumerate(VIEW_LABELS):
        v = wmat[qi, j]
        if np.isfinite(v) and float(v) > 0:
            keys.append(kname)
            vals.append(float(v))
    x = np.arange(len(keys), dtype=float)
    ax.bar(x, vals, color=[colors[k] for k in keys], edgecolor="white",
           linewidth=0.4, width=0.62)
    for xi, yi in zip(x, vals):
        ax.text(xi, yi + 0.025, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Mean attention", fontsize=9)
    ax.set_xlabel("Present neighbor (query = missing RVOT)", fontsize=9)
    ax.set_title(
        f"Imputation when RVOT is missing\n({int(nq.get('RVOT', 0))} of {n_inc} incomplete bags; not a saliency map)",
        fontsize=10, pad=6,
    )
    ax.tick_params(length=0)
    ax.text(0.5, -0.28,
            "LVOT/3VT missing: 1 bag each, only 4CH present → attention = 1 (omitted).",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#444444")
    ax.text(-0.12, 1.08, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, wspace=0.32)
    out = HERE / "fig_alvg_attn.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12, dpi=300)
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
