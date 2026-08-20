#!/usr/bin/env python3
"""Drop-k F1 summary figure from published Table missing_main numbers."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "fig_dropk_robustness.pdf"

# keep-4 / drop-1 / drop-2  (mean, std) from tab:missing_main
alvg = np.array([[0.8473, 0.0184], [0.8403, 0.0150], [0.8224, 0.0154]])
gnn = np.array([[0.8437, 0.0190], [0.8323, 0.0228], [0.8193, 0.0185]])
ft = np.array([[0.8369, 0.0177], [0.8120, 0.0185], [0.7541, 0.0204]])
x = np.array([0, 1, 2])

fig, ax = plt.subplots(figsize=(3.4, 2.45), dpi=300)
for arr, name, color, mk in (
    (alvg, "ALVG", "#1F4E79", "o"),
    (gnn, "GNN", "#2A9D8F", "s"),
    (ft, "Transformer", "#C45C26", "^"),
):
    ax.errorbar(
        x, arr[:, 0], yerr=arr[:, 1],
        color=color, marker=mk, ms=5, lw=1.4, capsize=2.5, label=name,
    )
ax.set_xticks(x)
ax.set_xticklabels(["Keep-4", "Drop-1", "Drop-2"])
ax.set_ylabel("F1")
ax.set_ylim(0.72, 0.88)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.tick_params(length=3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=300)
print("Wrote", OUT)
