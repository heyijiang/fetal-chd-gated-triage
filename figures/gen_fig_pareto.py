#!/usr/bin/env python3
"""Performance vs encoder-frame budget (not wall-clock)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "fig_pareto_efficiency.pdf"

# x = encoded-frame fraction on the test-normal accounting split (17.4%).
# Hybrid mixes gated (17.4%) with 19 empty-view exams at 100%.
# 19/907 * 1 + 888/907 * 0.174 ≈ 0.191
pts = [
    (1.00, 0.887, "Full-frame\npatient-mean", "#6B6B6B", "s"),
    (0.191, 0.857, "Hybrid\n(907 exams)", "#2A9D8F", "D"),
    (0.174, 0.847, "Gated ALVG\n(888 exams)", "#1F4E79", "o"),
]

fig, ax = plt.subplots(figsize=(3.45, 2.55), dpi=300)
for x, y, lab, c, mk in pts:
    ax.scatter([x], [y], s=42, c=c, marker=mk, zorder=3, label=lab.replace("\n", " "))
    ax.annotate(lab, (x, y), textcoords="offset points",
                xytext=(8, -14) if x < 0.5 else (-52, 8), fontsize=7, color=c)
ax.set_xlabel("Encoded-frame fraction")
ax.set_ylabel("F1")
ax.set_xlim(0.05, 1.12)
ax.set_ylim(0.82, 0.905)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print("Wrote", OUT)
