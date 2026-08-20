#!/usr/bin/env python3
"""R4-M4: ECE / reliability for full-frame vs hybrid on tertiary test.

CPU-only. Reuses the same score paths as eval_tertiary_unified_testset.py.

  cd /path/to/fetal-chd-gated-triage
  python -u experiments/eval_tertiary_hybrid_ece.py \\
    --unified-fusion-root outputs/tertiary_20241125_vs_chd_unified_eval \\
    --fusion-run attention_mil_seed42 \\
    --out-dir outputs/tertiary_r4_calibration

Outputs:
  ece_summary.json
  fig_reliability_ece.pdf / .png  (copy PDF → docs/paper_tertiary_overleaf/figures/)
  LATEX_SNIPPET.txt             (paste into appendix tab:calibration)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from eval_tertiary_unified_testset import (  # noqa: E402
    find_fusion_json,
    hybrid_scores,
    load_fullframe_scores,
    load_fusion_unified,
    load_manifest_test,
)

INK = "#1B1B1B"
MUTED = "#555555"
CURVE = "#1F3A5F"
DIAG = "#8A8A8A"
PANEL_FACE = "#FFFFFF"


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid vs full-frame ECE")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument(
        "--audit-json",
        type=Path,
        default=ROOT / "outputs/tertiary_unified_audit/coverage.json",
    )
    p.add_argument(
        "--fullframe-root",
        type=Path,
        default=ROOT / "outputs/tertiary_20241125_vs_chd_patient_full_raw_seeds",
    )
    p.add_argument("--fullframe-seed", type=int, default=42)
    p.add_argument(
        "--unified-fusion-root",
        type=Path,
        default=ROOT / "outputs/tertiary_20241125_vs_chd_unified_eval",
    )
    p.add_argument(
        "--fusion-run",
        default="attention_mil_seed42",
        help="Run dir under unified-fusion-root used for hybrid gated branch",
    )
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_r4_calibration",
    )
    return p.parse_args()


def expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, dict]:
    """Equal-width ECE on [0,1] probability scores."""
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            rows.append({"lo": lo, "hi": hi, "n": 0, "conf": None, "acc": None})
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        w = float(mask.sum()) / n
        ece += w * abs(acc - conf)
        rows.append({"lo": lo, "hi": hi, "n": int(mask.sum()), "conf": conf, "acc": acc})
    return float(ece), {"bins": rows, "n_bins": n_bins}


def reliability_diagram(
    ax,
    y,
    p,
    title: str,
    n_bins: int = 15,
    *,
    panel: str | None = None,
) -> float:
    """Nature/IEEE-style reliability diagram; marker area ∝ bin count."""
    ece, detail = expected_calibration_error(y, p, n_bins=n_bins)
    xs, ys, ns = [], [], []
    for b in detail["bins"]:
        if b["n"] and b["conf"] is not None:
            xs.append(b["conf"])
            ys.append(b["acc"])
            ns.append(b["n"])
    ax.set_facecolor(PANEL_FACE)
    ax.plot([0, 1], [0, 1], "--", color=DIAG, lw=1.05, zorder=1, label="Perfect")
    if xs:
        ns_arr = np.asarray(ns, dtype=float)
        sizes = 28 + 110 * (ns_arr / ns_arr.max())
        ax.plot(xs, ys, color=CURVE, lw=1.35, alpha=0.85, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=CURVE,
            edgecolors="white",
            linewidths=0.55,
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Mean predicted probability", fontsize=8.5, color=INK)
    ax.set_ylabel("Empirical positive rate", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7.5, colors=INK)
    ax.set_title(title, fontsize=9.0, color=INK, pad=8)
    ax.text(
        0.04,
        0.96,
        f"ECE = {ece:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#D0D0D0",
            linewidth=0.55,
        ),
        zorder=4,
    )
    if panel is not None:
        ax.text(
            -0.12,
            1.05,
            panel,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="right",
            color=INK,
        )
    ax.set_aspect("equal", adjustable="box")
    return ece


def scores_to_arrays(
    score_map: dict[str, tuple[int, float]],
    pids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    keys = pids if pids is not None else sorted(score_map)
    y, s = [], []
    for pid in keys:
        if pid not in score_map:
            continue
        lab, sc = score_map[pid]
        y.append(lab)
        s.append(sc)
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest_test(args.manifest)
    ff_path = args.fullframe_root / f"seed{args.fullframe_seed}" / "results.json"
    if not ff_path.is_file():
        print(f"ERROR: missing fullframe {ff_path}")
        return 1
    fullframe = load_fullframe_scores(ff_path)

    gated_pids: set[str] = set()
    if args.audit_json.is_file():
        audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
        gated_pids = {
            pid
            for pid, c in (audit.get("patients") or {}).items()
            if c.get("fusion_gated")
        }
    else:
        print(f"WARN: missing audit {args.audit_json}; hybrid falls back more often")

    run_dir = args.unified_fusion_root / args.fusion_run
    jpath = find_fusion_json(run_dir)
    if jpath is None:
        print(f"ERROR: no fusion JSON in {run_dir}")
        return 1
    fusion = load_fusion_unified(jpath)
    if not fusion:
        print(f"ERROR: empty fusion scores in {jpath}")
        return 1

    hybrid = hybrid_scores(manifest, fusion, fullframe, gated_pids)

    # Full-frame: embed-present only (matches paper n_pos=90 block)
    y_ff, p_ff = scores_to_arrays(fullframe)
    # Hybrid: all manifest patients that received a score
    y_hy, p_hy = scores_to_arrays(hybrid, pids=sorted(manifest))

    ece_ff, _ = expected_calibration_error(y_ff, p_ff, n_bins=args.n_bins)
    ece_hy, _ = expected_calibration_error(y_hy, p_hy, n_bins=args.n_bins)
    brier_ff = float(brier_score_loss(y_ff, np.clip(p_ff, 1e-6, 1 - 1e-6)))
    brier_hy = float(brier_score_loss(y_hy, np.clip(p_hy, 1e-6, 1 - 1e-6)))
    auc_ff = float(roc_auc_score(y_ff, p_ff)) if len(set(y_ff.tolist())) > 1 else float("nan")
    auc_hy = float(roc_auc_score(y_hy, p_hy)) if len(set(y_hy.tolist())) > 1 else float("nan")

    summary = {
        "n_bins": args.n_bins,
        "fusion_run": args.fusion_run,
        "fusion_json": str(jpath),
        "fullframe": {
            "n": int(len(y_ff)),
            "n_pos": int((y_ff == 1).sum()),
            "n_neg": int((y_ff == 0).sum()),
            "ece": ece_ff,
            "brier": brier_ff,
            "auc": auc_ff,
        },
        "hybrid": {
            "n": int(len(y_hy)),
            "n_pos": int((y_hy == 1).sum()),
            "n_neg": int((y_hy == 0).sum()),
            "ece": ece_hy,
            "brier": brier_hy,
            "auc": auc_hy,
            "n_gated_pids": int(len(gated_pids)),
        },
    }
    (args.out_dir / "ece_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), dpi=300)
    reliability_diagram(
        axes[0],
        y_ff,
        p_ff,
        f"Full-frame (embed-present, n={len(y_ff)})",
        n_bins=args.n_bins,
        panel="a",
    )
    reliability_diagram(
        axes[1],
        y_hy,
        p_hy,
        f"Hybrid deploy (unified, n={len(y_hy)})",
        n_bins=args.n_bins,
        panel="b",
    )
    fig.suptitle(
        "Reliability: full-frame vs hybrid deployment scores",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    pdf = args.out_dir / "fig_reliability_ece.pdf"
    png = args.out_dir / "fig_reliability_ece.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.12, dpi=300)
    plt.close(fig)

    latex = f"""Full-frame patient-mean (embed-present) & {summary['fullframe']['n']} & {ece_ff:.4f} & {brier_ff:.4f} \\\\
Hybrid gated$\\vee$full-frame (unified) & {summary['hybrid']['n']} & {ece_hy:.4f} & {brier_hy:.4f} \\\\
"""
    (args.out_dir / "LATEX_SNIPPET.txt").write_text(latex, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(f"Paste LATEX_SNIPPET.txt into appendix tab:calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
