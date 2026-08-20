#!/usr/bin/env python3
"""M2 DCA + M3 gated ECE from existing score dumps. CPU only. Does not invent curves.

DCA uses full-frame patient-mean probabilities (continuous). Gated ECE uses
patient_scores.fusion.p_test from ALVG / MIL fusion JSONs.

  python -u experiments/eval_tertiary_dca_and_gated_ece.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from eval_tertiary_hybrid_ece import (  # noqa: E402
    expected_calibration_error,
    reliability_diagram,
)
from eval_tertiary_unified_testset import (  # noqa: E402
    find_fusion_json,
    load_fullframe_scores,
)

INK = "#1B1B1B"
MUTED = "#555555"
CURVE = "#1F3A5F"
ALVG_C = "#1F3A5F"
MIL_C = "#8B3A3A"
TREAT = "#8A8A8A"
PANEL_FACE = "#FFFFFF"


def _rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCA + gated ECE from dumps")
    p.add_argument(
        "--fullframe-json",
        type=Path,
        default=ROOT
        / "outputs/tertiary_20241125_vs_chd_patient_full_raw_seeds/seed42/results.json",
    )
    p.add_argument(
        "--alvg-root",
        type=Path,
        default=ROOT / "outputs/tertiary_20241125_vs_chd_fusion_raw",
    )
    p.add_argument(
        "--mil-root",
        type=Path,
        default=ROOT / "outputs/tertiary_feat_ablation",
    )
    p.add_argument("--alvg-glob", default="anatomy_graph_seed*")
    p.add_argument("--mil-glob", default="attention_mil_seed*__r-mil_clip_anat")
    p.add_argument("--n-bins", type=int, default=15)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_bspc_final_probes",
    )
    return p.parse_args()


def net_benefit_observed(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    if pt <= 0.0 or pt >= 1.0:
        return float("nan")
    pred = p >= pt
    n = len(y)
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    return tp / n - fp / n * (pt / (1.0 - pt))


def net_benefit_pi(y: np.ndarray, p: np.ndarray, pt: float, pi: float) -> float:
    if pt <= 0.0 or pt >= 1.0:
        return float("nan")
    pred = p >= pt
    pos = y == 1
    neg = y == 0
    sens = float(pred[pos].mean()) if pos.any() else 0.0
    spec = float((~pred)[neg].mean()) if neg.any() else 1.0
    return pi * sens - (1.0 - pi) * (1.0 - spec) * (pt / (1.0 - pt))


def treat_all_nb(pi: float, pt: float) -> float:
    if pt <= 0.0 or pt >= 1.0:
        return float("nan")
    return pi - (1.0 - pi) * (pt / (1.0 - pt))


def arrays_from_fullframe(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores = load_fullframe_scores(path)
    y = np.asarray([v[0] for v in scores.values()], dtype=np.int64)
    p = np.asarray([v[1] for v in scores.values()], dtype=np.float64)
    return y, p


def load_gated_block(jpath: Path) -> dict | None:
    data = json.loads(jpath.read_text(encoding="utf-8"))
    fold = (data.get("folds") or [{}])[0]
    block = (fold.get("patient_scores") or {}).get("fusion") or {}
    y = block.get("y_test")
    p = block.get("p_test")
    if not y or not p:
        return None
    meta = {
        "path": str(jpath),
        "method": data.get("method"),
        "extra_feats": data.get("extra_feats"),
        "m0_features": data.get("m0_features"),
        "seed": data.get("seed"),
        "y": np.asarray(y, dtype=np.int64),
        "p": np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6),
    }
    return meta


def collect_runs(root: Path, glob_pat: str) -> list[dict]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.glob(glob_pat)):
        if not d.is_dir():
            continue
        jpath = find_fusion_json(d)
        if jpath is None:
            continue
        block = load_gated_block(jpath)
        if block is None:
            print(f"  WARN: no p_test in {jpath}")
            continue
        out.append(block)
    return out


def ece_summary(runs: list[dict], n_bins: int) -> dict:
    eces = []
    briers = []
    aucs = []
    rows = []
    for r in runs:
        ece, _ = expected_calibration_error(r["y"], r["p"], n_bins=n_bins)
        brier = float(brier_score_loss(r["y"], r["p"]))
        auc = float(roc_auc_score(r["y"], r["p"])) if len(set(r["y"].tolist())) > 1 else float("nan")
        eces.append(ece)
        briers.append(brier)
        aucs.append(auc)
        rows.append(
            {
                "seed": r["seed"],
                "n": int(len(r["y"])),
                "n_pos": int((r["y"] == 1).sum()),
                "ece": ece,
                "brier": brier,
                "auc": auc,
                "path": r["path"],
            }
        )
    def _ms(xs: list[float]) -> dict:
        xs = [x for x in xs if x == x]
        if not xs:
            return {"mean": None, "std": None, "n": 0}
        if len(xs) == 1:
            return {"mean": xs[0], "std": 0.0, "n": 1}
        return {"mean": float(statistics.mean(xs)), "std": float(statistics.stdev(xs)), "n": len(xs)}

    return {"seeds": rows, "ece": _ms(eces), "brier": _ms(briers), "auc": _ms(aucs)}


def dca_table(y: np.ndarray, p: np.ndarray, pts: np.ndarray) -> dict:
    pi_obs = float(y.mean())
    obs = [net_benefit_observed(y, p, float(pt)) for pt in pts]
    pi001 = [net_benefit_pi(y, p, float(pt), 0.01) for pt in pts]
    all_obs = [treat_all_nb(pi_obs, float(pt)) for pt in pts]
    all_001 = [treat_all_nb(0.01, float(pt)) for pt in pts]
    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "pi_observed": pi_obs,
        "thresholds": [float(x) for x in pts],
        "nb_model_observed": obs,
        "nb_treat_all_observed": all_obs,
        "nb_treat_none": [0.0] * len(pts),
        "nb_model_pi001": pi001,
        "nb_treat_all_pi001": all_001,
    }


def plot_dca(dca_ff: dict, dca_alvg: dict | None, out_pdf: Path) -> None:
    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), dpi=200)
    pts = np.asarray(dca_ff["thresholds"])
    panels = [
        (
            axes[0],
            "a",
            "Referral-enriched test",
            dca_ff["nb_model_observed"],
            dca_ff["nb_treat_all_observed"],
            None if dca_alvg is None else dca_alvg["nb_model_observed"],
            f"$\\pi$={dca_ff['pi_observed']:.3f}",
        ),
        (
            axes[1],
            "b",
            "Prevalence-adjusted $\\pi$=0.01",
            dca_ff["nb_model_pi001"],
            dca_ff["nb_treat_all_pi001"],
            None if dca_alvg is None else dca_alvg["nb_model_pi001"],
            "$\\pi$=0.01",
        ),
    ]
    for ax, lab, title, nb_ff, nb_all, nb_alvg, sub in panels:
        ax.set_facecolor(PANEL_FACE)
        ax.axhline(0.0, color=TREAT, lw=0.9, ls=":")
        ax.plot(pts, nb_all, color=TREAT, lw=1.1, ls="--", label="Treat all")
        ax.plot(pts, nb_ff, color=CURVE, lw=1.6, label="Full-frame")
        if nb_alvg is not None:
            ax.plot(pts, nb_alvg, color=MIL_C, lw=1.4, label="Gated ALVG")
        ax.set_xlim(0.01, 0.30)
        ys = list(nb_ff)
        if nb_alvg is not None:
            ys += list(nb_alvg)
        ymax = max(0.02, float(np.nanmax(ys)) * 1.20)
        ax.set_ylim(-0.02, ymax)
        ax.set_xlabel("Threshold probability $p_t$", fontsize=8.5, color=INK)
        ax.set_ylabel("Net benefit", fontsize=8.5, color=INK)
        ax.set_title(f"{title}\n{sub}", fontsize=9.0, color=INK, pad=6)
        ax.tick_params(labelsize=7.5, colors=INK)
        ax.text(
            -0.12,
            1.08,
            lab,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="right",
            color=INK,
        )
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def plot_reliability(alvg: dict | None, mil: dict | None, n_bins: int, out_pdf: Path) -> None:
    _rc()
    n_panel = int(alvg is not None) + int(mil is not None)
    if n_panel == 0:
        return
    fig, axes = plt.subplots(1, max(n_panel, 1), figsize=(3.4 * max(n_panel, 1), 3.2), dpi=200)
    if n_panel == 1:
        axes = [axes]
    i = 0
    if alvg is not None:
        reliability_diagram(
            axes[i],
            alvg["y"],
            alvg["p"],
            title=f"Gated ALVG (seed {alvg['seed']})",
            n_bins=n_bins,
        )
        axes[i].text(
            -0.12, 1.05, "a", transform=axes[i].transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right", color=INK,
        )
        i += 1
    if mil is not None:
        reliability_diagram(
            axes[i],
            mil["y"],
            mil["p"],
            title=f"Gated MIL (seed {mil['seed']})",
            n_bins=n_bins,
        )
        axes[i].text(
            -0.12, 1.05, "b" if alvg is not None else "a",
            transform=axes[i].transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right", color=INK,
        )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def fmt_ms(d: dict) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    if d["n"] <= 1:
        return f"{d['mean']:.4f}"
    return f"{d['mean']:.4f} $\\pm$ {d['std']:.4f} (n={d['n']})"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    missing = []

    if not args.fullframe_json.is_file():
        print(f"ERROR: missing full-frame scores {args.fullframe_json}")
        missing.append("fullframe")
        y_ff = p_ff = None
    else:
        y_ff, p_ff = arrays_from_fullframe(args.fullframe_json)
        print(
            f"full-frame n={len(y_ff)} n_pos={int((y_ff == 1).sum())} "
            f"pi={float(y_ff.mean()):.4f}"
        )

    alvg_runs = collect_runs(args.alvg_root, args.alvg_glob)
    mil_runs = collect_runs(args.mil_root, args.mil_glob)
    if not mil_runs:
        fallback = ROOT / "outputs/tertiary_20241125_vs_chd_fusion_raw"
        mil_runs = collect_runs(fallback, "attention_mil_seed*")
        if mil_runs:
            print("WARN: MIL from fusion_raw attention_mil_* (check m0/extra vs paper recipe)")
    print(f"gated dumps ALVG={len(alvg_runs)} MIL={len(mil_runs)}")
    if not alvg_runs:
        missing.append("alvg")
    if not mil_runs:
        missing.append("mil")

    pts = np.linspace(0.01, 0.50, 50)
    dca_ff = dca_alvg = None
    if y_ff is not None:
        dca_ff = dca_table(y_ff, p_ff, pts)
        ece_ff, _ = expected_calibration_error(y_ff, np.clip(p_ff, 1e-6, 1 - 1e-6), n_bins=args.n_bins)
        dca_ff["ece"] = ece_ff
        dca_ff["auc"] = float(roc_auc_score(y_ff, p_ff))
    if alvg_runs:
        r0 = alvg_runs[0]
        dca_alvg = dca_table(r0["y"], r0["p"], pts)
        dca_alvg["seed"] = r0["seed"]

    alvg_sum = ece_summary(alvg_runs, args.n_bins) if alvg_runs else None
    mil_sum = ece_summary(mil_runs, args.n_bins) if mil_runs else None

    summary = {
        "missing": missing,
        "fullframe": dca_ff,
        "gated_alvg_ece": alvg_sum,
        "gated_mil_ece": mil_sum,
        "n_bins": args.n_bins,
    }
    # Drop huge threshold vectors from a slim headline file
    slim = {
        "missing": missing,
        "fullframe": None
        if dca_ff is None
        else {
            "n": dca_ff["n"],
            "n_pos": dca_ff["n_pos"],
            "pi_observed": dca_ff["pi_observed"],
            "auc": dca_ff.get("auc"),
            "ece": dca_ff.get("ece"),
            "nb_at_pt_0.01_obs": dca_ff["nb_model_observed"][0],
            "nb_at_pt_0.10_obs": dca_ff["nb_model_observed"][int(np.argmin(np.abs(pts - 0.10)))],
            "nb_at_pt_0.01_pi001": dca_ff["nb_model_pi001"][0],
        },
        "gated_alvg_ece": alvg_sum,
        "gated_mil_ece": mil_sum,
    }
    (args.out_dir / "dca_ece_summary.json").write_text(
        json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "dca_full.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else o) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"% M2 DCA + M3 gated ECE",
        f"full-frame ECE={slim['fullframe']['ece']:.4f}" if slim["fullframe"] else "full-frame missing",
        f"gated ALVG ECE {fmt_ms((alvg_sum or {}).get('ece') or {})}",
        f"gated MIL ECE {fmt_ms((mil_sum or {}).get('ece') or {})}",
        f"missing={missing}",
    ]
    (args.out_dir / "LATEX_SNIPPET.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if dca_ff is not None:
        plot_dca(dca_ff, dca_alvg, args.out_dir / "fig_dca.pdf")
    plot_reliability(
        alvg_runs[0] if alvg_runs else None,
        mil_runs[0] if mil_runs else None,
        args.n_bins,
        args.out_dir / "fig_gated_reliability.pdf",
    )

    if missing:
        print("INCOMPLETE: retrain gated dumps then re-run this script")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
