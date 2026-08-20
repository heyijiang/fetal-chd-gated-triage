#!/usr/bin/env python3
"""Summarize Table 6 — missing-view robustness from view_decomp json.

Works with:
  - legacy view_decomp: leave_one_out + single_view
  - extended view_decomp: keep_k + progressive (after re-run with updated fusion code)

Usage:
  python -u experiments/summarize_view_missing_robustness.py \\
    --suite-root outputs/homologous_miccai_suite \\
    --out outputs/homologous_miccai_suite/TABLE6_view_missing.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("four_chamber", "lvot", "rvot", "vvt")


def _find_main(suite: Path, *, allow_nvdrop: bool = False) -> Path | None:
    cands = sorted(suite.glob("private_main_transformer_lora*/view_token_results_*.json"))
    # Exclude dedicated nvdrop tags and in-folder __nvdrop__ stems by default
    if not allow_nvdrop:
        cands = [
            p for p in cands
            if "_nvdrop" not in p.parent.name and "__nvdrop__" not in p.name
        ]
    if not cands:
        return None
    for p in cands:
        if "__seed42" in p.parent.name:
            return p
    for p in cands:
        if "__seed" not in p.parent.name:
            return p
    return cands[0]


def _mode_metrics(modes: dict | None) -> dict:
    modes = modes or {}
    vt = modes.get("val_f1_tuned") or {}
    s90 = modes.get("sens_at_spec_0.90") or {}
    return {
        "f1": vt.get("f1"),
        "auc": vt.get("auc"),
        "sens90": s90.get("sensitivity"),
        "ppv": vt.get("ppv"),
    }


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=ROOT / "outputs" / "homologous_miccai_suite")
    p.add_argument("--json", type=Path, default=None, help="Explicit view_token_results json")
    p.add_argument("--allow-nvdrop", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path = args.json or _find_main(args.suite_root, allow_nvdrop=args.allow_nvdrop)
    if path is None or not path.is_file():
        raise SystemExit(f"missing main json under {args.suite_root}")
    if "__nvdrop__" in path.name and not args.allow_nvdrop:
        raise SystemExit(f"refusing nvdrop json (pass --allow-nvdrop): {path}")
    print(f"Using: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    folds = data.get("folds") or []
    if not folds:
        raise SystemExit("no folds")
    vd = folds[0].get("view_decomp") or {}
    if not vd:
        raise SystemExit("no view_decomp in fold0 — re-run TRACKS=main with view_decomp enabled")

    fus = (folds[0].get("fusion") or {}).get("val_f1_tuned") or {}
    full_f1, full_auc = fus.get("f1"), fus.get("auc")

    lines = [
        "# Table 6 — Missing-view robustness (no retrain; mask present views)",
        "",
        f"Source: `{path}`",
        "",
        "## Progressive (mean over keep-k combos)",
        "",
        "| #views kept (k) | F1 mean | AUC mean | Sens@90% mean | note |",
        "|-----------------|---------|----------|---------------|------|",
    ]

    progressive = vd.get("progressive") or {}
    if progressive:
        for k in sorted(progressive.keys(), key=lambda s: -int(s)):
            blk = progressive[k]
            lines.append(
                f"| k={k} | {_fmt(blk.get('f1_mean'))} | {_fmt(blk.get('auc_mean'))} | "
                f"{_fmt(blk.get('sens90_mean'))} | n_combos={blk.get('n_combos')} |"
            )
    else:
        # Fallback from leave-one / single-view
        leave = vd.get("leave_one_out") or {}
        single = vd.get("single_view") or {}
        leave_auc = [(_mode_metrics(m).get("auc")) for m in leave.values()]
        leave_f1 = [(_mode_metrics(m).get("f1")) for m in leave.values()]
        leave_s = [(_mode_metrics(m).get("sens90")) for m in leave.values()]
        only_auc = [(_mode_metrics(m).get("auc")) for m in single.values()]
        only_f1 = [(_mode_metrics(m).get("f1")) for m in single.values()]
        only_s = [(_mode_metrics(m).get("sens90")) for m in single.values()]

        def _mean(xs):
            xs = [float(x) for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None

        lines.append(
            f"| k=4 (full) | {_fmt(full_f1)} | {_fmt(full_auc)} | — | suite main |"
        )
        lines.append(
            f"| k=3 (leave-one mean) | {_fmt(_mean(leave_f1))} | {_fmt(_mean(leave_auc))} | "
            f"{_fmt(_mean(leave_s))} | from leave_one_out |"
        )
        lines.append("| k=2 | — | — | — | need extended view_decomp (re-run main) |")
        lines.append(
            f"| k=1 (single-view mean) | {_fmt(_mean(only_f1))} | {_fmt(_mean(only_auc))} | "
            f"{_fmt(_mean(only_s))} | from single_view |"
        )

    lines += [
        "",
        "## Leave-one-out",
        "",
        "| Dropped view | F1 | AUC | Sens@90% | PPV |",
        "|--------------|----|-----|----------|-----|",
    ]
    for v in VIEWS:
        m = _mode_metrics((vd.get("leave_one_out") or {}).get(v))
        lines.append(
            f"| leave-out `{v}` | {_fmt(m['f1'])} | {_fmt(m['auc'])} | "
            f"{_fmt(m['sens90'])} | {_fmt(m['ppv'])} |"
        )

    lines += [
        "",
        "## Single-view only",
        "",
        "| Only view | F1 | AUC | Sens@90% | PPV |",
        "|-----------|----|-----|----------|-----|",
    ]
    for v in VIEWS:
        m = _mode_metrics((vd.get("single_view") or {}).get(v))
        lines.append(
            f"| only `{v}` | {_fmt(m['f1'])} | {_fmt(m['auc'])} | "
            f"{_fmt(m['sens90'])} | {_fmt(m['ppv'])} |"
        )

    keep2 = (vd.get("keep_k") or {}).get("2") or {}
    if keep2:
        lines += [
            "",
            "## Keep-2 combinations",
            "",
            "| Views | F1 | AUC | Sens@90% |",
            "|-------|----|-----|----------|",
        ]
        for name, modes in sorted(keep2.items()):
            m = _mode_metrics(modes)
            lines.append(
                f"| `{name}` | {_fmt(m['f1'])} | {_fmt(m['auc'])} | {_fmt(m['sens90'])} |"
            )

    lines += [
        "",
        "## How to refresh keep-2 / progressive",
        "",
        "```bash",
        "SEEDS=42 SKIP_TRAIN=1 TRACKS=main CUDA_VISIBLE_DEVICES=0 \\",
        "  bash experiments/run_homologous_view_missing_table.sh",
        "```",
        "",
    ]

    text = "\n".join(lines) + "\n"
    out = args.out or (args.suite_root / "TABLE6_view_missing.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
