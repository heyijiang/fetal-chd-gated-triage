#!/usr/bin/env python3
"""Summarize YOLO standard-view coverage for the tertiary--CHD manifest.

Outputs JSON + Markdown + optional LaTeX table for Methods / appendix.

  python -u experiments/summarize_yolo_view_coverage.py
  python -u experiments/summarize_yolo_view_coverage.py --split test --latex
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Tags store CARDIUM canonical names (four_chamber/lvot/rvot/vvt), not 4CH/3VT aliases.
STANDARD_VIEWS = ("four_chamber", "lvot", "rvot", "vvt")
VIEW_ALIASES = {
    "four_chamber": "four_chamber",
    "4ch": "four_chamber",
    "4c": "four_chamber",
    "4CH": "four_chamber",
    "lvot": "lvot",
    "LVOT": "lvot",
    "rvot": "rvot",
    "RVOT": "rvot",
    "vvt": "vvt",
    "3vt": "vvt",
    "3VT": "vvt",
    "3vv": "vvt",
}
VIEW_LABELS = {
    "four_chamber": "Four-chamber",
    "lvot": "LVOT",
    "rvot": "RVOT",
    "vvt": "3VT",
}


def normalize_view(view: str | None) -> str | None:
    if not view:
        return None
    return VIEW_ALIASES.get(str(view)) or VIEW_ALIASES.get(str(view).lower())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO view coverage summary")
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl",
    )
    p.add_argument(
        "--tags",
        type=Path,
        default=ROOT / "data/study_screening/yolo_image_tags_tertiary_20241125.jsonl",
    )
    p.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Restrict to one split or all patients",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/tertiary_yolo_coverage",
    )
    p.add_argument("--latex", action="store_true", help="Write coverage_table.tex")
    return p.parse_args()


def load_tag_views(path: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            if sid:
                out[str(sid)] = row.get("cardium_view")
    return out


def load_manifest(path: Path, split: str) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if split != "all" and s.get("split") != split:
                continue
            rows.append(s)
    return rows


def patient_view_sets(studies: list[dict], tag_views: dict[str, str | None]) -> list[dict]:
    out = []
    for st in studies:
        pid = str(st["patient_id"])
        lab = int(st.get("label_binary", 0))
        views_hit: set[str] = set()
        n_tagged = 0
        for rel in st.get("frame_paths") or []:
            sid = f"{st['study_id']}|{rel}"
            view = normalize_view(tag_views.get(sid))
            if view:
                n_tagged += 1
            if view in STANDARD_VIEWS:
                views_hit.add(view)
        out.append(
            {
                "patient_id": pid,
                "label": lab,
                "split": st.get("split"),
                "n_frames": len(st.get("frame_paths") or []),
                "n_tagged_frames": n_tagged,
                "n_std_views": len(views_hit),
                "std_views": sorted(views_hit),
            }
        )
    return out


def summarize(patients: list[dict]) -> dict:
    def bucket(lab: int) -> list[dict]:
        return [p for p in patients if p["label"] == lab]

    pos = bucket(1)
    neg = bucket(0)

    def hist(rows: list[dict]) -> dict[int, int]:
        c = Counter(p["n_std_views"] for p in rows)
        return {k: c.get(k, 0) for k in range(5)}

    def per_view_rate(rows: list[dict]) -> dict[str, float]:
        n = len(rows) or 1
        rates = {}
        for v in STANDARD_VIEWS:
            rates[v] = sum(1 for p in rows if v in p["std_views"]) / n
        return rates

    return {
        "n_patients": len(patients),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "view_count_hist_pos": hist(pos),
        "view_count_hist_neg": hist(neg),
        "view_count_hist_all": hist(patients),
        "per_view_patient_rate_pos": per_view_rate(pos),
        "per_view_patient_rate_neg": per_view_rate(neg),
        "per_view_patient_rate_all": per_view_rate(patients),
        "mean_std_views_pos": float(sum(p["n_std_views"] for p in pos) / max(len(pos), 1)),
        "mean_std_views_neg": float(sum(p["n_std_views"] for p in neg) / max(len(neg), 1)),
        "pct_gated_pos": sum(1 for p in pos if p["n_std_views"] > 0) / max(len(pos), 1),
        "pct_gated_neg": sum(1 for p in neg if p["n_std_views"] > 0) / max(len(neg), 1),
    }


def latex_table(summary: dict, split: str) -> str:
    hp = summary["view_count_hist_pos"]
    hn = summary["view_count_hist_neg"]
    rp = summary["per_view_patient_rate_pos"]
    rn = summary["per_view_patient_rate_neg"]
    lines = [
        "% Auto-generated by summarize_yolo_view_coverage.py",
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{YOLO standard-view coverage ({split} split). "
        "Patient-level: fraction with $\\ge$1 frame tagged to each view; "
        "histogram counts patients by number of distinct standard views detected.}}",
        "\\label{tab:yolo_coverage}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & CHD & Normal \\\\",
        "\\midrule",
        f"Patients & {summary['n_pos']} & {summary['n_neg']} \\\\",
        f"Gated ($\\ge$1 std view) & "
        f"{100 * summary['pct_gated_pos']:.1f}\\% & "
        f"{100 * summary['pct_gated_neg']:.1f}\\% \\\\",
        f"Mean distinct views & "
        f"{summary['mean_std_views_pos']:.2f} & "
        f"{summary['mean_std_views_neg']:.2f} \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Patients with view detected (\\%)}} \\\\",
    ]
    for v in STANDARD_VIEWS:
        lines.append(
            f"{VIEW_LABELS[v]} & {100 * rp[v]:.1f}\\% & {100 * rn[v]:.1f}\\% \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{3}{l}{\\textit{Histogram: \\# distinct std views per patient}} \\\\",
        ]
    )
    for k in range(5):
        lines.append(f"{k} views & {hp[k]} & {hn[k]} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def markdown_report(summary: dict, split: str) -> str:
    hp = summary["view_count_hist_pos"]
    hn = summary["view_count_hist_neg"]
    rp = summary["per_view_patient_rate_pos"]
    rn = summary["per_view_patient_rate_neg"]
    lines = [
        f"# YOLO view coverage ({split})",
        "",
        f"- Patients: **{summary['n_pos']} CHD + {summary['n_neg']} normal**",
        f"- Gated (≥1 std view): CHD **{100 * summary['pct_gated_pos']:.1f}%**, "
        f"normal **{100 * summary['pct_gated_neg']:.1f}%**",
        f"- Mean distinct views: CHD **{summary['mean_std_views_pos']:.2f}**, "
        f"normal **{summary['mean_std_views_neg']:.2f}**",
        "",
        "## Per-view patient detection rate",
        "",
        "| View | CHD | Normal |",
        "|---|---:|---:|",
    ]
    for v in STANDARD_VIEWS:
        lines.append(f"| {VIEW_LABELS[v]} | {100 * rp[v]:.1f}% | {100 * rn[v]:.1f}% |")
    lines.extend(
        [
            "",
            "## Histogram (# distinct std views / patient)",
            "",
            "| # views | CHD | Normal |",
            "|---:|---:|---:|",
        ]
    )
    for k in range(5):
        lines.append(f"| {k} | {hp[k]} | {hn[k]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag_views = load_tag_views(args.tags)
    studies = load_manifest(args.manifest, args.split)
    patients = patient_view_sets(studies, tag_views)
    summary = summarize(patients)

    payload = {
        "split": args.split,
        "paths": {"manifest": str(args.manifest), "tags": str(args.tags)},
        "summary": summary,
        "patients": patients,
    }
    (args.out_dir / f"coverage_{args.split}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / f"coverage_{args.split}.md").write_text(
        markdown_report(summary, args.split), encoding="utf-8"
    )
    if args.latex:
        tex = latex_table(summary, args.split)
        (args.out_dir / f"coverage_table_{args.split}.tex").write_text(tex, encoding="utf-8")
        print(f"Wrote {args.out_dir / f'coverage_table_{args.split}.tex'}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out_dir / f'coverage_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
