#!/usr/bin/env python3
"""Compare full-era vs main (≥2020) patient-mean probes for Appendix A1.

  python -u experiments/summarize_full_era_vs_main.py \\
    --main-root outputs/tertiary_full_era_probe/main_era2020 \\
    --full-root outputs/tertiary_full_era_probe/full_era \\
    --latex --out-dir outputs/tertiary_full_era_probe
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-root", type=Path, required=True)
    p.add_argument("--full-root", type=Path, required=True)
    p.add_argument("--main-manifest", type=Path, default=None)
    p.add_argument("--full-manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "tertiary_full_era_probe")
    p.add_argument("--latex", action="store_true")
    return p.parse_args()


def _mean_std(xs: list[float]) -> tuple[float | None, float | None]:
    xs = [float(x) for x in xs if x is not None and x == x]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _fmt(m: float | None, s: float | None = None) -> str:
    if m is None:
        return "—"
    if s is None or s == 0.0:
        return f"{m:.4f}"
    return f"{m:.4f}±{s:.4f}"


def _tex(m: float | None, s: float | None = None) -> str:
    if m is None:
        return "---"
    if s is None or s == 0.0:
        return f"{m:.4f}"
    return f"{m:.4f}\\pm{s:.4f}"


def load_probe(root: Path) -> dict:
    rows = []
    for p in sorted(root.glob("seed*/results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("test_patient_mean") or {}
        rows.append(
            {
                "seed": p.parent.name.replace("seed", ""),
                "auc": m.get("auc"),
                "f1": m.get("f1"),
                "ppv": m.get("ppv"),
                "sens": m.get("sensitivity"),
                "spec": m.get("specificity"),
                "n_pos": m.get("n_pos"),
                "n_neg": m.get("n_neg"),
            }
        )
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "auc": _mean_std([r["auc"] for r in rows]),
        "f1": _mean_std([r["f1"] for r in rows]),
        "ppv": _mean_std([r["ppv"] for r in rows]),
        "sens": _mean_std([r["sens"] for r in rows]),
        "spec": _mean_std([r["spec"] for r in rows]),
        "n_pos": rows[0].get("n_pos"),
        "n_neg": rows[0].get("n_neg"),
        "rows": rows,
    }


def manifest_counts(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    by = Counter()
    n_chd = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split", "?")
        lab = int(row.get("label_binary", -1))
        by[(lab, split)] += 1
        if lab == 1:
            n_chd[split] += 1
    return {
        "chd_train": n_chd.get("train", 0),
        "chd_val": n_chd.get("val", 0),
        "chd_test": n_chd.get("test", 0),
        "chd_total": sum(n_chd.values()),
        "norm_total": sum(v for (lab, _), v in by.items() if lab == 0),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_m = load_probe(args.main_root)
    full_m = load_probe(args.full_root)
    main_c = manifest_counts(args.main_manifest)
    full_c = manifest_counts(args.full_manifest)

    lines = [
        "# Full-era vs main (≥2020) patient-mean",
        "",
        "| setting | #CHD (man.) | test n_pos/n_neg | F1 | AUC |",
        "|---|---:|---:|---:|---:|",
        "| Main CHD ≥2020 | {ct} | {np}/{nn} | {f1} | {auc} |".format(
            ct=main_c.get("chd_total", "—"),
            np=main_m.get("n_pos", "—"),
            nn=main_m.get("n_neg", "—"),
            f1=_fmt(*(main_m.get("f1") or (None, None))),
            auc=_fmt(*(main_m.get("auc") or (None, None))),
        ),
        "| Full-era CHD | {ct} | {np}/{nn} | {f1} | {auc} |".format(
            ct=full_c.get("chd_total", "—"),
            np=full_m.get("n_pos", "—"),
            nn=full_m.get("n_neg", "—"),
            f1=_fmt(*(full_m.get("f1") or (None, None))),
            auc=_fmt(*(full_m.get("auc") or (None, None))),
        ),
        "",
        "Inflated full-era metrics are interpreted as era/device co-linearity, not anatomy gains.",
        "",
    ]
    md = args.out_dir / "FULL_ERA.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "main": {k: v for k, v in main_m.items() if k != "rows"},
        "full": {k: v for k, v in full_m.items() if k != "rows"},
        "main_manifest_counts": main_c,
        "full_manifest_counts": full_c,
    }
    # jsonify tuples
    def fix(obj):
        if isinstance(obj, dict):
            return {k: fix(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return {"mean": obj[0], "std": obj[1]}
        return obj

    (args.out_dir / "full_era_summary.json").write_text(
        json.dumps(fix(payload), indent=2), encoding="utf-8"
    )
    print(md.read_text(encoding="utf-8"))

    if args.latex:
        tex = f"""% Auto-generated by summarize_full_era_vs_main.py
\\begin{{table}}[!t]
\\centering
\\caption{{Full-era CHD vs.\\ main cohort (same tertiary normals; patient-mean LR on raw FetalCLIP).
Inflated full-era performance is interpreted as era/device confounding.}}
\\label{{tab:a1_fullera}}
\\begin{{tabular}}{{lrrll}}
\\toprule
Setting & \\#CHD patients & train/val/test & Patient AUC & F1 \\\\
\\midrule
Main: CHD $\\ge$2020 & {main_c.get('chd_total', '---')} & {main_c.get('chd_train', '---')}/{main_c.get('chd_val', '---')}/{main_c.get('chd_test', '---')} & ${_tex(*(main_m.get('auc') or (None, None)))}$ & ${_tex(*(main_m.get('f1') or (None, None)))}$ \\\\
Full-era CHD & {full_c.get('chd_total', '---')} & {full_c.get('chd_train', '---')}/{full_c.get('chd_val', '---')}/{full_c.get('chd_test', '---')} & ${_tex(*(full_m.get('auc') or (None, None)))}$ & ${_tex(*(full_m.get('f1') or (None, None)))}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
        (args.out_dir / "full_era_table.tex").write_text(tex, encoding="utf-8")
        print(f"Wrote {args.out_dir / 'full_era_table.tex'}")

    if main_m.get("n", 0) == 0 and full_m.get("n", 0) == 0:
        print("WARNING: no results.json found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
