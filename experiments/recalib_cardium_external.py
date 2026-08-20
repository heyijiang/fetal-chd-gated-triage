#!/usr/bin/env python3
"""Post-hoc Platt + temperature recalibration on CARDIUM external patient_scores (R7).

Scans one or more roots for view_token_results_*.json that contain
cardium_external.patient_scores {y,p}. For each dump:

  - stratified 25% cal / 75% eval (seed from JSON or filename; RNG = seed+17
    matching masvf_view_token_fusion label_light)
  - fit temperature T by minimizing BCE of sigmoid(logit(p)/T) on cal
  - fit Platt: LogisticRegression on logit(p) -> y on cal
  - eval holdout: private-val-threshold baseline (fusion.val_f1_tuned if present),
    after-temp and after-platt with val-F1-tuned thr fit on cal
  - also report label_light F1 already stored in the dump (if any)

Writes summary.json + AGGREGATE.md + LATEX_SNIPPET.txt under --out.

  python -u experiments/recalib_cardium_external.py \\
    --roots outputs/tertiary_20241125_vs_chd_t6_cardium \\
            outputs/tertiary_r6_zeroshot_anat \\
    --out outputs/tertiary_r7_recalib_cardium
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_clinical_ops_tmi import (  # noqa: E402
    _to_logit,
    apply_temperature,
    fit_temperature,
)
from chd_baseline.metrics import (  # noqa: E402
    binary_metrics_at_threshold,
    find_best_binary_threshold,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more dirs to scan for view_token_results_*.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/tertiary_r7_recalib_cardium",
    )
    p.add_argument("--cal-frac", type=float, default=0.25)
    p.add_argument(
        "--name-contains",
        default="",
        help="Optional substring filter on parent dir name (e.g. m0-none)",
    )
    return p.parse_args()


def _seed_from_path(path: Path) -> int | None:
    text = f"{path.parent.name}/{path.name}"
    m = re.search(r"__seed-?(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[/_\-])seed[_-]?(\d+)", text, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def _resolve_seed(dump: dict, path: Path) -> int:
    for key in ("seed", "random_seed"):
        if dump.get(key) is not None:
            try:
                return int(dump[key])
            except (TypeError, ValueError):
                pass
    repro = dump.get("repro") or {}
    if isinstance(repro, dict) and repro.get("seed") is not None:
        try:
            return int(repro["seed"])
        except (TypeError, ValueError):
            pass
    s = _seed_from_path(path)
    return int(s) if s is not None else 0


def _load_cardium_block(dump: dict) -> dict | None:
    ce = dump.get("cardium_external")
    if isinstance(ce, dict) and isinstance(ce.get("patient_scores"), dict):
        ps = ce["patient_scores"]
        if "y" in ps and "p" in ps:
            return ce
    for fold in dump.get("folds") or []:
        ce = fold.get("cardium_external")
        if isinstance(ce, dict) and isinstance(ce.get("patient_scores"), dict):
            ps = ce["patient_scores"]
            if "y" in ps and "p" in ps:
                return ce
    return None


def _stratified_cal_idx(y: np.ndarray, cal_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Match masvf label_light: RandomState(seed+17), 25% stratified, leave ≥1 eval."""
    rng = np.random.RandomState(int(seed) + 17)
    cal, rest = [], []
    for lab in (0, 1):
        idx = np.where(y == lab)[0]
        rng.shuffle(idx)
        n_cal = max(1, int(round(len(idx) * cal_frac))) if len(idx) else 0
        if len(idx) > 1:
            n_cal = min(n_cal, len(idx) - 1)
        cal.extend(idx[:n_cal].tolist())
        rest.extend(idx[n_cal:].tolist())
    return np.asarray(cal, dtype=np.int64), np.asarray(rest, dtype=np.int64)


def _metrics_pack(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    m = binary_metrics_at_threshold(y, p, float(thr))
    return {
        "threshold": float(thr),
        "auc": m.get("auc"),
        "f1": m.get("f1"),
        "sensitivity": m.get("sensitivity"),
        "specificity": m.get("specificity"),
        "ppv": m.get("ppv"),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }


def _fit_platt(y_cal: np.ndarray, p_cal: np.ndarray) -> LogisticRegression:
    z = _to_logit(p_cal).reshape(-1, 1)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(z, y_cal.astype(np.int64))
    return clf


def _apply_platt(clf: LogisticRegression, p: np.ndarray) -> np.ndarray:
    z = _to_logit(p).reshape(-1, 1)
    return clf.predict_proba(z)[:, 1]


def _label_light_f1(ce: dict) -> float | None:
    ll = ce.get("label_light")
    if not isinstance(ll, dict):
        return None
    block = ll.get("val_f1_tuned") or ll.get("f1_tuned") or {}
    if isinstance(block, dict) and block.get("f1") is not None:
        return float(block["f1"])
    return None


def analyze_one(path: Path, cal_frac: float) -> dict:
    dump = json.loads(path.read_text(encoding="utf-8"))
    ce = _load_cardium_block(dump)
    out: dict = {"path": str(path), "root_hint": str(path.parent)}
    if ce is None:
        out["error"] = "missing cardium_external.patient_scores"
        return out

    y = np.asarray(ce["patient_scores"]["y"], dtype=np.int64).ravel()
    p = np.asarray(ce["patient_scores"]["p"], dtype=np.float64).ravel()
    if len(y) != len(p) or len(y) < 4:
        out["error"] = f"bad scores len y={len(y)} p={len(p)}"
        return out
    if len(np.unique(y)) < 2:
        out["error"] = "need both classes in patient_scores"
        return out

    seed = _resolve_seed(dump, path)
    cal_i, te_i = _stratified_cal_idx(y, cal_frac, seed)
    if len(cal_i) == 0 or len(te_i) == 0:
        out["error"] = "empty cal/eval split"
        return out

    y_cal, p_cal = y[cal_i], p[cal_i]
    y_te, p_te = y[te_i], p[te_i]

    # Baseline: private-val threshold transferred (from dump), applied on eval
    fus = ce.get("fusion") if isinstance(ce.get("fusion"), dict) else {}
    transferred = fus.get("val_f1_tuned") if isinstance(fus, dict) else None
    baseline = None
    if isinstance(transferred, dict) and transferred.get("threshold") is not None:
        thr0 = float(transferred["threshold"])
        baseline = _metrics_pack(y_te, p_te, thr0)
        baseline["source"] = "cardium_external.fusion.val_f1_tuned"
        baseline["dump_f1"] = transferred.get("f1")
        baseline["dump_auc"] = transferred.get("auc") or ce.get("auc")
    else:
        # fall back: thr=0.5 on raw probs
        baseline = _metrics_pack(y_te, p_te, 0.5)
        baseline["source"] = "fixed_0.5_fallback"

    T = fit_temperature(y_cal, p_cal)
    p_cal_t = apply_temperature(p_cal, T)
    p_te_t = apply_temperature(p_te, T)
    thr_t, _ = find_best_binary_threshold(y_cal, p_cal_t)
    after_temp = _metrics_pack(y_te, p_te_t, thr_t)
    after_temp["T"] = float(T)

    clf = _fit_platt(y_cal, p_cal)
    p_cal_p = _apply_platt(clf, p_cal)
    p_te_p = _apply_platt(clf, p_te)
    thr_p, _ = find_best_binary_threshold(y_cal, p_cal_p)
    after_platt = _metrics_pack(y_te, p_te_p, thr_p)
    after_platt["platt_coef"] = float(clf.coef_.ravel()[0])
    after_platt["platt_intercept"] = float(clf.intercept_.ravel()[0])

    ll_f1 = _label_light_f1(ce)
    out.update({
        "seed": seed,
        "cal_frac": cal_frac,
        "n_patients": int(len(y)),
        "n_cal": int(len(cal_i)),
        "n_eval": int(len(te_i)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "baseline_transfer": baseline,
        "after_temp": after_temp,
        "after_platt": after_platt,
        "label_light_f1_from_dump": ll_f1,
        "cardium_auc_reported": ce.get("auc"),
    })
    return out


def discover(roots: list[Path], name_contains: str = "") -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    needle = (name_contains or "").strip()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("view_token_results_*.json")):
            if needle and needle not in f"{p.parent.name}/{p.name}":
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if _load_cardium_block(d) is None:
                continue
            seen.add(key)
            found.append(p)
    return found


def _mean_std(xs: list[float]) -> dict | None:
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": mu, "std": 0.0, "n": 1}
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": mu, "std": math.sqrt(var), "n": len(xs)}


def _fmt(agg: dict | None) -> str:
    if not agg:
        return "—"
    if agg["n"] == 1:
        return f"{agg['mean']:.4f}"
    return f"{agg['mean']:.4f}±{agg['std']:.4f}"


def main() -> int:
    args = parse_args()
    paths = discover(args.roots, name_contains=getattr(args, "name_contains", "") or "")
    if not paths:
        print("ERROR: no view_token_results_*.json with cardium patient_scores found")
        print(f"  roots={args.roots}")
        return 1

    results = [analyze_one(p, args.cal_frac) for p in paths]
    ok = [r for r in results if "error" not in r]
    err = [r for r in results if "error" in r]

    def collect(key_path: list[str]) -> list[float]:
        vals = []
        for r in ok:
            cur: object = r
            for k in key_path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(k)
            if cur is not None:
                try:
                    vals.append(float(cur))
                except (TypeError, ValueError):
                    pass
        return vals

    aggregate = {
        "baseline_f1": _mean_std(collect(["baseline_transfer", "f1"])),
        "baseline_auc": _mean_std(collect(["baseline_transfer", "auc"])),
        "after_temp_f1": _mean_std(collect(["after_temp", "f1"])),
        "after_temp_auc": _mean_std(collect(["after_temp", "auc"])),
        "after_platt_f1": _mean_std(collect(["after_platt", "f1"])),
        "after_platt_auc": _mean_std(collect(["after_platt", "auc"])),
        "label_light_f1_from_dump": _mean_std(collect(["label_light_f1_from_dump"])),
        "T": _mean_std(collect(["after_temp", "T"])),
    }

    summary = {
        "roots": [str(r) for r in args.roots],
        "cal_frac": args.cal_frac,
        "n_dumps": len(results),
        "n_ok": len(ok),
        "n_err": len(err),
        "aggregate": aggregate,
        "per_dump": results,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    sum_path = args.out / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# CARDIUM external post-hoc recalibration (R7)",
        "",
        f"cal_frac={args.cal_frac}  n_ok={len(ok)}/{len(results)}",
        "",
        "| readout | F1 | AUC |",
        "|---|---:|---:|",
        f"| private-val thr (transfer) | {_fmt(aggregate['baseline_f1'])} | {_fmt(aggregate['baseline_auc'])} |",
        f"| after temperature | {_fmt(aggregate['after_temp_f1'])} | {_fmt(aggregate['after_temp_auc'])} |",
        f"| after Platt | {_fmt(aggregate['after_platt_f1'])} | {_fmt(aggregate['after_platt_auc'])} |",
        f"| label_light (from dump) | {_fmt(aggregate['label_light_f1_from_dump'])} | — |",
        "",
        f"Temperature T: {_fmt(aggregate['T'])}",
        "",
        "## Per dump",
        "",
        "| file | seed | base F1 | temp F1 | platt F1 | ll F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        name = Path(r["path"]).name
        if "error" in r:
            md.append(f"| {name} | — | ERR: {r['error']} | — | — | — |")
            continue
        b = (r.get("baseline_transfer") or {}).get("f1")
        t = (r.get("after_temp") or {}).get("f1")
        p = (r.get("after_platt") or {}).get("f1")
        ll = r.get("label_light_f1_from_dump")
        md.append(
            f"| {name} | {r.get('seed')} | "
            f"{b if b is None else f'{b:.4f}'} | "
            f"{t if t is None else f'{t:.4f}'} | "
            f"{p if p is None else f'{p:.4f}'} | "
            f"{ll if ll is None else f'{ll:.4f}'} |"
        )
    md_path = args.out / "AGGREGATE.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    latex = [
        "% CARDIUM external recalibration (mean±std over dumps)",
        f"Transfer & {_fmt(aggregate['baseline_f1'])} & {_fmt(aggregate['baseline_auc'])} \\\\",
        f"Temp. scaling & {_fmt(aggregate['after_temp_f1'])} & {_fmt(aggregate['after_temp_auc'])} \\\\",
        f"Platt & {_fmt(aggregate['after_platt_f1'])} & {_fmt(aggregate['after_platt_auc'])} \\\\",
        f"Label-light (dump) & {_fmt(aggregate['label_light_f1_from_dump'])} & --- \\\\",
    ]
    latex_path = args.out / "LATEX_SNIPPET.txt"
    latex_path.write_text("\n".join(latex) + "\n", encoding="utf-8")

    print(f"OK {len(ok)}/{len(results)} dumps")
    for k, v in aggregate.items():
        print(f"  {k}: {_fmt(v)}")
    print(f"Wrote {sum_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {latex_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
