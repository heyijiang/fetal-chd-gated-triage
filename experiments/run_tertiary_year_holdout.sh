#!/usr/bin/env bash
# R6-F4: year-holdout ablation (patient-mean).
#
# Default HOLD_MODE=chd_year keeps all normals in train/val so holding out
# 2024 does NOT create a one-class training set (normals are 2024-heavy).
#
#   cd /path/to/fetal-chd-gated-triage
#   CUDA_VISIBLE_DEVICES=0 bash experiments/run_tertiary_year_holdout.sh
#
# Env:
#   HOLD_YEARS=2023,2024,2025
#   HOLD_MODE=chd_year     # or both (strict; may skip 2024)
#   SEEDS=42
#   FORCE=1                # rebuild manifests + rerun
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-0}"
HOLD_YEARS="${HOLD_YEARS:-2023,2024,2025}"
HOLD_MODE="${HOLD_MODE:-chd_year}"
SEEDS="${SEEDS:-42}"
FORCE="${FORCE:-0}"
SKIP_ALVG="${SKIP_ALVG:-1}"
PREPROCESS="${PREPROCESS:-raw}"
ABN_MIN_YEAR="${ABN_MIN_YEAR:-2020}"

MANIFEST="${MANIFEST:-$ROOT/data/study_screening/manifest_tertiary_20241125_vs_chd_era${ABN_MIN_YEAR}.jsonl}"
CACHE="${CACHE:-$ROOT/data/study_screening/tertiary_20241125_vs_chd_raw_base_embeddings.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_r6_year_holdout}"
MAN_DIR="$OUT_ROOT/manifests"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

log() { echo "[$(date '+%F %T')] [year-holdout] $*"; }

[[ -f "$MANIFEST" ]] || { log "ERROR: missing $MANIFEST"; exit 1; }
[[ -f "$CACHE" ]] || { log "ERROR: missing $CACHE"; exit 1; }
mkdir -p "$OUT_ROOT"

log "=== build holdout manifests (mode=$HOLD_MODE) ==="
# Always rebuild manifests when FORCE=1 or missing inventory
if [[ "$FORCE" == "1" || ! -f "$MAN_DIR/inventory.json" ]]; then
  rm -f "$MAN_DIR"/manifest_hold_*.jsonl 2>/dev/null || true
fi
python -u experiments/build_tertiary_year_holdout_manifests.py \
  --manifest "$MANIFEST" \
  --hold-years "$HOLD_YEARS" \
  --mode "$HOLD_MODE" \
  --embed-cache "$CACHE" \
  --out-dir "$MAN_DIR"

if [[ "$SKIP_TRAIN" == "1" ]]; then
  log "SKIP_TRAIN=1: inventory only"
  python -u experiments/summarize_tertiary_year_holdout.py --root "$OUT_ROOT" || true
  exit 0
fi

log "=== patient-mean per hold year ==="
shopt -s nullglob
ok=0
fail=0
for man in "$MAN_DIR"/manifest_hold_*.jsonl; do
  year="$(basename "$man" | sed -E 's/manifest_hold_([0-9]+)\.jsonl/\1/')"
  out="$OUT_ROOT/hold_${year}/patient_mean"
  mkdir -p "$out"
  log "year=$year manifest=$man → $out"
  set +e
  FORCE="$FORCE" SEEDS="$SEEDS" PREPROCESS="$PREPROCESS" \
    MANIFEST="$man" \
    CACHE="$CACHE" \
    OUT_ROOT="$out" \
    DEVICE="$DEVICE" \
    bash experiments/run_tertiary_canon_patient_mean_seeds.sh
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
    log "WARN: year=$year patient-mean failed rc=$rc (continue)"
  fi
done

if [[ "$SKIP_ALVG" != "1" ]]; then
  log "ALVG not wired; patient-mean is enough for R6-F4."
fi

log "=== summarize (ok=$ok fail=$fail) ==="
python -u experiments/summarize_tertiary_year_holdout.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/summary.json"

if [[ "$ok" -eq 0 ]]; then
  log "ERROR: all holdout years failed"
  exit 1
fi

log "DONE → $OUT_ROOT/summary.json  $OUT_ROOT/AGGREGATE.md"
log "Pack: tar -czf tertiary_r6_year_holdout.tgz -C outputs tertiary_r6_year_holdout"
