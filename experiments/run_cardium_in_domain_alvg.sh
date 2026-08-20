#!/usr/bin/env bash
# CARDIUM **in-domain** protocol transfer (NOT tertiary→CARDIUM external).
# Ensures caches, then trains ALVG / MIL / view-token on CARDIUM folds 1–3.
#
# Default channels match tertiary ALVG winner: m0=none + view_anat.
# Each JSON also reports C2-b patient-mean/max (in-domain pooling baselines).
#
#   GPUS=0,1,4,5 bash experiments/run_cardium_in_domain_alvg.sh
#   SEEDS=42,43 GPUS=0 bash experiments/run_cardium_in_domain_alvg.sh  # smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p outputs logs_parallel

GPUS="${GPUS:-0,1,4,5}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
SEEDS="${SEEDS:-42,43,44}"
FOLDS="${FOLDS:-1,2,3}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/cardium_in_domain_fusion}"

CARD_TAGS="${CARD_TAGS:-$ROOT/data/study_screening/yolo_image_tags_cardium.jsonl}"
CARD_FEAT="${CARD_FEAT:-$ROOT/data/study_screening/masvf_m0_cardium_feature_cache.jsonl}"
CARD_EMBED="${CARD_EMBED:-$ROOT/data/study_screening/cardium_raw_base_embeddings.jsonl}"
CARDIUM_PROCESSED="${CARDIUM_PROCESSED:-$ROOT/CARDIUM dataset/processed}"

# recipe_id:fusion:m0:extra:mvp
DEFAULT_RECIPES=(
  "alvg:anatomy_graph:none:view_anat:1"
  "mil:attention_mil:none:view_anat:0"
  "ft:feature_transformer:none:view_anat:0"
  "mil_core3:attention_mil:true_ratios_core3:none:0"
)

RECIPES_CSV="${RECIPES:-}"
if [[ -n "$RECIPES_CSV" ]]; then
  IFS=',' read -ra WANT <<< "$RECIPES_CSV"
  SELECTED=()
  for want in "${WANT[@]}"; do
    want="$(echo "$want" | tr -d '[:space:]')"
    for rec in "${DEFAULT_RECIPES[@]}"; do
      [[ "${rec%%:*}" == "$want" ]] && SELECTED+=("$rec") && break
    done
  done
  RECIPE_ARR=("${SELECTED[@]}")
else
  RECIPE_ARR=("${DEFAULT_RECIPES[@]}")
fi

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
log() { echo "[$(date '+%F %T')] [cardium_id] $*" >&2; }

wait_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if wait "$pid" 2>/dev/null; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
  done
}

[[ -f "$CARD_TAGS" ]] || { log "ERROR missing $CARD_TAGS"; exit 1; }
[[ -d "$CARDIUM_PROCESSED" ]] || { log "ERROR missing $CARDIUM_PROCESSED"; exit 1; }
mkdir -p "$OUT_ROOT"

# --- ensure CARDIUM M0 ---
log "Ensure CARDIUM M0 → $CARD_FEAT"
CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u - <<PY
import sys
from pathlib import Path
ROOT = Path(r"$ROOT")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
from masvf_m0_screening import load_feature_cache, load_tags, build_cardium_frame_rows
from cardium_dataset import load_fold_split
from yolo_io import resolve_weights, load_yolo_model
import argparse

card_feat = Path(r"$CARD_FEAT")
card_tags = Path(r"$CARD_TAGS")
processed = Path(r"$CARDIUM_PROCESSED")
tags = load_tags(card_tags)
cache = load_feature_cache(card_feat)
recs = []
for fold in "1,2,3".split(","):
    for sp in ("train", "test"):
        recs.extend(load_fold_split(fold, sp, processed))
miss = [r for r in recs if r.sample_id not in cache]
print(f"cardium frames={len(recs)} cache_hit={len(recs)-len(miss)} miss={len(miss)}", flush=True)
if not miss:
    raise SystemExit(0)
ns = argparse.Namespace(
    batch_size=64, yolo_conf=0.25, yolo_imgsz=640, device="0",
    min_plane_conf=0.15, rebuild_cache=False, feature_cache=card_feat,
    allow_missing_cache=False, _unreadable_skip_total=0,
)
yolo = load_yolo_model(resolve_weights(None))
build_cardium_frame_rows(miss, tags=tags, yolo_model=yolo, args=ns, cache=cache)
print("cardium M0 backfill done", flush=True)
PY

# --- ensure CARDIUM raw embeds ---
log "Ensure CARDIUM embeds → $CARD_EMBED"
CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u - <<PY
import sys
from pathlib import Path
ROOT = Path(r"$ROOT")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments"))
from agcd.fetalclip_embed import build_or_load_embeddings
from cardium_dataset import load_fold_split

out = Path(r"$CARD_EMBED")
processed = Path(r"$CARDIUM_PROCESSED")
items = []
for fold in "1,2,3".split(","):
    for sp in ("train", "test"):
        for rec in load_fold_split(fold, sp, processed):
            items.append((rec.sample_id, Path(rec.image_path)))
print(f"cardium embed inventory={len(items)}", flush=True)
cache = build_or_load_embeddings(
    items, cache_path=out, device="0", batch_size=64,
    lora_adapter=None, grayscale=False,
)
print(f"cardium embed cache size≈{len(cache)} → {out}", flush=True)
PY

launch() {
  local rec="$1" gpu="$2"
  local rid fusion m0 extra mvp
  IFS=':' read -r rid fusion m0 extra mvp <<< "$rec"
  local jobs=() seed
  for seed in "${SEED_ARR[@]}"; do
    seed="$(echo "$seed" | tr -d '[:space:]')"
    [[ -n "$seed" ]] || continue
    jobs+=("${fusion}:${seed}")
  done
  local jobs_csv; jobs_csv=$(IFS=','; echo "${jobs[*]}")
  local envx=(
    OUT_ROOT="$OUT_ROOT" FORCE="$FORCE" FOLDS="$FOLDS"
    CARD_TAGS="$CARD_TAGS" CARD_FEAT="$CARD_FEAT" CARD_EMBED="$CARD_EMBED"
    CARDIUM_PROCESSED="$CARDIUM_PROCESSED"
    FEAT_MODEL=m1 M0_FEATURES="$m0" EXTRA_FEATS="$extra"
    NO_VIEW_DECOMP=1 RUN_TAG_SUFFIX="__r-${rid}"
  )
  if [[ "$mvp" == "1" ]]; then envx+=(MVP=1); else envx+=(NO_MVP=1); fi
  log "GPU$gpu recipe=$rid fusion=$fusion m0=$m0 extra=$extra"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 JOBS="$jobs_csv" "${envx[@]}" \
    bash experiments/run_cardium_in_domain_fusion_worker.sh \
    >"logs_parallel/cardium_id_${rid}_gpu${gpu}.log" 2>&1 &
  LAST_PID=$!
}

log "=== CARDIUM in-domain fusion ==="
PIDS=()
idx=0
for rec in "${RECIPE_ARR[@]}"; do
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  idx=$((idx + 1))
  launch "$rec" "$gpu"
  [[ "$LAST_PID" =~ ^[0-9]+$ ]] || { log "bad pid $LAST_PID"; exit 1; }
  PIDS+=("$LAST_PID")
  log "launched pid=$LAST_PID"
done
for pid in "${PIDS[@]}"; do
  log "waiting $pid"
  wait_pid "$pid"
done

log "summarize"
python -u experiments/summarize_cardium_in_domain.py --fusion-root "$OUT_ROOT" \
  --out "$OUT_ROOT/CARDIUM_IN_DOMAIN.md" || true
log "DONE → $OUT_ROOT"
