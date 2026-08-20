#!/usr/bin/env bash
# BSPC M5 probe: FetalCLIP LoRA embeds + matched ALVG vs Attention-MIL (view_anat).
#
# Smoke (3 seeds, 1 GPU):
#   CUDA_VISIBLE_DEVICES=0 SEEDS=42,43,44 bash experiments/run_tertiary_lora_probe.sh
#
# Full (10 seeds, multi-GPU):
#   GPUS=0,1 SEEDS=42,43,44,45,46,47,48,49,50,51 bash experiments/run_tertiary_lora_probe.sh
#
# Requires existing homologous LoRA adapter (default path below). To train first:
#   bash experiments/run_homologous_midlate_fetalclip_lora.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-42,43,44}"
FORCE="${FORCE:-0}"

MANIFEST="${MANIFEST:-$ROOT/data/study_screening/manifest_tertiary_20241125_vs_chd_era2020.jsonl}"
TAGS="${TAGS:-$ROOT/data/study_screening/yolo_image_tags_tertiary_20241125.jsonl}"
FEAT_CACHE="${FEAT_CACHE:-$ROOT/data/study_screening/masvf_m0_tertiary_20241125_era2020_feature_cache.jsonl}"
LORA_ADAPTER="${LORA_ADAPTER:-$ROOT/outputs/homologous_midlate_fetalclip_lora/best_adapter.pt}"
[[ -f "$LORA_ADAPTER" ]] || LORA_ADAPTER="${LORA_ADAPTER%/*}/adapter_best.pt"
EMBED_LORA="${EMBED_LORA:-$ROOT/data/study_screening/tertiary_20241125_vs_chd_raw_homologous_lora_embeddings.jsonl}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/tertiary_bspc_lora_probe}"

log() { echo "[$(date '+%F %T')] [lora_probe] $*" >&2; }

[[ -f "$MANIFEST" ]] || { log "ERROR: missing $MANIFEST"; exit 1; }
[[ -f "$TAGS" ]] || { log "ERROR: missing $TAGS"; exit 1; }
[[ -f "$FEAT_CACHE" ]] || { log "ERROR: missing $FEAT_CACHE"; exit 1; }
[[ -f "$LORA_ADAPTER" ]] || {
  log "ERROR: missing LoRA adapter at $LORA_ADAPTER"
  log "Train first: bash experiments/run_homologous_midlate_fetalclip_lora.sh"
  exit 1
}

mkdir -p "$OUT_ROOT" logs_parallel
chmod +x experiments/run_tertiary_fusion_worker.sh 2>/dev/null || true

# Phase 1: build LoRA embed cache via first MIL seed with EMBED_LOAD_ONLY=0
if [[ "$FORCE" == "1" || ! -s "$EMBED_LORA" ]]; then
  log "BUILD LoRA embeds via MIL seed42 (EMBED_LOAD_ONLY=0) -> $EMBED_LORA"
  mkdir -p "$(dirname "$EMBED_LORA")"
  gpu0="${GPUS%%,*}"
  env CUDA_VISIBLE_DEVICES="$gpu0" DEVICE=0 JOBS="attention_mil:42" \
    OUT_ROOT="$OUT_ROOT" FORCE=1 \
    EMBED_CACHE="$EMBED_LORA" \
    FEAT_CACHE="$FEAT_CACHE" MANIFEST="$MANIFEST" TAGS="$TAGS" \
    M0_FEATURES=none EXTRA_FEATS=view_anat \
    NO_VIEW_DECOMP=0 RUN_TAG_SUFFIX="__lora-mil-embedbuild" \
    LORA_ADAPTER="$LORA_ADAPTER" \
    EMBED_LOAD_ONLY=0 \
    NO_MVP=1 \
    bash experiments/run_tertiary_fusion_worker.sh \
    2>&1 | tee "$OUT_ROOT/embed_build.log"
  [[ -s "$EMBED_LORA" ]] || { log "ERROR: embed cache still empty: $EMBED_LORA"; exit 1; }
else
  log "SKIP embed build (exists): $EMBED_LORA"
fi

# Phase 2: ALVG + MIL under LoRA embeds
IFS=',' read -ra GPU_ARR <<< "$GPUS"
recipes=("alvg:anatomy_graph" "mil:attention_mil")
idx=0
pids=()
for rec in "${recipes[@]}"; do
  rid="${rec%%:*}"
  fusion="${rec##*:}"
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  jobs=()
  IFS=',' read -ra SEED_ARR <<< "$SEEDS"
  for seed in "${SEED_ARR[@]}"; do
    seed="$(echo "$seed" | tr -d '[:space:]')"
    [[ -n "$seed" ]] || continue
    jobs+=("${fusion}:${seed}")
  done
  jobs_csv=$(IFS=','; echo "${jobs[*]}")
  mvp_env=(NO_MVP=1)
  [[ "$fusion" == "anatomy_graph" ]] && mvp_env=(MVP=1)
  log "LAUNCH $rid on GPU$gpu jobs=$jobs_csv"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" DEVICE=0 JOBS="$jobs_csv" \
    OUT_ROOT="$OUT_ROOT" FORCE="$FORCE" \
    EMBED_CACHE="$EMBED_LORA" \
    FEAT_CACHE="$FEAT_CACHE" MANIFEST="$MANIFEST" TAGS="$TAGS" \
    M0_FEATURES=none EXTRA_FEATS=view_anat \
    NO_VIEW_DECOMP=0 RUN_TAG_SUFFIX="__lora-${rid}" \
    LORA_ADAPTER="$LORA_ADAPTER" \
    EMBED_LOAD_ONLY=1 \
    "${mvp_env[@]}" \
    bash experiments/run_tertiary_fusion_worker.sh \
    >"logs_parallel/lora_probe_${rid}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  idx=$((idx + 1))
done

ec=0
for pid in "${pids[@]}"; do
  wait "$pid" || { log "WARN: pid $pid exited non-zero"; ec=1; }
done

log "DONE (ec=$ec). Result files under $OUT_ROOT"
find "$OUT_ROOT" -name 'view_token_results_*.json' | head -40
exit "$ec"
