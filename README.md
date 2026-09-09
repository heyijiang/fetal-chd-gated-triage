# View-gated four-view fetal CHD triage (code drop)

Companion code for the manuscript *Multi-View Gated Fusion with an Anatomy-Linked Graph for Fetal CHD Referral Triage*.

This is a **sanitised** extract of the training / evaluation scripts. It does **not** contain:

- private ultrasound
- the hospital-trained plane+anatomy detector checkpoint (`yolo_full_with_tags_best.pt`)
- FetalCLIP weights (download from Maani et al., *npj Digital Medicine* 2026)
- CARDIUM pixels (request from the CARDIUM authors)

Published fusion tables use **precomputed embedding + tag caches** (`--embed-load-only`). You only need detector weights if you tag new images or run the optional wall-clock script.

## Layout

```
experiments/          fusion trainer, ALVG/MIL/GNN/Transformer heads, paper runners
configs/paper_recipe.yaml
examples/CACHE_FORMAT.md
tests/test_fusion_forward.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Paper timing used PyTorch `2.7.0+cu128`. CPU is enough to import the heads.

```bash
python tests/test_fusion_forward.py
```

## Paper recipe (once you have caches)

Seeds **42–51**. Frozen FetalCLIP, `view_anat` geometry, MVP on ALVG/GNN.

```bash
# ALVG (anatomy-graph design prior)
GPUS=0 SEEDS=42,43,44,45,46,47,48,49,50,51 \
  bash experiments/run_tertiary_alvg.sh

# Fair keep-4 heads (MIL / GNN / Transformer)
GPUS=0 bash experiments/run_tertiary_feat_ablation.sh

# Primary missing-view table
GPUS=0 bash experiments/run_tertiary_missing_view_heads.sh

# Keep-4 adjacency ablation
GPUS=0 bash experiments/run_tertiary_graph_ablation.sh
```

Point caches with env vars (see `experiments/run_tertiary_fusion_worker.sh`):

```bash
export MANIFEST=/path/to/manifest.jsonl
export TAGS=/path/to/image_tags.jsonl
export FEAT_CACHE=/path/to/m0_feature_cache.jsonl
export EMBED_CACHE=/path/to/fetalclip_embeddings.jsonl
```

Full-frame patient-mean probe (no view gate):

```bash
python -u experiments/private_fetalclip_fullframe_linear.py \
  --embed-load-only --cache "$EMBED_CACHE" --manifest "$MANIFEST"
```

Optional gated vs full-frame wall-clock (you supply detector weights):

```bash
export YOLO_WEIGHTS=/your/detector.pt
python -u experiments/profile_yolo_clip_e2e.py --device 0 --max-exams 32
```

## Reproducibility pins

| Item | Value |
|---|---|
| Fusion seeds | 42–51 (CARDIUM in-domain 42–44) |
| Detector (not shipped) | plane+anatomy checkpoint `yolo_full_with_tags_best.pt`, conf 0.25, imgsz 640 |
| Appearance | frozen FetalCLIP, dim 768 |
| Main channels | CLIP + `view_anat`, `m0=none` |
| Primary table | missing-view keep-$k$ (`run_tertiary_missing_view_heads.sh`) |

## License

MIT for this code drop. FetalCLIP, Ultralytics, and CARDIUM remain under their own licenses.
