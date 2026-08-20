# Cache format (no private images)

Fusion training in this drop is **cache-first**: `--embed-load-only` reads
FetalCLIP vectors + YOLO *tags* (JSONL), not pixels and not a `.pt` detector.

## Manifest (one exam / folder per line)

```json
{"patient_id": "id", "split": "train|val|test", "label": 0, "rel_dir": "relative/path"}
```

`label` is 1 for CHD, 0 for screening-negative.

## Image tags (YOLO outputs, not weights)

Each line is one frame: plane tag, anatomy boxes, mapped CARDIUM view
(`four_chamber` / `lvot` / `rvot` / `vvt`). Schema matches
`experiments/agcd/yolo_image_tag_cache.py` in the private working tree.

## FetalCLIP embedding cache

```json
{"sample_id": "...", "image_path": "...", "embedding": [768 floats]}
```

Build this with `agcd/fetalclip_embed.py` after downloading official FetalCLIP
weights (not shipped here).

## What is not in this repository

- Private tertiary ultrasound
- Hospital-trained YOLO checkpoint `yolo_full_with_tags_best.pt`
- FetalCLIP `.pt` weights (get them from Maani et al., npj Digital Medicine 2026)
- CARDIUM pixels (request from the CARDIUM authors)
