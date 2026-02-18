---
name: run-pipeline
description: Run the pickinsta selection pipeline on an input folder with common presets
disable-model-invocation: true
---

Run the pickinsta pipeline. Usage: /run-pipeline <input_folder> [clip|claude] [--all]

Common invocations:
- CLIP (free, fast):   pickinsta <input> --output ./output/test_clip --scorer clip --top 10
- Claude (quality):    pickinsta <input> --output ./output/test_claude --scorer claude --all --claude-crop-first
- Budget Claude:       pickinsta <input> --output ./output/test_budget --scorer claude --vision-pct 0.5 --top 10

Reminder: Claude scoring costs ~$0.005/image. Cache files (*.pickinsta.json) prevent re-scoring unchanged images.
