# pickinsta Reference

**Main file:** `src/pickinsta/ig_image_selector.py` (~5000 lines)

---

## 1. CLI Flags

| Flag | Short | Type | Default | Effect |
|------|-------|------|---------|--------|
| `--output` | `-o` | path | `selected` | Output folder for final variants |
| `--work` | `-w` | path | `<input>_work` (sibling of input) | Intermediate work folder (resized images, caches) |
| `--top` | `-n` | int | `10` | Number of top-scoring images to output |
| `--scorer` | `-s` | enum | — | Vision scorer: `clip`, `claude`, or `ollama` |
| `--all` | — | flag | off | Score all Stage 2 images; ignores `--vision-pct` |
| `--vision-pct` | — | float | `0.5` | Fraction of technically-filtered images sent to vision scoring |
| `--claude-model` | — | str | `$ANTHROPIC_MODEL` or `claude-haiku-4-5-20251001` | Override Claude model for this run |
| `--claude-crop-first` | — | flag | off | Pre-crop to 1080×1440 before sending to Claude |
| `--rescore` | — | flag | off | Ignore all cached vision scores; force re-scoring |
| `--dedup-only` | — | flag | off | Output best shot per burst + all unique images as full/hd/cropped; skips scoring, ranking, debug |

---

## 2. Environment Variables

**Search order:** current process environment → `<cwd>/.env` → `<input_folder>/.env`

### Anthropic / Claude

| Variable | Default | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | — | Required for Claude scorer |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Primary Claude model override |
| `CLAUDE_MODEL` | — | Alias fallback; checked after `ANTHROPIC_MODEL` |

Model resolution order: `--claude-model` flag → `ANTHROPIC_MODEL` → `CLAUDE_MODEL` → `claude-haiku-4-5-20251001` → undated alias → `claude-3-5-sonnet-latest`.

### HuggingFace

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_TOKEN` | — | Reduces rate limit warnings on CLIP model download |

### pickinsta-specific

| Variable | Default | Range / Notes |
|----------|---------|---------------|
| `PICKINSTA_ACCOUNT_CONTEXT` | — | Injected into Claude/Ollama prompts; changing this value invalidates all vision caches (prompt hash includes context) |
| `PICKINSTA_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server endpoint |
| `PICKINSTA_OLLAMA_MODEL` | `qwen2.5vl:7b` | Ollama model tag |
| `PICKINSTA_OLLAMA_CONCURRENCY` | `2` | Parallel Ollama requests; range 1–16 |
| `PICKINSTA_OLLAMA_MAX_RETRIES` | `2` | Retries on transient failures |
| `PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC` | `0.75` | Exponential backoff base (seconds) |
| `PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS` | `6` | Consecutive failures before fallback triggers |
| `PICKINSTA_YOLO_MODEL` | `~/.cache/pickinsta/models/yolov8n.pt` | YOLO model path override |

---

## 3. Pipeline Stages

### Stage 0 — Resize

| Item | Value |
|------|-------|
| Input | Source images from input folder |
| Output | Resized JPEGs in work folder |
| Skip condition | Work image exists and mtime is current |
| Parallelism | `ProcessPoolExecutor` (PIL only, safe to fork) |

Resizes longest edge to max 1920px. Preserves EXIF data; resets orientation tag to 1 so downstream EXIF timestamps are available.

---

### Stage 1 — Deduplicate

| Item | Value |
|------|-------|
| Input | Work folder images |
| Output | Deduplicated candidate list; burst group metadata |
| Parallelism | `ProcessPoolExecutor` for feature extraction; grouping sequential |

**Two-pass dedup:**

- **Pass 1:** Perceptual hash (distance ≤ 8) — groups pixel-identical/near-identical images; selects sharpest (Laplacian variance).
- **Pass 2:** Histogram correlation + EXIF temporal chaining + ORB feature verification.
  - Images sorted by EXIF timestamp.
  - Each candidate compared against chain tail (last member of current group).
  - Temporal window: within 3 seconds of chain tail.
  - Matching tiers:
    | Condition | Histogram threshold | ORB threshold |
    |-----------|--------------------|-|
    | Temporal + strong ORB (≥ 0.25) | ≥ 0.60 | ≥ 0.25 |
    | Temporal only | ≥ 0.80 | ≥ 0.25 |
    | Non-temporal | ≥ 0.92 | ORB ≥ 0.25 |
  - ORB confirms subject identity, not just scene similarity — prevents grouping different riders at the same track position where background histograms are similar.

Burst metadata (count, selection method, members) tracked in report and gallery.

---

### Stage 2 — Technical Scoring

| Item | Value |
|------|-------|
| Input | Deduplicated work images |
| Output | Technical score per image; cache: `<work_image>.jpg.techscore.json` |
| Skip condition | Cache exists and keyed mtime matches |
| Parallelism | `ThreadPoolExecutor` (YOLO/PyTorch; see Section 7) |

Runs OpenCV-based quality metrics. See Section 4a for weights.

---

### Stage 2b — Burst Re-evaluation

| Item | Value |
|------|-------|
| Input | Top candidates from burst groups |
| Output | Possibly replaces sharpness-based pick with highest-technical-score member |
| Parallelism | `ThreadPoolExecutor` |

All burst members for top candidates are fully technically scored in parallel. The burst member with the highest composite score replaces the original sharpness-based selection if it scores higher.

---

### Stage 3 — Vision Scoring

| Item | Value |
|------|-------|
| Input | Top fraction of technically-scored images (controlled by `--vision-pct` or `--all`) |
| Output | Vision score per image; cache: `<source_filename>.pickinsta.json` |
| Skip condition | Valid cache entry exists (SHA256 + prompt hash match); bypass with `--rescore` |

Three scorer options: CLIP, Claude, Ollama. See Section 4b.

---

### Stage 4 — Smart Crop + Output

| Item | Value |
|------|-------|
| Input | Final ranked images |
| Output | Three variants per image in output folder; `index.html`; `selection_report.json`; `selection_report.md` |
| Parallelism | `ThreadPoolExecutor` (YOLO/PyTorch) |

YOLO detects subjects (motorcycles, people, vehicles). Crop scored on: power point placement (40%), lead room (35%), subject not clipped (25%). Blur padding applied if crop window can't be filled. Falls back to saliency detection if YOLO finds nothing.

Shot classification: `close-up`, `medium`, `environmental`, `scenic`, `extreme_wide` based on subject area ratio.

---

## 4. Scoring Reference

### 4a. Technical Scoring

All metrics return a value in [0.0, 1.0]. Weights sum to 1.0.

| Metric | Weight | Measurement |
|--------|--------|-------------|
| Composition | 0.20 | Rule-of-thirds / Phi Grid power points + horizon tilt + lead room |
| Sharpness | 0.18 | Laplacian variance on subject region |
| Lighting | 0.18 | Histogram clipping + mean luminance balance |
| Color harmony | 0.13 | Hasler-Süsstrunk colorfulness + subject-bg hue contrast |
| Background separation | 0.12 | Subject-to-background sharpness ratio |
| Visual clutter | 0.12 | Inverse edge density in background |
| Aesthetic | 0.07 | Contrast + saturation balance |

Output: `technical_composite` in [0.0, 1.0].

---

### 4b. Vision Scorers

#### CLIP (`--scorer clip`)

- **Model:** Loaded lazily from `src/pickinsta/clip_scorer.py`.
- **Input:** Work image.
- **Prompts:** 4 positive + 2 negative zero-shot classification prompts.
- **Output:** Logits mapped to 0–60 scale.
- **Cost:** Free, local, no API calls.
- **Cache key:** Same `<source_filename>.pickinsta.json` as Claude; keyed on prompt hash.

#### Claude (`--scorer claude`)

- **Default model:** `claude-haiku-4-5-20251001`
- **Image preparation:** Downsized to 1024px / q75 JPEG before API call (reduces token cost).
- **Prompt:** `VISION_PROMPT_TEMPLATE` / `build_vision_prompt(...)` in main file. Includes YOLO detection context and `PICKINSTA_ACCOUNT_CONTEXT`.
- **Scored criteria** (each 0–10):

  | Criterion | Notes |
  |-----------|-------|
  | `subject_clarity` | Ducati brand bonus: +2 |
  | `lighting` | — |
  | `color_pop` | — |
  | `emotion` | Ducati brand bonus: +2 |
  | `scroll_stop` | — |
  | `crop_4x5` | — |

- **Total:** 0–60 (before brand bonus).
- **Response JSON keys:** `subject_clarity`, `lighting`, `color_pop`, `emotion`, `scroll_stop`, `crop_4x5`, `total`, `one_line`.
- **Concurrency:** Adaptive; starts at 3 workers, scales to max 8 on success, backs off on HTTP 429 / rate limit errors, retries up to 3 times with exponential backoff. Implemented with `ThreadPoolExecutor`.
- **Cost estimate:** Printed before scoring run. ~$0.005/image (varies by model and image size).
- **Cache key:** SHA256 of source file + hash of prompt (includes account context). Model not checked by default; use `--rescore` to force re-scoring when switching models.
- **Cache file:** `<source_filename>.pickinsta.json` next to the source file.

#### Ollama (`--scorer ollama`)

- **Default model:** `qwen2.5vl:7b`
- **Supported model families:** `qwen2.5vl` (Qwen 2.5 VL), `gemma4` (Gemma 4). Other models fall back to a generic JSON prompt.
- **Same 0–60 rubric as Claude.**
- **Concurrency:** Configured via `PICKINSTA_OLLAMA_CONCURRENCY` (default 2, range 1–16).
- **Reliability:** Retry/backoff (`PICKINSTA_OLLAMA_MAX_RETRIES`, `PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC`) + circuit breaker (`PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS` consecutive failures triggers fallback).
- **Cache:** Same mechanism as Claude — `<source_filename>.pickinsta.json`, keyed on SHA256 + prompt hash.

**Model-specific behaviour:**

| Model family | Compact JSON prompt | `think: false` in payload | Temperature | Token budget |
|---|---|---|---|---|
| `qwen2.5vl:*` | Yes | Yes | 0 | 512 |
| `gemma4:*` | Yes | **Omitted** (Ollama bug [#15260](https://github.com/ollama/ollama/issues/15260)) | 0.3 | 280 |
| Others | No (generic) | Yes | 0 | 220 |

> **Gemma 4 note:** Sending `think=false` alongside the `format` parameter causes Ollama to silently ignore the structured output schema (bug #15260). The workaround is to omit the `think` key entirely for `gemma4:*` models. Thinking mode adds ~3–5 s latency per image but structured output works correctly.

---

### 4c. Final Score Formula

```
final_score = 0.3 × technical_composite + 0.7 × vision_normalized
```

- `technical_composite`: weighted sum of Stage 2 metrics, [0.0, 1.0].
- `vision_normalized`: vision scorer output normalized to [0.0, 1.0] (i.e., raw 0–60 score ÷ 60).

---

## 5. Caching

| Cache | File | Key | Invalidation |
|-------|------|-----|--------------|
| Vision (Claude/Ollama/CLIP) | `<source_filename>.pickinsta.json` (next to source) | SHA256 of source file + prompt hash (prompt hash includes `PICKINSTA_ACCOUNT_CONTEXT`) | Source file changes, prompt template changes, `PICKINSTA_ACCOUNT_CONTEXT` changes, or `--rescore` flag |
| Technical score | `<work_image>.jpg.techscore.json` (in work folder) | Work image mtime | Work image mtime changes (e.g., Stage 0 re-run) |
| Stage 0 resize | Work image mtime vs source mtime | mtime comparison | Source file modified |

**`--rescore`:** Forces all vision caches to be ignored; does not affect technical score caches.

**Model switching:** By default, switching Claude models reuses cached vision scores (model not included in cache key). Use `--rescore` to invalidate after a model change.

**Prompt hash includes:** Prompt template content + `PICKINSTA_ACCOUNT_CONTEXT`. Changing either value automatically invalidates all vision caches without needing `--rescore`.

---

## 6. Output File Conventions

### Per-image Variants

All variants for rank `XX` and base name `<name>` are written to the output folder:

| File | Description |
|------|-------------|
| `XX_cropped_<name>.jpg` | 1080×1440 IG-ready smart crop; blur padding applied if needed |
| `XX_hd_<name>.jpg` | 1920px longest edge, original aspect ratio |
| `XX_full_<name>.<ext>` | Original source file, untouched (extension preserved) |

`XX` is zero-padded rank (e.g., `01`, `02`, …).

### Run Artifacts

| File | Location | Description |
|------|----------|-------------|
| `index.html` | Output folder | Standalone HTML gallery; auto-generated after each run. Regenerate manually: `python scripts/generate_gallery.py <folder>` |
| `selection_report.json` | Output folder | Machine-readable structured report with scores, burst info, YOLO metadata |
| `selection_report.md` | Output folder | Human-readable summary |

### Gallery Features

- Detail panel: cropped/hd/full tabs, YOLO detection overlay, EXIF info, score bars, burst info, AI assessment.
- Breadcrumb navigation, recursive folder index with image counts and thumbnails.
- Uncertain crop warning badges; burst count badges.
- `--dedup-only` mode does not generate debug/gallery artifacts.

---

## 7. Parallelism Model

| Executor | Stages | Reason |
|----------|--------|--------|
| `ProcessPoolExecutor` | Stage 0 (resize), Stage 1 feature extraction | PIL / hash / histogram / EXIF only; no YOLO or PyTorch; safe to fork |
| `ThreadPoolExecutor` | Stage 2 (technical scoring), Stage 2b (burst re-eval), Stage 3 Claude (adaptive concurrency), Stage 4 (smart crop + output) | These code paths load YOLO/PyTorch; `fork()` with PyTorch causes deadlocks; OpenCV and YOLO release the GIL so threads achieve real parallelism |

**Rule:** Never use `ProcessPoolExecutor` for any code path that loads YOLO or PyTorch.

**Cached results** skip worker pools entirely — no executor overhead for already-scored images.

**Stage 1 grouping** is always sequential (grouping logic depends on order of EXIF-sorted candidates; cannot be parallelized).

**Stage 3 Claude concurrency:** Starts at 3 threads, scales to 8 on consecutive successes, backs off to lower concurrency on HTTP 429 or rate limit errors, retries individual requests up to 3 times with exponential backoff.
