# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pickinsta** is an Instagram image selection pipeline for motorcycle/Ducati photography. It processes event photo dumps into Instagram-ready cover candidates (1080x1440 portrait) by scoring technical quality and aesthetic potential.

## Architecture

### 6-Stage Pipeline

The core pipeline in `src/pickinsta/ig_image_selector.py` processes images through:

1. **Stage 0 - Resize**: Resize to max 1920px, parallel across CPU cores (saves compute, preserves EXIF orientation)
2. **Stage 1 - Deduplicate**: Two-pass dedup: perceptual hashing + histogram correlation with EXIF temporal burst detection. Feature extraction parallelized, grouping sequential.
3. **Stage 2 - Technical Scoring**: OpenCV-based quality metrics, parallel across CPU cores with per-image cache (`.techscore.json`)
4. **Stage 2b - Burst Re-evaluation**: For top candidates from burst groups, all burst members are technically scored in parallel; best replaces the sharpness pick if it scores higher.
5. **Stage 3 - Vision Scoring**: CLIP (local/free), Claude API (adaptive concurrency with rate limit backoff), or self-hosted Ollama for aesthetic evaluation
6. **Stage 4 - Smart Crop + Output**: YOLO-guided crop to 1080x1440, parallel via ThreadPoolExecutor. Outputs three variants per image: full (original), hd (1920px), cropped (IG 1080x1440).

### Key Design Decisions

**Parallelization**:
- Stage 0 (resize): `ProcessPoolExecutor` — PIL only, safe to fork
- Stage 1 (dedup feature extraction): `ProcessPoolExecutor` — hash/histogram/EXIF, no YOLO
- Stages 2, 2b, 4 (tech scoring, burst re-eval, smart crop): `ThreadPoolExecutor` — these use YOLO/PyTorch which deadlocks with `fork()`, but OpenCV/YOLO release the GIL so threads still get parallelism
- Stage 3 Claude: `ThreadPoolExecutor` with adaptive concurrency (starts at 3, scales to 8, backs off on 429/rate limits, retries up to 3 times with exponential backoff)
- Cached results skip worker pools entirely
- **Important**: Never use `ProcessPoolExecutor` for code paths that load YOLO/PyTorch — use `ThreadPoolExecutor` instead to avoid fork deadlocks

**Burst Detection** (three-layer dedup):
- Pass 1: perceptual hash (distance ≤8) groups pixel-identical images, selects sharpest (Laplacian variance)
- Pass 2: histogram correlation + EXIF temporal chaining + ORB feature verification
  - Images sorted by EXIF timestamp; each candidate compared against last group member (chain tail)
  - Must be within 3 seconds of the chain tail to be considered temporal
  - Three matching tiers:
    - **Temporal + strong ORB** (≥0.25): histogram only needs 0.60 (handles exposure shifts in burst)
    - **Temporal only**: histogram ≥0.80 + ORB ≥0.25
    - **Non-temporal**: histogram ≥0.92 + ORB ≥0.25
  - **ORB verification** confirms the subject matches, not just the scene — prevents grouping different riders at the same track position (histogram alone can't distinguish these since the background dominates)
- Stage 2b re-evaluates top candidates from bursts using full technical scoring in parallel
- Burst metadata (count, selection method, members) tracked in report and gallery
- **Important**: Stage 0 resize preserves EXIF data (with orientation tag reset to 1) so timestamps are available for burst detection on work images

**YOLO Integration**:
- YOLOv8 detects subjects (motorcycles, people, vehicles) before cropping
- Ensures crops keep the full subject in frame (previously used unreliable saliency detection)
- YOLO context is passed to Claude and Ollama to improve scoring accuracy
- Ultralytics banner/warnings suppressed (`YOLO_VERBOSE=false`, `task="detect"`, `verbose=False`)
- Graceful fallback to saliency detection if YOLO finds nothing
- See `debug/README.md` and `debug/debug_yolo_claude.py` for debugging details

**Three-Scorer Architecture**:
- **CLIP** (`--scorer clip`): Free, local, zero-shot classification. Uses 4 positive + 2 negative prompts. Maps logits to 0-60 scale to match Claude's range.
- **Claude** (`--scorer claude`): Default model `claude-haiku-4-5-20251001`. Images downsized to 1024px/q75 before API call to reduce token cost. Scores 6 criteria (subject_clarity, lighting, color_pop, emotion, scroll_stop, crop_4x5). Returns JSON with scores + one-line summary. Brand bonus: Ducati bikes get +2 on subject_clarity and emotion.
- **Ollama** (`--scorer ollama`): Self-hosted vision scoring with the same 0-60 rubric. Supports retry/backoff, circuit breaker, and configurable request concurrency.

**Final Score Calculation**: `final_score = 0.3 * technical_composite + 0.7 * vision_normalized`

**Caching Strategy**:
- Claude vision responses cached per source file as `<filename>.pickinsta.json` (SHA256 + prompt hash; model check skipped by default for cross-model reuse, forced with `--rescore`). Prompt hash includes account context — changing `PICKINSTA_ACCOUNT_CONTEXT` invalidates caches. Cost estimator resolves the same account context as the scorer.
- Technical scores cached per work image as `<filename>.jpg.techscore.json` (keyed on file mtime)
- Stage 0 resize cached via mtime check on work folder output

**Output Variants**:
- `XX_cropped_<name>.jpg` — 1080x1440 IG smart crop (with blur padding if needed)
- `XX_hd_<name>.jpg` — 1920px longest edge, original aspect ratio
- `XX_full_<name>.<ext>` — original source file, untouched

**HTML Gallery** (`index.html`):
- Auto-generated at end of pipeline run in output folder
- Standalone regeneration: `python scripts/generate_gallery.py <folder>`
- Recursive folder indexes with image counts and thumbnails
- Detail panel: preview (cropped/hd/full tabs), YOLO detection overlay, EXIF info, score bars, burst info, AI assessment
- Breadcrumb navigation, GitHub link, uncertain crop warning badges, burst count badges

### Composition Rules Implementation

Technical scoring follows a weighted rubric (see `docs/composition-rules.md`):
- Sharpness (0.18): Laplacian variance on subject region
- Background separation (0.12): Subject-to-background sharpness ratio
- Composition (0.20): Rule-of-thirds/Phi Grid power points + horizon tilt + lead room
- Lighting (0.18): Histogram clipping + mean luminance balance
- Color harmony (0.13): Hasler-Süsstrunk colorfulness + subject-bg hue contrast
- Visual clutter (0.12): Inverse edge density in background
- Aesthetic (0.07): Contrast + saturation balance

Smart cropping uses these rules to:
1. Classify shot type (close-up, medium, environmental, scenic, extreme_wide) from subject area
2. Guess facing direction (left/right/head-on) using horizontal Sobel gradient
3. Generate candidate crop windows respecting lead room (60-70% space ahead of subject)
4. Score each candidate on: power point placement (40%), lead room (35%), subject not clipped (25%)

## Development Commands

```bash
make install-dev        # Install dev + all scorer extras (clip, claude, yolo)
make test               # Run pytest suite
make lint               # Run ruff linting checks
make format             # Auto-format with ruff
make check              # lint + test (run before committing)
make pre-commit-install # Install pre-commit hooks (ruff + formatting on git commit)
```

### Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-dev
```

This installs dev tools and all scorer extras (clip, claude, yolo). To install only specific scorers: `pip install -e ".[dev,claude]"`, etc.

### Running the Pipeline

```bash
# CLIP scorer (free, local), top 10
pickinsta ./input --output ./selected --top 10 --scorer clip

# Claude scorer, all images, with separate work folder
pickinsta ./input --output ./selected --work ./work --scorer claude --all

# Override model, force re-scoring
pickinsta ./input --scorer claude --claude-model claude-sonnet-4-6 --rescore

# Ollama scorer (self-hosted), score all candidates
pickinsta ./input --output ./selected --scorer ollama --all

# Dedup-only: best shot per burst, all unique images, no scoring/ranking
pickinsta ./input --output ./deduped --dedup-only

# Dedup-only with separate work folder
pickinsta ./input --output ./deduped --work ./work --dedup-only
```

### Environment Setup

```bash
cp .env.example .env
```

`.env` search order: current environment → `cwd/.env` → `input_folder/.env`

Key variables:
- `ANTHROPIC_API_KEY` — required for Claude scorer
- `ANTHROPIC_MODEL` — override default model (default: `claude-haiku-4-5-20251001`)
- `CLAUDE_MODEL` — alias fallback for Claude model resolution
- `HF_TOKEN` — reduces HuggingFace rate limit warnings (CLIP)
- `PICKINSTA_ACCOUNT_CONTEXT` — custom account context injected into Claude/Ollama prompts (e.g. `"Ducati/motorcycle enthusiast account in Southern California."`)
- `PICKINSTA_OLLAMA_BASE_URL` — Ollama endpoint (default: `http://127.0.0.1:11434`)
- `PICKINSTA_OLLAMA_MODEL` — Ollama model tag (default: `qwen2.5vl:7b`)
- `PICKINSTA_OLLAMA_CONCURRENCY` — parallel requests submitted by pickinsta (default: `2`, min `1`, max `16`)
- `PICKINSTA_OLLAMA_MAX_RETRIES` — retries for transient failures (default: `2`)
- `PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC` — exponential backoff base seconds (default: `0.75`)
- `PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS` — consecutive failure threshold before fallback (default: `6`)
- `PICKINSTA_YOLO_MODEL` — override YOLO model path (default: `~/.cache/pickinsta/models/yolov8n.pt`)

### CLI Flags

- `--output, -o` — output folder (default: `selected`)
- `--work, -w` — work folder for intermediate files (default: `<input>_work` next to input)
- `--top, -n` — number of top images to output (default: 10)
- `--scorer, -s` — vision scorer: `clip`, `claude`, or `ollama`
- `--all` — score all Stage 2 images (ignore `--vision-pct`)
- `--vision-pct` — fraction of images to send to vision scoring (default: 0.5)
- `--claude-model` — override Claude model
- `--claude-crop-first` — pre-crop to 1080x1440 before Claude scoring
- `--rescore` — force re-scoring, ignoring all cached vision scores
- `--dedup-only` — dedup-only mode: best shot per burst, output all unique images as full/hd/cropped (no scoring, no ranking, no debug)

### Testing

```bash
pytest                            # Run all tests
pytest tests/test_crop.py        # Single test file
pytest -k "test_smart_crop"      # Single test by name

python debug/debug_yolo_claude.py  # Manual debug: inspect YOLO → Claude prompt enrichment
```

Debug mode creates visualizations showing:
- Green boxes: YOLO detections
- Red dots: Subject center points
- Yellow lines: Rule-of-thirds grid
- Cyan lines: Phi Grid

### Gallery Generation

```bash
# Regenerate galleries for all output folders
python scripts/generate_gallery.py ~/Photos/td6_selected

# Single folder
python scripts/generate_gallery.py ~/Photos/td6_selected/2ab/Session_1_Turn_2
```

Gallery is also auto-generated at the end of each pipeline run.

## Important File Locations

- **Main pipeline**: `src/pickinsta/ig_image_selector.py` (~4,000 lines, all stages + gallery + CLI)
- **CLIP scorer**: `src/pickinsta/clip_scorer.py` (separate module, loaded lazily on `--scorer clip`)
- **Gallery generator**: `scripts/generate_gallery.py` (standalone, imports from main module)
- **Config**: `pyproject.toml` (defines `pickinsta` console script, optional dependencies, ruff config)
- **Docs**: `docs/composition-rules.md` (technical scoring weights, cropping heuristics)

## Common Tasks

### Adding a New Technical Metric

1. Implement metric function in `ig_image_selector.py` (follow pattern: returns 0.0-1.0)
2. Add to `score_technical()` with appropriate weight (ensure weights sum to 1.0)
3. Update `docs/composition-rules.md` to document the metric

### Changing Claude Scoring Criteria

1. Edit `VISION_PROMPT_TEMPLATE` / `build_vision_prompt(...)` in `ig_image_selector.py`
2. Updated prompt hash will invalidate caches (intentional for consistency)
3. Ensure Claude returns JSON with expected keys: `subject_clarity`, `lighting`, `color_pop`, `emotion`, `scroll_stop`, `crop_4x5`, `total`, `one_line`

### Adjusting Crop Behavior

- Modify `_ideal_subject_x()` / `_ideal_subject_y()` for different placement preferences
- Change `_score_crop_candidate()` weights to prioritize different composition factors
- Toggle `use_yolo` parameter in `smart_crop()` to disable YOLO detection

### Supporting New Image Formats

Add extension to `SUPPORTED_EXTENSIONS` set (line 53). Ensure PIL can open the format.

## Troubleshooting

**CLIP model download fails**: First run requires internet to download ~1.7GB model from HuggingFace. Set `HF_TOKEN` in environment to avoid rate limits.

**Claude model not found**: Use `--claude-model` flag or set `ANTHROPIC_MODEL`/`CLAUDE_MODEL` in `.env`. Pipeline tries fallback models: preferred → undated alias → `claude-haiku-4-5-20251001` → `claude-3-5-sonnet-latest`.

**Ollama scoring fails/unavailable**: Verify `PICKINSTA_OLLAMA_BASE_URL` and `PICKINSTA_OLLAMA_MODEL`, then test with `curl <base_url>/api/tags`. Tune server (`OLLAMA_NUM_PARALLEL`, `OLLAMA_NUM_THREAD`) and client (`PICKINSTA_OLLAMA_CONCURRENCY`) together.

**YOLO not available**: Optional dependency. Install with `pip install -e '.[yolo]'`. Pipeline falls back to saliency detection automatically.

**Poor crop quality**: Enable debug mode (`debug=True` in `smart_crop()`) to visualize what's being detected. Check if YOLO is finding the right subject or if saliency fallback is being used.

## API Cost Management

Claude scoring costs ~$0.005/image (varies by model/prompt):
- Images are downsized to 1024px/q75 before sending to reduce token cost
- Cost estimate shown before scoring: `💰 Claude estimate: 23/42 images to score, ~$0.12 (19 cached)`
- Use `--vision-pct 0.5` to score only top 50% of technically-filtered images
- Use `--all` to score all images (more accurate but higher cost)
- Caching prevents re-scoring unchanged images (cache files: `*.pickinsta.json`)
- Cache is model-agnostic by default — switching models reuses cached scores unless `--rescore` is passed
- Adaptive concurrency throttles down on rate limits and scales up on success
- Consider CLIP pre-filtering for large batches (>100 images)
