# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pickinsta** is an Instagram image selection pipeline for motorcycle/Ducati photography. It processes event photo dumps into Instagram-ready cover candidates (1080x1440 portrait) by scoring technical quality and aesthetic potential.

## Architecture

### 5-Stage Pipeline

The core pipeline in `src/pickinsta/ig_image_selector.py` processes images through:

1. **Stage 0 - Resize**: Resize to max 1920px (saves compute, preserves EXIF orientation)
2. **Stage 1 - Deduplicate**: Perceptual hashing (imagehash) removes near-duplicates
3. **Stage 2 - Technical Scoring**: OpenCV-based quality metrics (sharpness, lighting, composition, color harmony)
4. **Stage 3 - Vision Scoring**: CLIP (local/free) or Claude API (best quality) for aesthetic evaluation
5. **Stage 4 - Smart Crop**: YOLO-guided crop to 1080x1440 following composition rules

### Key Design Decisions

**YOLO Integration (Recent Enhancement)**:
- YOLOv8 detects subjects (motorcycles, people, vehicles) before cropping
- Ensures crops keep the full subject in frame (previously used unreliable saliency detection)
- YOLO context is passed to Claude to improve scoring accuracy
- Graceful fallback to saliency detection if YOLO finds nothing
- See `YOLO_CLAUDE_INTEGRATION.md` for details

**Dual Scorer Architecture**:
- **CLIP** (`--scorer clip`): Free, local, zero-shot classification. Uses 4 positive + 2 negative prompts. Maps logits to 0-60 scale to match Claude's range.
- **Claude** (`--scorer claude`): Best quality, API costs ~$0.50/100 images. Scores 6 criteria (subject_clarity, lighting, color_pop, emotion, scroll_stop, crop_4x5). Returns JSON with scores + one-line summary.

**Final Score Calculation**: `final_score = 0.3 * technical_composite + 0.7 * vision_normalized`

**Caching Strategy**: Claude responses cached per original source file as `<filename>.pickinsta.json`. Cache includes image SHA256 + model + prompt hash for validity checking.

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

### Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base + both scorers
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[clip,claude]"

# Or install individual scorers
python -m pip install -e ".[clip]"    # CLIP only
python -m pip install -e ".[claude]"  # Claude only
python -m pip install -e ".[yolo]"    # YOLO support (recommended)

# Verify installation
python -c "import transformers, torch, anthropic; print(transformers.__version__, torch.__version__, anthropic.__version__)"
```

### Running the Pipeline

```bash
# Basic usage (CLIP scorer, top 10)
pickinsta ./input --output ./selected --top 10 --scorer clip

# Claude scorer with all images analyzed
pickinsta ./input --output ./selected --scorer claude --all

# As module
python -m pickinsta ./input --scorer claude --top 5

# Override Claude model
pickinsta ./input --scorer claude --claude-model claude-sonnet-4-5
```

### Environment Setup

Create `.env` file for Claude API key:

```bash
cp .env.example .env
# Edit .env and set:
ANTHROPIC_API_KEY=your_key_here
# Optional:
HF_TOKEN=hf_xxx  # Reduces HuggingFace warnings
ANTHROPIC_MODEL=claude-sonnet-4-5  # Override default model
```

`.env` search order: current environment → cwd/.env → input_folder/.env

### Testing

```bash
# Run all tests
pytest

# Test smart cropping with debug visualization
python tests/test_crop.py

# Test full integration (YOLO + cropping)
python tests/test_full_integration.py

# Debug YOLO + Claude prompt enhancement (manual debug script)
python debug/debug_yolo_claude.py
```

Debug mode creates visualizations showing:
- Green boxes: YOLO detections
- Red dots: Subject center points
- Yellow lines: Rule-of-thirds grid
- Cyan lines: Phi Grid

## Important File Locations

- **Main entry**: `src/pickinsta/ig_image_selector.py` (1700+ lines, all pipeline logic)
- **Package init**: `src/pickinsta/__init__.py` (exports `__version__`)
- **CLI entry**: `src/pickinsta/__main__.py` (calls main())
- **Config**: `pyproject.toml` (defines `pickinsta` console script, optional dependencies)
- **Tests**: `tests/` (pytest-based, includes integration tests)
- **Docs**: `YOLO_CLAUDE_INTEGRATION.md`, `ENHANCEMENT_SUMMARY.md`, `docs/composition-rules.md`

## Common Tasks

### Adding a New Technical Metric

1. Implement metric function in `ig_image_selector.py` (follow pattern: returns 0.0-1.0)
2. Add to `score_technical()` with appropriate weight (ensure weights sum to 1.0)
3. Update `docs/composition-rules.md` to document the metric

### Changing Claude Scoring Criteria

1. Edit `VISION_PROMPT` constant in `ig_image_selector.py`
2. Update prompt hash will invalidate caches (intentional for consistency)
3. Ensure Claude returns JSON with expected keys: `subject_clarity`, `lighting`, `color_pop`, `emotion`, `scroll_stop`, `crop_4x5`, `total`, `one_line`

### Adjusting Crop Behavior

- Modify `_ideal_subject_x()` / `_ideal_subject_y()` for different placement preferences
- Change `_score_crop_candidate()` weights to prioritize different composition factors
- Toggle `use_yolo` parameter in `smart_crop()` to disable YOLO detection

### Supporting New Image Formats

Add extension to `SUPPORTED_EXTENSIONS` set (line 51). Ensure PIL can open the format.

## Troubleshooting

**CLIP model download fails**: First run requires internet to download ~1.7GB model from HuggingFace. Set `HF_TOKEN` in environment to avoid rate limits.

**Claude model not found**: Use `--claude-model` flag or set `ANTHROPIC_MODEL` in .env. Pipeline tries fallback models: preferred → undated alias → `claude-sonnet-4-5` → `claude-3-5-sonnet-latest`.

**YOLO not available**: Optional dependency. Install with `pip install -e '.[yolo]'`. Pipeline falls back to saliency detection automatically.

**Poor crop quality**: Enable debug mode (`debug=True` in `smart_crop()`) to visualize what's being detected. Check if YOLO is finding the right subject or if saliency fallback is being used.

## API Cost Management

Claude scoring costs ~$0.005/image (varies by model/prompt):
- Use `--vision-pct 0.5` to score only top 50% of technically-filtered images
- Use `--all` to score all images (more accurate but higher cost)
- Caching prevents re-scoring unchanged images (cache files: `*.pickinsta.json`)
- Consider CLIP pre-filtering for large batches (>100 images)
