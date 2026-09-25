# pickinsta Reference

**CLI entry point:** `src/pickinsta/cli.py`, which dispatches into the compatibility facade in
`src/pickinsta/ig_image_selector.py`, supported by the
`config`, `pipeline`, `vision`, `infrastructure`, `reporting`, and `telemetry` modules.

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
| `PICKINSTA_MAX_WORKERS` | Up to 8 logical CPUs (minimum 1) | Shared local-pool cap; range 1-256. Set explicitly to use more than the conservative default. |
| `PICKINSTA_PROCESS_WORKERS` | `PICKINSTA_MAX_WORKERS` | Resize and dedup feature-pool override; range 1-256 |
| `PICKINSTA_THREAD_WORKERS` | `PICKINSTA_MAX_WORKERS` | Technical, burst, and crop-pool override; range 1-256 |
| `PICKINSTA_OPENCV_THREADS` | `0` | OpenCV native threads; `0` preserves its default, range 0-256 |

---

## 3. Pipeline Stages

### Stage 0 — Resize

| Item | Value |
|------|-------|
| Input | Source images from input folder |
| Output | Resized JPEGs in work folder |
| Skip condition | Work image exists and mtime is current |
| Parallelism | `ProcessPoolExecutor` (PIL only, safe to fork) |

Rejects source images larger than 100 megapixels before decoding, then resizes the longest edge to max 1920px. Preserves EXIF data; resets orientation tag to 1 so downstream EXIF timestamps are available.

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
| Skip condition | Cache schema, work-image mtime, and algorithm/dependency fingerprint match |
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
| Output | Vision score per image; Claude cache: `<source_filename>.pickinsta.json` |
| Skip condition | For Claude, a valid cache identity exists; bypass with `--rescore` |

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
- **Cache:** CLIP does not currently use the Claude per-source cache described below.

#### Claude (`--scorer claude`)

- **Default model:** `claude-haiku-4-5-20251001`
- **Image preparation:** Downsized to 1024px / q75 JPEG before API call (reduces token cost).
- **Prompt:** `VISION_PROMPT_TEMPLATE` / `build_vision_prompt(...)` in `src/pickinsta/vision/prompts.py`. Includes YOLO detection context and `PICKINSTA_ACCOUNT_CONTEXT`.
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
- **Cache identity:** Vision cache schema v2 + source SHA256 + scorer + model + prompt
  SHA256 + scoring options. Scoring options include YOLO-context use, image edge and
  JPEG settings, whether the source or preprocessed image was scored, and the scored
  input SHA256.
- **Cache file:** `<source_filename>.pickinsta.json` next to the source file.

#### Ollama (`--scorer ollama`)

- **Default model:** `qwen2.5vl:7b`
- **Supported model families:** `qwen2.5vl` (Qwen 2.5 VL), `gemma4` (Gemma 4). Other models fall back to a generic JSON prompt.
- **Same 0–60 rubric as Claude.**
- **Concurrency:** Configured via `PICKINSTA_OLLAMA_CONCURRENCY` (default 2, range 1–16).
- **Reliability:** Retry/backoff (`PICKINSTA_OLLAMA_MAX_RETRIES`, `PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC`) + circuit breaker (`PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS` consecutive failures triggers fallback).
- **Cache:** Ollama results are not currently persisted in the Claude per-source cache.

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
| Vision (Claude, schema v2) | `<source_filename>.pickinsta.json` (next to source) | Source SHA256 + scorer + model + prompt SHA256 + scoring options | Any identity field changes, old/missing schema, or `--rescore` |
| Technical score (schema v1) | `<work_image>.jpg.techscore.json` (in work folder) | Work image mtime + technical algorithm/dependency fingerprint | Work image changes, cache schema changes, scoring implementation/weights change, or relevant dependency versions change |
| Stage 0 resize | Work image mtime vs source mtime | mtime comparison | Source file modified |

**`--rescore`:** Forces all vision caches to be ignored; does not affect technical score caches.

**Model and option switching:** Claude model identity and scoring/preprocessing options
are part of schema v2. Switching them invalidates the existing entry automatically.

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
| `run_manifest.json` | Output folder | Compatibility copy of the schema-versioned completion record |
| `current_run.json` | Output folder | Atomic pointer to the authoritative immutable snapshot under `.pickinsta-runs/` |

### Run Manifest Contract

`run_manifest.json` uses schema version 1. Its top-level fields are:

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | integer | Manifest schema; currently `1` |
| `status` | string | `complete` when no issue was recorded; otherwise `degraded` |
| `processed` | integer | Selected entries included in the reports |
| `skipped` | integer | Selected entries skipped because crop generation failed |
| `failed` | integer | Non-crop artifact operations that failed |
| `issues` | array | Failure records with `stage`, `artifact`, and `reason` |
| `input_folder` | string | Input folder used for the run |
| `output_folder` | string | Publication destination |
| `scorer` | string | Requested vision scorer |
| `top_n_requested` | integer | Requested selection count |
| `analyzed` | integer | Ranked candidates available before output generation |
| `artifacts` | array | Managed run artifacts staged for publication, excluding the manifest itself |
| `configuration` | object | Effective paths, mode/scorer options, and worker settings for this run |
| `stage_timings_seconds` | object | Cumulative wall-clock seconds per measured stage, rounded to six decimals |
| `stage_invocations` | object | Number of measured invocations contributing to each stage total |
| `caches` | object | Per-cache `hits`, `misses`, and `hit_ratio` for lookups performed during the run |
| `runtime` | object | Python/runtime, platform, logical CPU, peak process memory, and package-version evidence |
| `warnings` | array | Structured non-fatal warnings; output issues may also appear here |

The full-pipeline stage names are `resize`, `deduplication`, `technical_scoring`,
`burst_reevaluation`, `vision_scoring`, `output_generation`, and `reporting`.
Dedup-only runs omit `technical_scoring` and `vision_scoring`. A stage can be present
with zero useful items because the timed block was still invoked. Timings are
process-local wall-clock measurements, not the sum of worker CPU time. Repeated or
nested measurements accumulate under the same stage name, and invocation counts make
that aggregation explicit.

Cache metrics currently use the names `technical` and `vision`. A hit means a lookup
returned a reusable score; a miss means scoring was required. `hit_ratio` is
`hits / (hits + misses)`. Cache entries are omitted when no lookup was observed, so
absence does not mean a 0% hit rate. Technical lookups are recorded per candidate.
Vision lookup metrics depend on the selected scorer and cache-capable path.

Example (paths and versions are illustrative; no credentials are included):

```json
{
  "schema_version": 1,
  "status": "complete",
  "configuration": {
    "scorer": "claude",
    "workers": {"processes": 4, "threads": 8, "opencv_threads": 1},
    "anthropic_api_key": "[REDACTED]"
  },
  "stage_timings_seconds": {"resize": 1.25, "technical_scoring": 3.5},
  "stage_invocations": {"resize": 1, "technical_scoring": 1},
  "caches": {
    "technical": {"hits": 8, "misses": 2, "hit_ratio": 0.8}
  },
  "runtime": {
    "python": "3.14.0",
    "implementation": "CPython",
    "platform": "macOS-example",
    "cpu_count": 8,
    "peak_memory_bytes": 268435456,
    "packages": {"pickinsta": "0.2.0", "Pillow": "12.3.0"}
  },
  "warnings": []
}
```

Manifest serialization recursively redacts values whose normalized key is, or ends
with, `apikey`, `authorization`, `cookie`, `password`, `secret`, or `token`. For
example, `ANTHROPIC_API_KEY`, `access-token`, and nested `password` values become
`[REDACTED]`. This is a defense-in-depth measure, not a reason to add secrets to run
configuration or warning records.

Known issue stages are `crop`, `copy_hd`, and `copy_full`. A crop issue increments
`skipped`; copy issues increment `failed`. A selected entry with a copy issue can still
appear in the report when its other artifacts were produced, so the three counts are
operational counters rather than a partition of one total.

All managed artifacts are created in a temporary `.pickinsta-run-*` directory inside
the output folder. Publication copies the complete set into `.pickinsta-runs/`, updates
flat compatibility files, then atomically advances `current_run.json`. Resolve the
pointer to read a consistent snapshot after interrupted publication. Files not managed
by the run are preserved.

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

### Local worker configuration

Worker settings are resolved once at pipeline startup. `PICKINSTA_PROCESS_WORKERS`
and `PICKINSTA_THREAD_WORKERS` override `PICKINSTA_MAX_WORKERS` for their pool type;
otherwise both inherit the shared cap. The shared default is the logical CPU count,
falling back to 4 when it cannot be detected. Every pool uses the smaller of its item
count and configured cap. Values must be integers from 1 through 256. Invalid values
fall back to the applicable default and produce a configuration warning.

`PICKINSTA_OPENCV_THREADS` independently controls OpenCV's process-global native
thread pool. Its default `0` leaves the OpenCV default unchanged and appears as `null`
in `configuration.workers.opencv_threads`; values from 1 through 256 call
`cv2.setNumThreads(...)` once before local stages run. When Python thread pools perform
OpenCV work, setting this to `1` usually avoids nested oversubscription. These local
limits do not override Claude's adaptive concurrency or
`PICKINSTA_OLLAMA_CONCURRENCY`.
