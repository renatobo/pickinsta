# Plan: Fix Gemma 4 Tier-Collapsed Scoring

## Context

`gemma4:e4b` produces tier-collapsed scores — all 6 vision criteria get the same integer per image (e.g., all 9s, all 5s). Three prompt tuning attempts already failed. This plan systematically tests three axes to fix the problem: richer prompt, system prompt, and larger model size.

**Branch**: `feat/gemma4-support` (committed at d347b1d)

## Success Metric: Score Differentiation Index (SDI)

Per image: standard deviation of its 6 criterion scores. Average across all images in the run.

| SDI | Meaning |
|-----|---------|
| 0.0 | Complete tier-collapse (current gemma4:e4b state) |
| ≥ 1.0 | Minimal differentiation |
| ≥ 1.5 | Moderate (meaningful criterion variance) |
| ≥ 2.0 | Good (comparable to Claude) |

Secondary metric: **unique score tuples** — how many distinct (s1,s2,s3,s4,s5,s6) combos across images. Currently ~4-5; target ≥15 out of 21.

## Test Matrix (ordered fail-fast, cheapest first)

| Run | Model | Prompt | System Msg | ~Time | Purpose |
|-----|-------|--------|------------|-------|---------|
| 1 | gemma4:e4b | Claude-rich | No | 9 min | Does richer prompt break tier-collapse? |
| 2 | gemma4:e4b | Default | Yes | 9 min | Does system prompt alone help? |
| 3 | gemma4:e4b | Claude-rich | Yes | 9 min | Combined — only if run 1 or 2 show partial improvement |
| 4 | gemma4:12b | Default | No | 15-20 min | Is it purely a model capacity issue? |
| 5 | gemma4:12b | Best from 1-3 | TBD | 15-20 min | Only if run 4 improves, combine with best prompt |

**Decision points:**
- After run 1: SDI ≥ 1.5 → rich prompt works, skip 2-3, go to 4
- After run 2: SDI ≥ 1.5 → system prompt works, skip 3
- Runs 1+2 both SDI < 0.5 → skip 3 (combining two ineffective axes won't help), go to 4
- Run 4 SDI ≥ 1.5 with default prompt → problem is purely model capacity, done

## Implementation

### Step 1: Add `PICKINSTA_OLLAMA_PROMPT_VARIANT` env var `[S]`

**File**: `src/pickinsta/ig_image_selector.py`

Add env var constant (near line 78) and resolver (near line 325):

```python
OLLAMA_PROMPT_VARIANT_ENV_VAR = "PICKINSTA_OLLAMA_PROMPT_VARIANT"

def resolve_ollama_prompt_variant() -> str:
    return (os.environ.get(OLLAMA_PROMPT_VARIANT_ENV_VAR) or "").strip().lower() or "default"
```

Values: `default`, `claude`, `system`, `claude+system`

### Step 2: Add system prompt constant `[S]`

**File**: `ig_image_selector.py` (near line 1420, after existing prompts)

```python
OLLAMA_SYSTEM_PROMPT = (
    "You are a professional motorsport photographer and Instagram content curator. "
    "You evaluate motorcycle photos for visual quality, composition, and social media impact. "
    "You always score each criterion independently based on what you observe — "
    "different aspects of a photo can have very different quality levels."
)
```

### Step 3: Modify prompt routing in `score_with_ollama()` `[M]`

**File**: `ig_image_selector.py`, lines 1971-1984

Currently Gemma 4 always uses `build_ollama_gemma4_prompt()`. Change to check `resolve_ollama_prompt_variant()`:

- `default` or `system` → current `build_ollama_gemma4_prompt(account_context)`
- `claude` or `claude+system` → `build_vision_prompt(account_context)` (the Claude template)

Apply same logic to the retry path (lines 1996-1998).

### Step 4: Inject system message in `_send_ollama_request()` `[M]`

**File**: `ig_image_selector.py`, line 1945

When variant includes `system`, prepend to messages array:

```python
messages = []
if variant in ("system", "claude+system") and _is_gemma4:
    messages.append({"role": "system", "content": OLLAMA_SYSTEM_PROMPT})
messages.append({"role": "user", "content": active_prompt, "images": [image_data]})
```

### Step 5: Bump num_predict for Claude variant `[S]`

**File**: `ig_image_selector.py`, `_resolve_ollama_num_predict()` (line 1513)

When Gemma 4 + `claude` variant: return 350 (vs 280 default). Richer prompt may elicit longer `one_line`.

### Step 6: Add SDI metric to benchmark report `[M]`

**File**: `tests/benchmarks/benchmark_ollama_models.py`

- Add `_compute_sdi()` helper: mean stdev of 6 criteria across images
- Add `_count_unique_tuples()` helper
- Add `--prompt-variant` CLI arg (choices: default/claude/system/claude+system)
- Pass variant to env vars in `_benchmark_variant()`
- Add SDI + unique tuples columns to Speed Summary table
- Add `prompt_variant` field to `BenchmarkVariant` dataclass

### Step 7: Pull gemma4:12b model `[prerequisite for run 4]`

```bash
ollama pull gemma4:12b
```

## Key Files

| File | Changes |
|------|---------|
| `src/pickinsta/ig_image_selector.py` | Env var, system prompt, prompt routing, num_predict |
| `tests/benchmarks/benchmark_ollama_models.py` | SDI metric, --prompt-variant flag, report columns |
| `CLAUDE.md` | Document new env var |

## Verification

1. `make check` — lint + tests pass
2. Run test matrix (runs 1-5 as needed, with decision points)
3. Compare SDI and unique tuples across runs
4. Update `docs/gemma4-benchmark-report.md` with findings

## Benchmark Commands

```bash
# Run 1: Claude-rich prompt, no system msg
PICKINSTA_OLLAMA_PROMPT_VARIANT=claude \
python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input --models gemma4:e4b --variants off --runs 1 \
  --prompt-variant claude --report docs/gemma4-run1-claude-prompt.md

# Run 2: Default prompt + system msg
PICKINSTA_OLLAMA_PROMPT_VARIANT=system \
python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input --models gemma4:e4b --variants off --runs 1 \
  --prompt-variant system --report docs/gemma4-run2-system-prompt.md

# Run 3: Claude + system (only if run 1 or 2 partially worked)
PICKINSTA_OLLAMA_PROMPT_VARIANT=claude+system \
python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input --models gemma4:e4b --variants off --runs 1 \
  --prompt-variant claude+system --report docs/gemma4-run3-combined.md

# Run 4: Larger model, default prompt
python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input --models gemma4:12b --variants off --runs 1 \
  --report docs/gemma4-run4-12b-default.md

# Run 5: Larger model + best prompt variant (TBD based on results)
```
