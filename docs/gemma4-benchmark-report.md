# Model Benchmark Report (Speed + Quality)

- Generated: `2026-04-08T22:18:30`
- Input folder: `/Users/renatobo/development/pickinsta/input`
- Candidates scored per run: `21`
- Runs per variant: `1`
- Warmup enabled: `True` (1 image + 10s wait)
- Ollama base URL: `http://localhost:11434`
- Ollama concurrency: `2`
- Ollama max retries: `2`
- Ollama keep_alive: `15m`

## Findings

> **Note:** `qwen2.5vl:7b` was not installed on the local Ollama server during this run, so all qwen variants show 100% failures and their speed numbers reflect the circuit-breaker fallback path (not real scoring). Gemma 4 e4b data is valid.

### gemma4:e4b — What worked
- **Zero failures** across both YOLO modes (21/21 images scored).
- Structured JSON output was returned and parsed successfully.
- Valid JSON criteria fields populated with numeric values.

### gemma4:e4b — Issues found

**1. Quantized / degenerate scoring** *(resolved in v1.1.0 — root cause was the Ollama format schema, not model capacity)*

Most images received identical scores across all 6 criteria (e.g., all 9s → total 54, or all 5s → total 30). Three separate prompt variants were tested (2026-04-09) and all collapsed. At the time, this was diagnosed as a model capacity issue. **That diagnosis was wrong.**

Root cause (identified 2026-04-10): the Ollama `format` schema parameter silently forces Gemma 4 to collapse all criteria to a single tier regardless of prompt content. Fix: remove the `format` parameter for all Gemma 4 models. After the fix, SDI rose from 0.00 to 0.17–0.19 on the same 4B model with no prompt changes. See `docs/gemma4-sdi-report.md` for full results.

**2. Thinking chain bleedthrough in `one_line`** *(mitigated in code)*

The `one_line` field contained `"Here's a thinking process to arrive at the desired JSON output: 1."` — the model's internal reasoning prefix leaking into the JSON field value. **Fixed** in `_sanitize_vision_one_line()` via `_THINKING_PREAMBLE_RE`: the preamble is detected and stripped; if nothing useful remains, the field falls back to `"Vision scoring summary"`.

**3. Throughput**
- ~26–28 sec/img (YOLO off), ~24–25 sec/img (YOLO on) — ~2.2–2.4 imgs/min
- YOLO on/off difference is within single-run variance noise.

### Recommendations

> **Updated 2026-04-10:** The tier-collapse finding was caused by the Ollama format schema bug, not model capacity. `gemma4:e4b` is usable after the v1.1.0 fix. See `docs/gemma4-sdi-report.md` for updated benchmarks.

- ~~**Do not use `gemma4:e4b` as primary scorer.**~~ **`gemma4:e4b` with `claude+system` is viable** (SDI 0.19, 19 unique score tuples) after the format-schema fix.
- ~~**Try `gemma4:12b` or `gemma4:27b`**~~ — the 4B model is sufficient once the schema parameter is removed. Larger variants are worth trying if available but not required.
- **Do not use `gemma4:e4b-it-q8_0`** — 48% slower with half the SDI on this hardware. No benefit over the base model.
- **`qwen2.5vl:7b` remains the recommended Ollama model.** Re-run this benchmark with it installed for a direct comparison.
- The `one_line` preamble-strip fix benefits all models that leak thinking text, not just Gemma 4.

---

## Speed Summary

| Variant | Scorer | Model | YOLO | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | Speed vs fastest |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen2.5vl:7b \| yolo=off | ollama | qwen2.5vl:7b | off | 0.01 | 6206.87 | 0.20 | 21.00 | 1.00x |
| qwen2.5vl:7b \| yolo=on | ollama | qwen2.5vl:7b | on | 0.02 | 3488.76 | 0.36 | 21.00 | 1.78x |
| gemma4:e4b \| yolo=on | ollama | gemma4:e4b | on | 24.50 | 2.45 | 514.42 | 0.00 | 2534.06x |
| gemma4:e4b \| yolo=off | ollama | gemma4:e4b | off | 26.47 | 2.27 | 555.85 | 0.00 | 2738.16x |

## Per-run Timing

| Variant | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| gemma4:e4b \| yolo=off | 1 | 555.85 | 26.47 | 2.27 | 0 |
| qwen2.5vl:7b \| yolo=off | 1 | 0.20 | 0.01 | 6206.87 | 21 |
| gemma4:e4b \| yolo=on | 1 | 514.42 | 24.50 | 2.45 | 0 |
| qwen2.5vl:7b \| yolo=on | 1 | 0.36 | 0.02 | 3488.76 | 21 |

## Image-by-Image Score Comparison (Run 1)

| Image | gemma4:e4b \| yolo=off final | gemma4:e4b \| yolo=off rank | qwen2.5vl:7b \| yolo=off final | qwen2.5vl:7b \| yolo=off rank | gemma4:e4b \| yolo=on final | gemma4:e4b \| yolo=on rank | qwen2.5vl:7b \| yolo=on final | qwen2.5vl:7b \| yolo=on rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 5 | 0.2017 | 9 | 0.1931 | 12 | 0.2017 | 9 |
| 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 15 | 0.2007 | 10 | 0.7607 | 2 | 0.2007 | 10 |
| 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 20 | 0.1756 | 20 | 0.1840 | 19 | 0.1756 | 20 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 6 | 0.1857 | 17 | 0.1875 | 17 | 0.1857 | 17 |
| AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8676 | 1 | 0.2376 | 1 | 0.2056 | 5 | 0.2376 | 1 |
| AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.8475 | 3 | 0.2175 | 8 | 0.1986 | 11 | 0.2175 | 8 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 10 | 0.2348 | 2 | 0.2047 | 6 | 0.2348 | 2 |
| C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 12 | 0.2298 | 4 | 0.2029 | 8 | 0.2298 | 4 |
| DSC-5436-NaraMedia.jpeg | 0.1885 | 18 | 0.1886 | 16 | 0.3372 | 4 | 0.1886 | 16 |
| IG cali_carnivores - DSC00013.jpg | 0.2002 | 14 | 0.2220 | 7 | 0.0648 | 21 | 0.2220 | 7 |
| IG cali_carnivores - DSC09850.jpg | 0.2024 | 13 | 0.2284 | 5 | 0.2024 | 9 | 0.2284 | 5 |
| IG cali_carnivores - DSC09857.jpg | 0.8076 | 4 | 0.1776 | 19 | 0.7376 | 3 | 0.1776 | 19 |
| IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 19 | 0.1851 | 18 | 0.1873 | 18 | 0.1851 | 18 |
| IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 17 | 0.1912 | 13 | 0.1894 | 14 | 0.1912 | 13 |
| IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0600 | 21 | 0.1901 | 15 | 0.1891 | 16 | 0.1901 | 15 |
| IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 11 | 0.2334 | 3 | 0.2042 | 7 | 0.2334 | 3 |
| IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 16 | 0.1951 | 11 | 0.8251 | 1 | 0.1951 | 11 |
| IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.8571 | 2 | 0.2271 | 6 | 0.2020 | 10 | 0.2271 | 6 |
| IG renatobo - IMG_5013.jpeg | 0.7345 | 7 | 0.1745 | 21 | 0.1836 | 20 | 0.1745 | 21 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.5446 | 9 | 0.1908 | 14 | 0.1893 | 15 | 0.1908 | 14 |
| caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.5476 | 8 | 0.1945 | 12 | 0.1906 | 13 | 0.1945 | 12 |

## Ranked Quality Details (All Images)

_Each section below is from timed run 1 for that variant._

### gemma4:e4b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.8676 | 0.7919 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 2 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.8571 | 0.7571 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 3 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.8475 | 0.7249 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 4 | IG cali_carnivores - DSC09857.jpg | 0.8076 | 0.5921 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 5 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.7617 | 0.6723 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 6 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.7457 | 0.6190 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 7 | IG renatobo - IMG_5013.jpeg | 0.7345 | 0.5817 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 8 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.5476 | 0.6484 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 9 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.5446 | 0.6360 | 42 | 7 | 7 | 7 | 7 | 7 | 7 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 10 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 11 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 12 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 13 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 14 | IG cali_carnivores - DSC00013.jpg | 0.2002 | 0.7401 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 15 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.1927 | 0.6688 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 16 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1908 | 0.6504 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 17 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 18 | DSC-5436-NaraMedia.jpeg | 0.1885 | 0.6287 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 19 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 20 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 21 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.0600 | 0.6338 | 18 | 3 | 3 | 3 | 3 | 3 | 3 | no | Here's a thinking process to arrive at the desired JSON output: 1. |

### qwen2.5vl:7b | yolo=off

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.2376 | 0.7919 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2348 | 0.7827 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 3 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2334 | 0.7780 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 4 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2298 | 0.7661 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 5 | IG cali_carnivores - DSC09850.jpg | 0.2284 | 0.7612 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 6 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2271 | 0.7571 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 7 | IG cali_carnivores - DSC00013.jpg | 0.2220 | 0.7401 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.2175 | 0.7249 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 9 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2017 | 0.6723 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 10 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2007 | 0.6688 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 11 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1951 | 0.6504 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 12 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1945 | 0.6484 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 13 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1912 | 0.6374 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 14 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1908 | 0.6360 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 15 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1901 | 0.6338 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 16 | DSC-5436-NaraMedia.jpeg | 0.1886 | 0.6287 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 17 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1857 | 0.6190 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 18 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1851 | 0.6171 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 19 | IG cali_carnivores - DSC09857.jpg | 0.1776 | 0.5921 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 20 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1756 | 0.5853 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 21 | IG renatobo - IMG_5013.jpeg | 0.1745 | 0.5817 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |

### gemma4:e4b | yolo=on

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.8251 | 0.6504 | 54 | 9 | 9 | 9 | 9 | 9 | 9 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 2 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.7607 | 0.6688 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 3 | IG cali_carnivores - DSC09857.jpg | 0.7376 | 0.5921 | 48 | 8 | 8 | 8 | 8 | 8 | 8 | no | Here's a thinking process to construct the JSON review: 1. |
| 4 | DSC-5436-NaraMedia.jpeg | 0.3372 | 0.6287 | 32 | 10 | 4 | 4 | 4 | 4 | 6 | no | ** One concise sentence. |
| 5 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.2056 | 0.7919 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 6 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2047 | 0.7827 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 7 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2042 | 0.7780 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 8 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2029 | 0.7661 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 9 | IG cali_carnivores - DSC09850.jpg | 0.2024 | 0.7612 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 10 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2020 | 0.7571 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 11 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.1986 | 0.7249 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 12 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.1931 | 0.6723 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 13 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1906 | 0.6484 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 14 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1894 | 0.6374 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 15 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1893 | 0.6360 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 16 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1891 | 0.6338 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 17 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1875 | 0.6190 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 18 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1873 | 0.6171 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 19 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1840 | 0.5853 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 20 | IG renatobo - IMG_5013.jpeg | 0.1836 | 0.5817 | 30 | 5 | 5 | 5 | 5 | 5 | 5 | no | Here's a thinking process to arrive at the desired JSON output: 1. |
| 21 | IG cali_carnivores - DSC00013.jpg | 0.0648 | 0.7401 | 18 | 2 | 4 | 4 | 4 | 4 | 0 | no | Here's a thinking process to arrive at the desired JSON output: 1. |

### qwen2.5vl:7b | yolo=on

| Rank | Image | Final | Tech | Vision | Subject | Light | Color | Emotion | Scroll | Crop | Failed | One line |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | AB Group - Session 1 (Turn 16 Entry) - CVR_0230_Mar1025_817AM_CaliPhoto.jpg | 0.2376 | 0.7919 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 2 | C Group - Session 2 Back Straight Speed Pans - CVR_2628_Mar1025_943AM_CaliPhoto.jpg | 0.2348 | 0.7827 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 3 | IG naramedia_official - DSC-5365-NaraMedia.jpeg | 0.2334 | 0.7780 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 4 | C Group - Session 2 Back Straight Speed Pans - CVR_2850_Mar1025_946AM_CaliPhoto.jpg | 0.2298 | 0.7661 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 5 | IG cali_carnivores - DSC09850.jpg | 0.2284 | 0.7612 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 6 | IG naramedia_official - DSC-5992-NaraMedia.jpeg | 0.2271 | 0.7571 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 7 | IG cali_carnivores - DSC00013.jpg | 0.2220 | 0.7401 |  |  |  |  |  |  |  | yes | Vision scoring failed — ranked by technical score only |
| 8 | AB Group - Session 4 (Turn 16) - CVR_6545_Mar1025_1122AM_CaliPhoto.jpg | 0.2175 | 0.7249 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 9 | 1-Around the Pits - CVR_0098_Mar1025_802AM_CaliPhoto.jpg | 0.2017 | 0.6723 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 10 | 1-Around the Pits - CVR_0792_Mar1025_832AM_CaliPhoto.jpg | 0.2007 | 0.6688 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 11 | IG naramedia_official - DSC-5897-NaraMedia.jpeg | 0.1951 | 0.6504 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 12 | caliphotovideo - Group picture - DROC Track Day March 10 202500002.jpg | 0.1945 | 0.6484 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 13 | IG luckie.moto - 20250310-075123-luckie.moto.jpg | 0.1912 | 0.6374 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 14 | caliphotovideo - Group picture - DROC Track Day March 10 202500001.jpg | 0.1908 | 0.6360 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 15 | IG m92663m - IMG-6019-Mark Momot.jpeg | 0.1901 | 0.6338 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 16 | DSC-5436-NaraMedia.jpeg | 0.1886 | 0.6287 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 17 | AB Group - Session 1 (Turn 16 Entry) - CVR_0148_Mar1025_815AM_CaliPhoto.jpg | 0.1857 | 0.6190 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 18 | IG kamiumitv - IMG-3981-Bao.jpeg | 0.1851 | 0.6171 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 19 | IG cali_carnivores - DSC09857.jpg | 0.1776 | 0.5921 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 20 | 2-Group Photo - Raffle - CVR_7677_Mar1025_1249PM_CaliPhoto.jpg | 0.1756 | 0.5853 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
| 21 | IG renatobo - IMG_5013.jpeg | 0.1745 | 0.5817 |  |  |  |  |  |  |  | yes | Ollama circuit breaker active — ranked by technical score only |
