# Ollama Model Speed Benchmark Report

- Generated: `2026-02-22T12:47:35`
- Candidates scored per run: `42`
- Runs per model: `1`
- Warmup run before timed runs: `True`
- YOLO context enabled: `False`
- Ollama base URL: `http://localhost:11434`
- Concurrency: `2`
- Max retries: `2`
- Keep alive: `15m`

## Summary

| Model | Avg sec/img | Avg imgs/min | Avg duration (s) | Avg failures/run | Speed vs fastest |
|---|---:|---:|---:|---:|---:|
| blaifa/InternVL3_5:4B | 10.23 | 5.87 | 429.53 | 0.00 | 1.00x |
| blaifa/InternVL3_5:8b | 18.47 | 3.25 | 775.87 | 0.00 | 1.81x |
| openbmb/minicpm-v4.5:8b | 21.17 | 2.83 | 889.19 | 0.00 | 2.07x |
| qwen3-vl:8b | 33.96 | 1.77 | 1426.23 | 0.00 | 3.32x |

## Per-run Details

| Model | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| qwen3-vl:8b | 1 | 1426.23 | 33.96 | 1.77 | 0 |
| blaifa/InternVL3_5:8b | 1 | 775.87 | 18.47 | 3.25 | 0 |
| blaifa/InternVL3_5:4B | 1 | 429.53 | 10.23 | 5.87 | 0 |
| openbmb/minicpm-v4.5:8b | 1 | 889.19 | 21.17 | 2.83 | 0 |
