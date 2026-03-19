# Ollama Model Speed Benchmark Report

- Generated: `2026-02-22T23:39:26`
- Input folder: `/home/renatobo/pickinsta/input`
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
| blaifa/InternVL3_5:4B | 29.54 | 2.03 | 1240.89 | 0.00 | 1.00x |
| blaifa/InternVL3_5:8b | 47.09 | 1.27 | 1977.66 | 0.00 | 1.59x |
| openbmb/minicpm-v4.5:8b | 79.36 | 0.76 | 3333.28 | 0.00 | 2.69x |
| qwen3-vl:8b | 168.54 | 0.36 | 7078.74 | 3.00 | 5.70x |

## Per-run Details

| Model | Run | Duration (s) | Sec/img | Imgs/min | Failures |
|---|---:|---:|---:|---:|---:|
| qwen3-vl:8b | 1 | 7078.74 | 168.54 | 0.36 | 3 |
| blaifa/InternVL3_5:8b | 1 | 1977.66 | 47.09 | 1.27 | 0 |
| blaifa/InternVL3_5:4B | 1 | 1240.89 | 29.54 | 2.03 | 0 |
| openbmb/minicpm-v4.5:8b | 1 | 3333.28 | 79.36 | 0.76 | 0 |
