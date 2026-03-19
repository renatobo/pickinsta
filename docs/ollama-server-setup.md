# Ollama Server Setup

This guide covers the practical setup needed to use `pickinsta --scorer ollama` against a local or remote Ollama host.

Use the official Ollama docs for platform-specific installation details and this page for the `pickinsta` integration layer.

## Official References

- Docs home: <https://docs.ollama.com/>
- Quickstart: <https://docs.ollama.com/quickstart>
- macOS install: <https://docs.ollama.com/macos>
- Linux install: <https://docs.ollama.com/linux>
- FAQ: <https://docs.ollama.com/faq>
- Modelfile reference: <https://docs.ollama.com/modelfile>
- Importing models: <https://docs.ollama.com/import>

Example vision model pages used by this repo:

- <https://ollama.com/library/qwen3-vl:8b>
- <https://ollama.com/blaifa/InternVL3_5>
- <https://ollama.com/openbmb/minicpm-v4.5>

## 1. Install Ollama

### macOS

Follow the official macOS instructions, then verify:

```bash
ollama -v
```

### Linux

Follow the official Linux instructions, then verify:

```bash
ollama -v
```

## 2. Start and Verify the Server

### macOS

If using the app-based install, start the Ollama app and verify:

```bash
curl http://127.0.0.1:11434/api/tags
```

To expose the server on your LAN:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

See the Ollama FAQ for the full persistence story on macOS.

### Linux

Typical service flow:

```bash
sudo systemctl start ollama
sudo systemctl status ollama
curl http://127.0.0.1:11434/api/tags
```

To expose the server on your LAN, create a systemd override:

```bash
sudo systemctl edit ollama.service
```

Add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## 3. Pull a Model

The default `pickinsta` Ollama model is:

```text
qwen2.5vl:7b
```

Pull it with:

```bash
ollama pull qwen2.5vl:7b
```

Verify:

```bash
ollama list
```

For comparison testing, common model tags in this repo are:

```bash
ollama pull qwen3-vl:8b
ollama pull blaifa/InternVL3_5:8b
ollama pull blaifa/InternVL3_5:4B
ollama pull openbmb/minicpm-v4.5:8b
```

If namespaced tags resolve differently on your host, use the exact local names shown by `ollama list`.

## 4. Point pickinsta at the Server

Set environment variables in `.env` or your shell:

```bash
PICKINSTA_OLLAMA_BASE_URL=http://YOUR_SERVER_IP:11434
PICKINSTA_OLLAMA_MODEL=qwen2.5vl:7b
```

Optional client-side tuning:

```bash
PICKINSTA_OLLAMA_TIMEOUT_SEC=300
PICKINSTA_OLLAMA_MAX_IMAGE_EDGE=1024
PICKINSTA_OLLAMA_JPEG_QUALITY=80
PICKINSTA_OLLAMA_KEEP_ALIVE=10m
PICKINSTA_OLLAMA_USE_YOLO_CONTEXT=false
PICKINSTA_OLLAMA_CONCURRENCY=2
PICKINSTA_OLLAMA_MAX_RETRIES=2
PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC=0.75
PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS=6
```

Then run:

```bash
pickinsta ./input --output ./selected --scorer ollama --all
```

## 5. Smoke Test the Remote Host

Quick API connectivity test:

```bash
curl http://<SERVER_IP>:11434/api/generate \
  -d '{"model":"qwen2.5vl:7b","prompt":"Hello","stream":false}'
```

If this fails, fix server reachability before debugging `pickinsta`.

## 6. Tune Throughput

Start conservatively. Measure actual throughput in seconds per image rather than optimizing for raw CPU utilization.

Linux `systemd` example:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_NUM_THREAD=3"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

Apply:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Then tune the client side:

```bash
export PICKINSTA_OLLAMA_CONCURRENCY=4
```

Practical notes:

- Increase server and client concurrency together.
- If latency spikes or failures rise, back concurrency down.
- `PICKINSTA_OLLAMA_USE_YOLO_CONTEXT=true` can improve quality but usually adds cost and latency.
- `PICKINSTA_OLLAMA_KEEP_ALIVE=10m` helps when the same model is used repeatedly in one batch.

## 7. Benchmark Model Speed

Run the multi-model benchmark from the project root:

```bash
.venv/bin/python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input \
  --all \
  --runs 3 \
  --models qwen3-vl:8b blaifa/InternVL3_5:8b blaifa/InternVL3_5:4B openbmb/minicpm-v4.5:8b \
  --report docs/ollama-model-speed-benchmark-report.md
```

This produces a Markdown report comparing:

- average seconds per image
- average images per minute
- average total duration
- failures per run

If your local tags differ, replace the `--models` values with the exact names from `ollama list`.

## 8. Benchmark With and Without YOLO Context

```bash
.venv/bin/python tests/benchmarks/benchmark_ollama_yolo.py \
  --input ./input \
  --runs 2 \
  --all \
  --report docs/ollama-yolo-benchmark-report.md
```

Use this when you want to measure the quality or latency impact of attaching YOLO subject context to the prompt.

## 9. Save Full Per-Model Output Artifacts

If you want each model run to produce its own selected outputs and reports, loop over models and write to separate output folders. Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="./input"
RUNS=2
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="./model_compare_${STAMP}"
mkdir -p "${OUT_ROOT}"

MODELS=(
  "qwen3-vl:8b"
  "blaifa/InternVL3_5:8b"
  "blaifa/InternVL3_5:4B"
  "openbmb/minicpm-v4.5:8b"
)

echo "run,model,duration_sec,output_dir" > "${OUT_ROOT}/timing_summary.csv"

for run in $(seq 1 "${RUNS}"); do
  for model in "${MODELS[@]}"; do
    slug="$(echo "${model}" | tr '/:' '__')"
    out_dir="${OUT_ROOT}/run_${run}/${slug}"
    mkdir -p "${out_dir}"

    start="$(python - <<'PY'
import time
print(time.time())
PY
)"

    PICKINSTA_OLLAMA_MODEL="${model}" \
    pickinsta "${INPUT_DIR}" --output "${out_dir}" --scorer ollama --all

    end="$(python - <<'PY'
import time
print(time.time())
PY
)"

    duration="$(python - <<PY
start=${start}
end=${end}
print(f"{end-start:.3f}")
PY
)"

    echo "${run},${model},${duration},${out_dir}" >> "${OUT_ROOT}/timing_summary.csv"
  done
done
```

## 10. Common Failure Cases

- `curl /api/tags` fails:
  The server is not reachable yet. Fix host, port, firewall, or service state first.

- `pickinsta` says the model is missing:
  Pull the exact model on the Ollama host and confirm it appears in `ollama list`.

- Requests time out:
  Raise `PICKINSTA_OLLAMA_TIMEOUT_SEC` and reduce `PICKINSTA_OLLAMA_CONCURRENCY`.

- Server is reachable but throughput is poor:
  Tune `OLLAMA_NUM_PARALLEL`, `OLLAMA_NUM_THREAD`, and `PICKINSTA_OLLAMA_CONCURRENCY` together, then rerun the benchmark scripts.

