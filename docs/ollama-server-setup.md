# Ollama Server Setup (macOS + Linux)

This is a quick setup reference for running your own Ollama server for `pickinsta`.
Use this page for the minimum commands, and rely on the official docs for full details and platform-specific troubleshooting.

## Official Documentation (primary source)

- Docs home: [https://docs.ollama.com/](https://docs.ollama.com/)
- Quickstart: [https://docs.ollama.com/quickstart](https://docs.ollama.com/quickstart)
- macOS install: [https://docs.ollama.com/macos](https://docs.ollama.com/macos)
- Linux install: [https://docs.ollama.com/linux](https://docs.ollama.com/linux)
- FAQ (server env/config): [https://docs.ollama.com/faq](https://docs.ollama.com/faq)
- Modelfile reference: [https://docs.ollama.com/modelfile](https://docs.ollama.com/modelfile)
- Importing models (Safetensors/GGUF): [https://docs.ollama.com/import](https://docs.ollama.com/import)
- Qwen3-VL 8B model page: [https://ollama.com/library/qwen3-vl:8b](https://ollama.com/library/qwen3-vl:8b)
- InternVL3.5 model page: [https://ollama.com/blaifa/InternVL3_5](https://ollama.com/blaifa/InternVL3_5)
- MiniCPM-v4.5 model page: [https://ollama.com/openbmb/minicpm-v4.5](https://ollama.com/openbmb/minicpm-v4.5)

## 1. Install Ollama

### macOS

Follow the official macOS instructions:
- [https://docs.ollama.com/macos](https://docs.ollama.com/macos)

Brief CLI check:

```bash
ollama -v
```

### Linux

Follow the official Linux instructions:
- [https://docs.ollama.com/linux](https://docs.ollama.com/linux)

Brief CLI install/check:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama -v
```

## 2. Start the server

### macOS

If using the app-based install, start the Ollama app and verify with:

```bash
curl http://127.0.0.1:11434/api/tags
```

To expose on your LAN, set host via `launchctl` (see FAQ for full steps):

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

Reference: [https://docs.ollama.com/faq](https://docs.ollama.com/faq)

### Linux

Typical service flow:

```bash
sudo systemctl start ollama
sudo systemctl status ollama
curl http://127.0.0.1:11434/api/tags
```

To expose on your LAN, set `OLLAMA_HOST` via systemd override (full steps in docs):

```bash
sudo systemctl edit ollama.service
# add:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Reference: [https://docs.ollama.com/faq](https://docs.ollama.com/faq)

## 3. Pull the models for speed comparison

For the 4-model comparison benchmark, prepare these tags on the server:

```bash
ollama pull qwen3-vl:8b
ollama pull blaifa/InternVL3_5:8b
ollama pull blaifa/InternVL3_5:4B
ollama pull openbmb/minicpm-v4.5:8b
```

Then verify:

```bash
ollama list
```

### 3.1 InternVL3.5 pull check

If the namespaced tag is available on your server:

```bash
ollama pull blaifa/InternVL3_5
ollama list
```

If your local tag name differs, use your local tag in the benchmark `--models` list.

## 3.2 Version check

Check your server version:

```bash
ollama -v
```

If your version is `0.16.3`, you are already above `0.7.0`, so version is not the blocker.

## 4. Point pickinsta to your remote server

In your `.env`:

```bash
PICKINSTA_OLLAMA_BASE_URL=http://YOUR_SERVER_IP:11434
PICKINSTA_OLLAMA_MODEL=qwen3-vl:8b
```

Run:

```bash
pickinsta ./input --output ./selected --scorer ollama --all
```

## 5. Simple remote test from your local machine

Assuming Ollama is running on your M1 machine and reachable on your network:

```bash
curl http://<M1_IP>:11434/api/generate \
  -d '{"model": "qwen3-vl:8b", "prompt": "Hello", "stream": false}'
```

Optional environment variable form:

```bash
export PICKINSTA_OLLAMA_HOST_URL="http://<M1_IP>:11434"
curl "${PICKINSTA_OLLAMA_HOST_URL}/api/generate" \
  -d '{"model": "qwen3-vl:8b", "prompt": "Hello", "stream": false}'
```

## 6. Run the 4-model manual speed benchmark

From the project root:

```bash
.venv/bin/python tests/benchmarks/benchmark_ollama_models.py \
  --input ./input \
  --all \
  --runs 3 \
  --models qwen3-vl:8b blaifa/InternVL3_5:8b blaifa/InternVL3_5:4B openbmb/minicpm-v4.5:8b \
  --report docs/ollama-model-speed-benchmark-report.md
```

If your tag names differ, replace them in `--models` with your exact local tags from `ollama list`.

The report compares:
- average seconds per image
- average images per minute
- average run duration
- failures per run

## 7. Save per-model outputs in separate folders (scored images + reports)

Use this when you want full result artifacts per model and per run, not only speed tables.

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

Output structure:
- `model_compare_<timestamp>/run_1/<model_slug>/...`
- `model_compare_<timestamp>/run_2/<model_slug>/...`
- each model folder contains ranked images + `selection_report.json` + `selection_report.md`
- `timing_summary.csv` gives runtime per model/run for direct speed comparison
