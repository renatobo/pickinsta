# Troubleshooting

This page covers the common failure modes for installing and running `pickinsta`.

## First Checks

Confirm the active environment and CLI entrypoint:

```bash
python --version
python -m pip show pickinsta
pickinsta -h
```

If the CLI is missing, reinstall in the active virtualenv:

```bash
python -m pip install -e ".[clip,claude,yolo]"
```

## OpenCV on Linux Servers

If startup fails with `ImportError: libGL.so.1: cannot open shared object file`:

Prefer the headless build:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -U opencv-python-headless
```

If you need GUI OpenCV bindings instead:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## CLIP Issues

If CLIP fails during import or model load:

```bash
python -m pip install -e ".[clip]"
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
```

Notes:

- The first successful CLIP run downloads model weights from Hugging Face.
- `HF_TOKEN` is optional but helps with download limits and warning noise.
- Supported project Python versions are 3.11 through 3.14.

If downloads stall or fail repeatedly, verify general outbound internet access and retry with `HF_TOKEN` set in `.env` or the shell environment.

## Claude Issues

If Claude scoring raises `No module named 'anthropic'`:

```bash
python -m pip install -e ".[claude]"
python -c "import anthropic; print(anthropic.__version__)"
```

If Claude scoring fails because no key is available:

```bash
cp .env.example .env
```

Then set:

```bash
ANTHROPIC_API_KEY=your_key_here
```

`pickinsta` searches for Claude credentials in this order:

1. Current process environment
2. `cwd/.env`
3. `<input-folder>/.env`

If the model id is rejected or unavailable:

```bash
pickinsta ./input --output ./selected --scorer claude --claude-model claude-sonnet-4-6
```

Or set:

```bash
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`CLAUDE_MODEL` is also accepted as an alias fallback.

If scores look stale, clear or bypass cache with:

```bash
pickinsta ./input --output ./selected --scorer claude --rescore
```

Claude cache files live beside original inputs as `*.pickinsta.json`.

Claude caches use schema v2. Cache reuse requires the source content, scorer, model,
prompt, and scoring/preprocessing options to match. Model changes and options such as
crop-first input invalidate an entry automatically. `--rescore` remains the direct way
to force fresh Claude scoring.

Technical cache entries in the work folder also validate their schema, work-image
modification time, and an algorithm/dependency fingerprint. If technical results still
look stale, remove the affected `*.techscore.json` files; `--rescore` affects vision
caches only.

## Ollama Issues

If Ollama scoring cannot connect:

```bash
curl "${PICKINSTA_OLLAMA_BASE_URL:-http://127.0.0.1:11434}/api/tags"
```

Minimum required config:

```bash
PICKINSTA_OLLAMA_BASE_URL=http://YOUR_SERVER_IP:11434
PICKINSTA_OLLAMA_MODEL=qwen2.5vl:7b
```

If the server is reachable but scoring still fails:

- Verify the model exists on the server with `ollama list`.
- Increase the client timeout if the model is slow to answer.
- Reduce `PICKINSTA_OLLAMA_CONCURRENCY` if the host is overloaded.

Useful env knobs:

```bash
PICKINSTA_OLLAMA_TIMEOUT_SEC=300
PICKINSTA_OLLAMA_CONCURRENCY=2
PICKINSTA_OLLAMA_MAX_RETRIES=2
PICKINSTA_OLLAMA_RETRY_BACKOFF_SEC=0.75
PICKINSTA_OLLAMA_CIRCUIT_BREAKER_ERRORS=6
PICKINSTA_OLLAMA_USE_YOLO_CONTEXT=false
```

Linux `systemd` tuning example:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_NUM_THREAD=3"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

If CPU appears capped or throughput is unexpectedly low, inspect service limits:

```bash
sudo systemctl show ollama -p CPUQuota -p AllowedCPUs
cat /sys/fs/cgroup/cpu/system.slice/ollama.service/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/system.slice/ollama.service/cpu.cfs_period_us
```

For full setup and deployment notes, see [`ollama-server-setup.md`](ollama-server-setup.md).

## YOLO Issues

If YOLO detection is unavailable or the model cannot initialize:

```bash
python -m pip install -e ".[yolo]"
```

On first use, the default weights are downloaded to:

```text
~/.cache/pickinsta/models/yolov8n.pt
```

To force a specific local model:

```bash
PICKINSTA_YOLO_MODEL=/absolute/path/to/model.pt
```

If YOLO is missing, `pickinsta` still runs, but smart crop falls back to non-YOLO behavior.

## CLI and Input Problems

Show the full CLI help:

```bash
pickinsta -h
```

If parsing fails, the CLI prints the full help text rather than only a short usage line.

If the output looks empty or too small:

- Confirm the input folder actually contains supported image files.
- Check whether `--dedup-only` was used instead of a scoring mode.
- Check whether the technically strongest burst representative was selected instead of another frame you expected.

If you want more images to reach vision scoring, use either:

```bash
pickinsta ./input --scorer clip --vision-pct 0.8
```

or:

```bash
pickinsta ./input --scorer clip --all
```

## Report and Output Checks

Successful runs should produce:

- ranked image outputs in the chosen output directory
- `selection_report.json`
- `selection_report.md`
- `index.html`
- `run_manifest.json`

Treat `current_run.json` as the completion marker. It points to a complete immutable
generation under `.pickinsta-runs/`; read that directory when a consistent snapshot is
needed. If the pointer is missing or invalid, inspect the terminal error and rerun.
Flat root artifacts remain compatibility copies, and unrelated files in the output
folder are preserved.

If the manifest reports `"status": "degraded"`, publication did finish, but at least
one crop or copy operation failed. Inspect the issue records:

```bash
python -m json.tool ./selected/run_manifest.json
```

- `crop` issues count as `skipped` because that selected entry was not added to the
  output report.
- `copy_hd` and `copy_full` issues count as `failed`; other variants for that entry may
  still be usable.
- `reason` contains the underlying exception text and `artifact` identifies the
  affected output.

Resolve the reported filesystem, source-file, or image-processing problem and rerun
the same command. The next successful publication replaces only its managed artifact
names; operator-created files remain untouched.

### Diagnose a slow or unexpectedly uncached run

Inspect the telemetry fields without printing unrelated manifest content:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("selected/run_manifest.json").read_text())
for key in ("stage_timings_seconds", "stage_invocations", "caches", "runtime"):
    print(f"{key}: {manifest.get(key)}")
PY
```

- A high stage time identifies where wall-clock time accumulated; divide by the
  corresponding invocation count only when an average invocation is meaningful.
- `technical` or `vision` cache misses mean those scores were recomputed. An absent
  cache entry means that cache was not consulted, not that every lookup missed.
- Compare `runtime.packages`, `cpu_count`, and worker configuration before comparing
  runs from different machines or environments.
- `peak_memory_bytes` is the process peak reported by the operating system and may be
  `null` where resource accounting is unavailable. It is not per-stage memory.

To establish a repeatable local baseline without models or network calls, run:

```bash
.venv/bin/python tests/benchmarks/offline_stage_benchmark.py \
  --repetitions 5 --images 4 --output /tmp/pickinsta-benchmark.json
```

Compare cold, warm, and cached medians and p95 separately on similar, otherwise-idle
hardware. A cached run should report one technical-cache hit per generated image. If
it does not, cache identity or persistence is broken. The benchmark deliberately runs
production workers sequentially, so it measures deterministic local computation and
cache reuse rather than parallel or model throughput.

### Tune local worker and OpenCV threads

The effective precedence is stage-specific setting, then shared setting, then logical
CPU count (or 4 when unavailable):

```bash
PICKINSTA_MAX_WORKERS=8
PICKINSTA_PROCESS_WORKERS=4
PICKINSTA_THREAD_WORKERS=8
PICKINSTA_OPENCV_THREADS=1
```

All worker caps must be integers from 1 through 256. Invalid values fall back and emit
a warning. `PICKINSTA_OPENCV_THREADS=0` leaves OpenCV unchanged; use `1` when multiple
Python threads run technical scoring or crops and CPU use or tail latency suggests
nested OpenCV oversubscription. The resolved values are recorded under
`configuration.workers` in `run_manifest.json`. Claude and Ollama request concurrency
remain governed by their own settings.

If ranked outputs exist but a crop looks unsafe, inspect the `uncertain_crop` fields in `selection_report.json` or open the local gallery to see the warning badges.
