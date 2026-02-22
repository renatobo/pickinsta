# Troubleshooting

## OpenCV on Linux servers

If startup fails with `ImportError: libGL.so.1: cannot open shared object file`:

- Prefer headless OpenCV in server environments:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -U opencv-python-headless
```

- Or install missing OS libs if you must use GUI OpenCV:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

## CLIP

If CLIP fails during model load:

- Ensure dependencies are installed in the active venv:

```bash
python -m pip install -e ".[clip]"
```

- Verify imports:

```bash
python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
```

- First CLIP run downloads model weights from Hugging Face, so internet access is required.
- Only if the import check fails on your Python version, use Python 3.10-3.12.

## Claude

If Claude scoring shows `No module named 'anthropic'`:

```bash
python -m pip install -e ".[claude]"
python -c "import anthropic; print(anthropic.__version__)"
```

If Claude scoring shows a model `not_found_error`:

```bash
pickinsta ./input --output ./selected --top 10 --scorer claude --claude-model claude-sonnet-4-6
```

Or set in env/.env:

```bash
ANTHROPIC_MODEL=claude-sonnet-4-6
```

## Ollama

If Ollama scoring fails to connect:

```bash
curl "${PICKINSTA_OLLAMA_BASE_URL:-http://127.0.0.1:11434}/api/tags"
```

Check required env values:

```bash
PICKINSTA_OLLAMA_BASE_URL=http://YOUR_SERVER_IP:11434
PICKINSTA_OLLAMA_MODEL=qwen2.5vl:7b
```

If scoring is slow, tune both client and server concurrency:

```bash
export PICKINSTA_OLLAMA_CONCURRENCY=4
```

Linux systemd override example:

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

If CPU appears capped near 50%, check cgroup limits:

```bash
sudo systemctl show ollama -p CPUQuota -p AllowedCPUs
cat /sys/fs/cgroup/cpu/system.slice/ollama.service/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/system.slice/ollama.service/cpu.cfs_period_us
```

## CLI usage/help

Show all command options:

```bash
pickinsta -h
```

If CLI parsing fails (missing required args or unknown flags), `pickinsta`
prints full help text (not only a one-line usage summary).

## YOLO

If YOLO detection is unavailable or model setup fails:

- Ensure YOLO dependency is installed:

```bash
python -m pip install -e ".[yolo]"
```

- On first YOLO run, `pickinsta` downloads model weights to:
  - `~/.cache/pickinsta/models/yolov8n.pt`

- To use a custom/local model path:

```bash
PICKINSTA_YOLO_MODEL=/absolute/path/to/model.pt
```
