# Troubleshooting

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
