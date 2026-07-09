# AGENTS.md

## Project Workflow

- Create and activate a virtual environment before installing dependencies.
- Preferred install path: `python3 -m venv .venv && source .venv/bin/activate && make install-dev`.
- `make install-dev` installs dev tooling plus the `clip`, `claude`, and `yolo` extras.
- Use `cp .env.example .env` for local environment setup when Claude, HF, or Ollama settings are needed.

## Common Commands

- `make test` runs the pytest suite.
- `make lint` runs Ruff checks on `src` and `tests`.
- `make format` runs Ruff formatting on `src` and `tests`.
- `make check` runs `lint` and `test`.
- `make pre-commit-install` installs the pre-commit hooks.

## CLI And Runbook

- Main CLI entrypoint is `pickinsta`.
- Common workflows:
  - `pickinsta ./input --output ./selected --top 10 --scorer clip`
  - `pickinsta ./input --output ./selected --scorer claude --all`
  - `pickinsta ./input --output ./selected --scorer ollama --all`
  - `pickinsta ./input --output ./deduped --dedup-only`
- `--claude-crop-first` is available for pre-cropping before Claude scoring.
- `--rescore` bypasses cached vision results.

## Notes

- CI runs on Python 3.10, 3.11, 3.12, and 3.13 with `make lint` and `make test`.
- Manual benchmark scripts live under `tests/benchmarks/` and are not part of the standard pytest run.
- If a workflow detail is unclear, prefer a short TODO over inventing new repo guidance.
