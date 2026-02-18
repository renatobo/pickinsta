# Contributing

Thanks for contributing to `pickinsta`.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
```

## Local checks

Run before opening a PR:

```bash
make lint
make test
```

Or run both:

```bash
make check
```

## Pre-commit hooks

Install once per clone:

```bash
make pre-commit-install
```

## Pull requests

- Keep PRs focused (one concern per PR when possible).
- Add/update tests for behavior changes.
- Update docs when user-visible behavior changes.
- Ensure CI is green.

## Project layout

- `src/pickinsta/` - application code
- `tests/` - automated pytest suite
- `docs/` - reference docs and diagrams
- `debug/` - manual debug scripts and example debug artifacts
