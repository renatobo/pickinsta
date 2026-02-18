# Debug Scripts

This folder contains manual debug utilities for `pickinsta`.

These scripts are **not** automated tests and are **not** run by `pytest`.
Use them when you want to inspect behavior interactively or troubleshoot a
specific pipeline step.

## Scripts

### `debug_yolo_claude.py`
- Purpose: inspect how YOLO detections are turned into extra context appended
  to the Claude vision prompt.
- What it does:
  - loads one hardcoded image from `input/`
  - runs `yolo_detect_subject(...)`
  - prints the final prompt (with or without YOLO context)
- Run:
  ```bash
  python debug/debug_yolo_claude.py
  ```

## Notes

- Keep production tests in `/Users/renatobo/development/pickinsta/tests`.
- Prefer these scripts for quick diagnostics, not for CI validation.
- Debug output folders include sample artifacts to illustrate expected behavior.
