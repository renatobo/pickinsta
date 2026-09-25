# Architecture Module Map

This document records the current module boundaries after the first extraction pass. It is a
current-state map, not a restatement of the broader action plan. Counts below were generated from
the Python AST and physical source lines on 2026-07-18.

## Current modules and dependency direction

Dependencies should flow from entry points and orchestration toward domain stages, then toward
small policy/data and infrastructure modules:

```text
__main__
  -> cli (argument parsing, validation, and dispatch)
       -> ig_image_selector (legacy facade and current orchestrator, loaded lazily)
       -> clip_scorer
       -> config, models, telemetry
       -> pipeline/{deduplication, technical_scoring, worker_tuning}
       -> infrastructure/{filesystem, vision_cache}
       -> reporting/{gallery, gallery_data, markdown, recursive}

reporting -> models and infrastructure/filesystem
pipeline  -> config, models and infrastructure/filesystem
infrastructure/vision_cache -> infrastructure/filesystem
config, models -> Python standard library only
```

The package currently has no package-local circular import. In particular, none of the extracted
modules imports `ig_image_selector`; that constraint must remain true.

| Module | Lines | Functions/classes | Current responsibility | Allowed package dependencies |
|---|---:|---:|---|---|
| `config.py` | 282 | 24 / 2 | Environment parsing and immutable integration/worker settings | none |
| `models.py` | 19 | 0 / 1 | Cross-stage `ImageScore` data model | none |
| `infrastructure/filesystem.py` | 74 | 3 / 0 | Atomic writes and managed-artifact publication | none |
| `infrastructure/vision_cache.py` | 74 | 3 / 0 | Vision cache identity, reads, and writes | `infrastructure.filesystem` |
| `pipeline/deduplication.py` | 374 | 10 / 4 | Perceptual hash index, burst features, and dedup execution | `events`, `pipeline.worker_tuning` |
| `pipeline/technical_scoring.py` | 299 | 14 / 0 | Technical metrics, cache, and batch execution | `config`, `events`, `models`, `infrastructure.filesystem`, `pipeline.worker_tuning` |
| `pipeline/worker_tuning.py` | 25 | 2 / 0 | Worker bounds and OpenCV thread policy | `config` |
| `pipeline/orchestration.py` | 326 | 8 / 3 | Stable collaborator contracts, recursive dispatch, and public use-case wrappers | `config`, `events`, `pipeline.full_run`, `pipeline.dedup_run` |
| `pipeline/full_run.py` | 465 | 1 / 0 | Full selection pipeline use-case execution | `config`, `events` |
| `pipeline/dedup_run.py` | 311 | 1 / 0 | Dedup-only pipeline use-case execution | `config`, `events` |
| `reporting/gallery_data.py` | 135 | 5 / 0 | Gallery serialization and metadata | none |
| `reporting/markdown.py` | 104 | 2 / 0 | Run report rendering | `models`, `infrastructure.filesystem` |
| `reporting/recursive.py` | 73 | 1 / 0 | Recursive summary rendering | `reporting.markdown`, `infrastructure.filesystem` |
| `reporting/gallery.py` | 277 | 4 / 0 | Gallery data-to-document orchestration | `reporting.gallery_data`, `reporting.templates` |
| `reporting/templates.py` | 808 | 0 / 0 | Static HTML, CSS, and JavaScript document templates | none |
| `telemetry.py` | 168 | 4 / 1 | Run manifest and timing/resource observations | none |
| `events.py` | 100 | 5 / 2 | Immutable structured events and human console rendering | none |
| `clip_scorer.py` | 86 | 3 / 0 | Local CLIP model loading and scoring | none |
| `cli.py` | 152 | 4 / 2 | Argument parsing, validation, and facade dispatch | `config`; selector loaded lazily |
| `ig_image_selector.py` | 298 | 27 / 0 | Compatibility aliases and thin live-collaborator wrappers | all extracted layers |

`reporting/__init__.py` eagerly imports all reporting implementations. Internal production code
should import the concrete reporting module, as the selector already does. New low-level modules
must not import from `reporting`'s package root because that would pull in gallery, markdown, and
filesystem dependencies together and increase circular-import risk.

## Compatibility facade

`ig_image_selector.py` is still the public compatibility seam. Its `main` wrapper injects live
facade callables into `cli.main`, preserving historical monkeypatch behavior while the installed
console script and `python -m pickinsta` use `pickinsta.cli:main`. It re-exports historical config
constants and `_read_env_file`; aliases extracted technical helpers; and wraps technical scoring so
existing monkeypatches of `yolo_detect_subject`, `score_technical`, cache loaders, and distribution
printing continue to work. `_atomic_write_text`, `_gallery_build_data`, and recursive reporting also
remain selector-level compatibility exports.

The safe migration rule is therefore:

1. Extract implementation into a module that does not import the selector.
2. Inject callbacks for selector-patchable collaborators instead of importing them backward.
3. Leave a thin selector wrapper or alias with the historical signature.
4. Run the existing focused tests before changing or removing a wrapper.

Do not make an extracted module import selector symbols to preserve monkeypatch behavior. That
would create the most likely future cycle: `selector -> extracted stage -> selector`.

## Selector facade status

The extraction is complete: the selector is a 298-line compatibility facade with no stage
algorithms, pool construction, image traversal, report rendering, or CLI parsing. Direct aliases
cover pure helpers and historical exports. Explicit thin wrappers inject live selector globals into
typed collaborator objects so existing monkeypatch-based callers continue to work.

The former `reporting/gallery.py` concentration has been split: rendering decisions remain in the
small gallery module while static HTML/CSS/JavaScript documents live in dependency-free
`reporting/templates.py`. Snapshot-equivalence tests protect the generated full and dedup gallery
documents.

## Completed extraction boundaries

- Vision prompts, Claude/Ollama transports, parsing, retry behavior, and batch coordination live
  under `vision/` and `pipeline/vision_scoring.py`.
- YOLO lifecycle and detection live under `detection/`; pure crop geometry and crop I/O live in
  separate pipeline modules.
- Resize, technical scoring, deduplication, full-run, dedup-only, and recursive coordination have
  distinct modules and immutable collaborator contracts.
- CLI parsing and dispatch live in `cli.py`; both console entry paths use it without eagerly loading
  optional model libraries.
- Structured stage events use `events.py` with a context-local sink and legacy-compatible human
  renderer. Configuration remains a standard-library-only bootstrap boundary.

## Enforced maintenance rules

`tests/test_architecture_boundaries.py` scans all package imports and rejects cycles, back-imports to
the selector, and invalid layer edges. It also enforces the standard-library-only config/model
boundary and prevents low-level modules from importing orchestration, CLI, or the reporting package
root. `tests/test_selector_compatibility.py` separately protects the intentionally retained facade
exports and live monkeypatch capture.

For future changes:

1. Put new behavior in the owning domain module, not the compatibility facade.
2. Inject patch-sensitive collaborators through the frozen facade contracts.
3. Keep optional model imports lazy and outside pure geometry/configuration modules.
4. Update this table when a module's responsibility or dependency direction changes.
