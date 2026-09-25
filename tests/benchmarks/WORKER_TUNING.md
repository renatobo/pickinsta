# Worker and OpenCV thread tuning

## Production policy

Local parallel stages choose `min(item_count, configured_cap)` from one immutable
settings snapshot per pipeline run:

- resize: a process pool;
- perceptual hash, duplicate sharpness, and burst features: three separate process pools;
- technical scoring, burst re-evaluation, and crop rendering: thread pools;
- Ollama and Claude use separately bounded integration-specific thread pools.

The process pools are appropriate for PIL-heavy work, and threads avoid forking a
process after YOLO/PyTorch is loaded. The risk is that OpenCV operations invoked
inside each Python thread may also create native threads. On a host with `N`
logical CPUs, a stage can therefore schedule approximately `N` Python workers,
each with an OpenCV native pool, causing CPU oversubscription, memory pressure,
and worse tail latency. The resize progress message now reports the configured
process-worker cap.

## Configuration contract

These optional environment variables are resolved when a run starts so tests
and embedding applications can change them without re-importing the package:

| Variable | Scope | Default | Valid effective value |
|---|---|---|---|
| `PICKINSTA_MAX_WORKERS` | fallback cap for all local pools | logical CPU count, or 4 | integer 1 through 256 |
| `PICKINSTA_PROCESS_WORKERS` | resize and dedup feature pools | `PICKINSTA_MAX_WORKERS` | integer 1 through 256 |
| `PICKINSTA_THREAD_WORKERS` | technical, burst, and crop pools | `PICKINSTA_MAX_WORKERS` | integer 1 through 256 |
| `PICKINSTA_OPENCV_THREADS` | OpenCV native pool; `0` leaves library default unchanged | `0` | integer 0 through 256 |

For every pool, the effective count remains `min(item_count, configured_cap)`,
preserving today’s behavior when variables are absent. Empty, non-integer, zero
for worker caps, or negative values should fall back to the documented default
and emit one configuration warning per run. Do not silently clamp malformed
values. The OpenCV setting should be applied once at pipeline startup, before
any worker pool starts; `cv2.setNumThreads(1)` is the recommended setting when
technical/crop Python worker counts exceed one. It is process-global, so library
callers need an explicit API rather than mutation during individual tasks.

Recommended internal API:

```python
@dataclass(frozen=True)
class WorkerSettings:
    process_workers: int
    thread_workers: int
    opencv_threads: int | None

def resolve_worker_settings(cpu_count: int | None = None) -> WorkerSettings: ...
def bounded_workers(item_count: int, configured_cap: int) -> int: ...
def configure_opencv_threads(settings: WorkerSettings) -> None: ...
```

## Acceptance tests for the production slice

1. With no variables, an eight-item stage on a four-CPU host selects four workers.
2. `PICKINSTA_MAX_WORKERS=2` caps both process and thread stages at two.
3. A stage-specific value overrides the shared cap only for its pool kind.
4. A cap larger than the item count never creates idle workers.
5. Invalid caps fall back and produce a structured warning; they never create a
   zero-worker executor.
6. `PICKINSTA_OPENCV_THREADS=1` calls `cv2.setNumThreads(1)` exactly once before
   stage execution; absent or `0` does not call it.
7. Sequential and bounded-thread technical scoring produce identical ordered
   score dictionaries with optional YOLO detection disabled.
8. Existing integration-specific concurrency limits retain precedence and are
   not changed by this slice.

## Offline measurement

Run the deterministic, model-free comparison from the repository root:

```bash
.venv/bin/python tests/benchmarks/worker_scaling_benchmark.py \
  --images 8 --repetitions 5 --output /tmp/pickinsta-worker-scaling.json
```

The script compares Python worker counts 1, 2, and 4 with OpenCV native thread
counts 1 and 2. It disables optional subject-model detection, restores the
process-global OpenCV setting afterward, and exits nonzero if any configuration
changes technical scores. Timing is descriptive only: compare medians on the
same otherwise-idle machine and do not add a fixed wall-clock CI threshold.
