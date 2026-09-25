"""Worker-limit policy for local image-processing stages."""

from typing import Literal

import cv2

from pickinsta.config import WorkerSettings

WorkerKind = Literal["process", "thread"]


def bounded_worker_count(kind: WorkerKind, item_count: int, settings: WorkerSettings) -> int:
    """Return a valid executor size bounded by work available and configured policy."""
    configured_cap = settings.process_workers if kind == "process" else settings.thread_workers
    return max(1, min(item_count, configured_cap))


def configure_opencv_threads(settings: WorkerSettings) -> None:
    """Apply the explicit process-global OpenCV thread limit, if configured."""
    if settings.opencv_threads is not None:
        cv2.setNumThreads(settings.opencv_threads)
