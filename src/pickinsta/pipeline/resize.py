"""Image discovery and resize preparation for the local pipeline."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Callable

from PIL import Image

from pickinsta.config import WorkerSettings, resolve_worker_settings
from pickinsta.events import EventLevel, emit
from pickinsta.pipeline.worker_tuning import bounded_worker_count

MAX_RESIZE_PX = 1920
# Refuse pathological images before EXIF transpose or pixel decode allocates
# a full-resolution buffer. This still accommodates large camera originals.
MAX_SOURCE_PIXELS = 100_000_000
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".bmp"}

ResizeResult = tuple[Path, Path] | None
ResizeWorker = Callable[[tuple[Path, Path]], ResizeResult]


def resize_one_image(args: tuple[Path, Path]) -> ResizeResult:
    """Resize one image and return its destination and source paths."""
    img_path, destination = args
    try:
        from PIL import ImageOps

        with Image.open(img_path) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
                return None
            exif_bytes = image.info.get("exif")
            image = ImageOps.exif_transpose(image)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            width, height = image.size
            longest = max(width, height)
            if longest > MAX_RESIZE_PX:
                scale = MAX_RESIZE_PX / longest
                image = image.resize(
                    (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
                )
            save_kwargs: dict[str, object] = {"quality": 90}
            if exif_bytes:
                try:
                    from PIL.Image import Exif

                    exif = Exif()
                    exif.load(exif_bytes)
                    if 0x0112 in exif:
                        exif[0x0112] = 1
                    save_kwargs["exif"] = exif.tobytes()
                except Exception:
                    save_kwargs["exif"] = exif_bytes
            image.save(destination, "JPEG", **save_kwargs)
        return destination, img_path
    except Exception:
        return None


def resize_for_processing(
    src_folder: Path,
    work_folder: Path,
    worker_settings: WorkerSettings | None = None,
    *,
    resize_worker: ResizeWorker = resize_one_image,
) -> tuple[list[Path], dict[Path, Path]]:
    """Prepare supported source images as bounded JPEG working files."""
    work_folder.mkdir(parents=True, exist_ok=True)
    resized: list[Path] = []
    source_map: dict[Path, Path] = {}
    reused = 0

    src_images = [
        path for path in sorted(src_folder.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    emit(
        "resize.discovered",
        f"📁 Found {len(src_images)} images in {src_folder}",
        stage="resize",
        count=len(src_images),
        path=str(src_folder),
    )

    to_resize: list[tuple[Path, Path]] = []
    stems_seen: dict[str, int] = {}
    for img_path in src_images:
        stem = img_path.stem
        if stem in stems_seen:
            stems_seen[stem] += 1
            destination_stem = f"{stem}_{stems_seen[stem]}"
            emit(
                "resize.stem_collision",
                f"  ⚠ Stem collision: '{img_path.name}' → work file '{destination_stem}.jpg'",
                level=EventLevel.WARNING,
                stage="resize",
            )
        else:
            stems_seen[stem] = 0
            destination_stem = stem
        destination = work_folder / f"{destination_stem}.jpg"
        if destination.exists() and destination.stat().st_mtime >= img_path.stat().st_mtime:
            resized.append(destination)
            source_map[destination] = img_path
            reused += 1
        else:
            to_resize.append((img_path, destination))

    settings = worker_settings or resolve_worker_settings()
    if to_resize:
        worker_count = bounded_worker_count("process", len(to_resize), settings)
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            for result in pool.map(resize_worker, to_resize):
                if result is None:
                    emit(
                        "resize.image.failed",
                        "  ⚠ Skipping an image: resize failed",
                        level=EventLevel.WARNING,
                        stage="resize",
                    )
                    continue
                destination, img_path = result
                resized.append(destination)
                source_map[destination] = img_path

    resized.sort(key=lambda path: path.name)
    newly_resized = len(resized) - reused
    emit(
        "resize.complete",
        f"  ✅ Prepared {len(resized)} images (new: {newly_resized}, reused: {reused})"
        f" to max {MAX_RESIZE_PX}px → {work_folder} "
        f"(up to {settings.process_workers} workers)",
        stage="resize",
        prepared=len(resized),
        new=newly_resized,
        reused=reused,
    )
    return resized, source_map
