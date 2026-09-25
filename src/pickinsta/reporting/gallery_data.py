"""Build presentation data consumed by the HTML selection gallery."""

import json
import math
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from PIL import Image
from PIL.ExifTags import TAGS

VISION_CRITERIA = (
    "subject_clarity",
    "lighting",
    "color_pop",
    "emotion",
    "scroll_stop",
    "crop_4x5",
)


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_input_folder(output_folder: Path) -> Optional[Path]:
    report = output_folder / "selection_report.md"
    if not report.exists():
        return None
    try:
        lines = report.read_text(encoding="utf-8").splitlines()[:10]
    except (OSError, UnicodeError):
        return None
    for line in lines:
        if line.startswith("- Input:") and "`" in line:
            candidate = Path(line.split("`")[1])
            if candidate.exists():
                return candidate
    return None


def _vision_detail(input_folder: Optional[Path], source_name: str) -> dict:
    if input_folder is None:
        return {}
    base = input_folder.resolve()
    cache_path = (base / f"{source_name}.pickinsta.json").resolve()
    if not cache_path.is_relative_to(base):
        return {}
    cached = _read_json(cache_path)
    if not isinstance(cached, dict):
        return {}
    vision = cached.get("vision", {})
    if not isinstance(vision, dict) or not any(key in vision for key in VISION_CRITERIA):
        return {}
    scores = {}
    for key in VISION_CRITERIA:
        try:
            value = float(vision.get(key, 0))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            scores[key] = max(0.0, min(10.0, value))
    return scores


def _exif(input_folder: Optional[Path], source_name: str) -> dict:
    if input_folder is None:
        return {}
    base = input_folder.resolve()
    source_file = (base / source_name).resolve()
    if not source_file.is_relative_to(base) or not source_file.exists():
        return {}
    try:
        with Image.open(source_file) as image:
            tagged = {TAGS.get(key, key): value for key, value in (image._getexif() or {}).items()}
        result = {}
        if tagged.get("Make"):
            make = tagged["Make"].strip()
            model = tagged.get("Model", "").strip()
            result["camera"] = model if model.startswith(make) else f"{make} {model}".strip()
        if tagged.get("LensModel"):
            result["lens"] = str(tagged["LensModel"]).strip()
        if tagged.get("FocalLength"):
            result["focal"] = f"{float(tagged['FocalLength']):.0f}mm"
        if tagged.get("FNumber"):
            result["aperture"] = f"f/{float(tagged['FNumber']):.1f}"
        if tagged.get("ExposureTime"):
            exposure = float(tagged["ExposureTime"])
            result["shutter"] = (
                f"{exposure:.1f}s" if exposure >= 1 else f"1/{int(round(1 / exposure))}s"
            )
        if tagged.get("ISOSpeedRatings"):
            result["iso"] = f"ISO {tagged['ISOSpeedRatings']}"
        if tagged.get("DateTimeOriginal"):
            result["date"] = str(tagged["DateTimeOriginal"])
        return result
    except Exception:
        return {}


def build_gallery_data(output_folder: Path, input_folder: Optional[Path] = None) -> list[dict]:
    """Build gallery data from a selection output folder."""
    report_data = _read_json(output_folder / "selection_report.json")
    if not isinstance(report_data, list):
        return []
    input_folder = input_folder or _infer_input_folder(output_folder)
    data = []
    for item in report_data:
        if not isinstance(item, dict) or "rank" not in item:
            continue
        cropped_name = item.get("output_cropped") or item.get("output", "")
        if not isinstance(cropped_name, str):
            cropped_name = ""
        yolo_debug_img = f"debug_yolo_{cropped_name}"
        yolo = _read_json(output_folder / f"{yolo_debug_img}.json")
        source_name = item.get("filename", "")
        if not isinstance(source_name, str):
            source_name = str(source_name)
        try:
            final_score = float(item.get("final_score", 0))
            technical_composite = float(item.get("technical_composite", 0))
            vision_total = float(item.get("vision_total", 0))
        except (TypeError, ValueError):
            continue
        if not all(
            math.isfinite(value) for value in (final_score, technical_composite, vision_total)
        ):
            continue
        try:
            rank = int(item["rank"])
        except (TypeError, ValueError, OverflowError):
            continue
        data.append(
            {
                "rank": rank,
                "filename": source_name,
                "final_score": final_score,
                "technical_composite": technical_composite,
                "vision_total": vision_total,
                "one_line": item.get("one_line", ""),
                "output_cropped": quote(cropped_name),
                "output_hd": quote(item.get("output_hd") or ""),
                "output_full": quote(item.get("output_full") or ""),
                "uncertain_crop": item.get("uncertain_crop", False),
                "vision_detail": _vision_detail(input_folder, source_name),
                "yolo": yolo,
                "exif": _exif(input_folder, source_name),
                "burst": item.get("burst"),
                "yolo_debug_img": (
                    quote(yolo_debug_img) if (output_folder / yolo_debug_img).exists() else None
                ),
            }
        )
    return data
