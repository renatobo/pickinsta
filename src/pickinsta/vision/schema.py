"""Validation for scores returned by model-backed vision providers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from pickinsta.vision.prompts import VISION_SCORE_KEYS


def normalize_vision_payload(payload: object, *, fallback_text: str = "") -> dict:
    """Return bounded, finite scores and a safe one-line summary.

    Invalid provider data is rejected so callers can use their normal fallback
    path instead of persisting values that break ranking or HTML rendering.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("vision response must be an object")
    scores: dict[str, int] = {}
    for key in VISION_SCORE_KEYS:
        raw = payload.get(key, 5)
        if isinstance(raw, bool):
            raise ValueError(f"vision score {key!r} must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"vision score {key!r} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"vision score {key!r} must be finite")
        scores[key] = max(0, min(10, round(value)))

    raw_total = payload.get("total", sum(scores.values()))
    if isinstance(raw_total, bool):
        raise ValueError("vision total must be numeric")
    try:
        total = float(raw_total)
    except (TypeError, ValueError) as error:
        raise ValueError("vision total must be numeric") from error
    if not math.isfinite(total):
        raise ValueError("vision total must be finite")
    raw_text = payload.get("one_line", fallback_text or "Vision scoring summary")
    if not isinstance(raw_text, str):
        raise ValueError("vision summary must be text")
    one_line = " ".join(raw_text.split())[:220] or fallback_text or "Vision scoring summary"
    return {**scores, "total": max(0, min(60, round(total))), "one_line": one_line}
