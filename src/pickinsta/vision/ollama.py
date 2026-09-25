"""Ollama vision response parsing and HTTP transport."""

import base64
import io
import json
import math
import re
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
from PIL import Image

from pickinsta.config import DEFAULT_ACCOUNT_CONTEXT, DEFAULT_CLAUDE_MODEL, DEFAULT_OLLAMA_MODEL
from pickinsta.vision.prompts import (
    OLLAMA_STRICT_JSON_SCHEMA,
    OLLAMA_SYSTEM_PROMPT,
    VISION_SCORE_KEYS,
    build_ollama_compact_json_prompt,
    build_vision_prompt,
)
from pickinsta.vision.prompts import (
    extract_account_context_from_prompt as _extract_account_context_from_prompt,
)
from pickinsta.vision.schema import normalize_vision_payload

OLLAMA_DEFAULT_NUM_PREDICT = 220
OLLAMA_QWEN_NUM_PREDICT_SMALL_EDGE = 650
OLLAMA_QWEN_NUM_PREDICT_LARGE_EDGE = 750
OLLAMA_QWEN_SMALL_EDGE_THRESHOLD = 512
OLLAMA_QWEN_MODEL_PREFIXES = ("qwen3-vl", "qwen2.5vl", "qwen2.5-vl")
OLLAMA_GEMMA4_MODEL_PREFIXES = ("gemma4",)
OLLAMA_GEMMA4_NUM_PREDICT = 350


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_qwen_ollama_model(model: str) -> bool:
    """Identify Qwen VL models that need stricter output controls."""
    normalized = (model or "").strip().lower()
    return any(normalized.startswith(prefix) for prefix in OLLAMA_QWEN_MODEL_PREFIXES)


def _is_gemma4_ollama_model(model: str) -> bool:
    """Identify Gemma 4 models.

    Gemma 4 requires the `think` key to be omitted entirely when using the `format`
    parameter — setting think=false breaks structured output (Ollama bug #15260).
    Also uses a slightly higher temperature per Google's recommendations.
    Handles both bare tags (gemma4:e4b) and namespaced models (user/gemma4-...).
    """
    normalized = (model or "").strip().lower()
    # Check bare name (gemma4:...) and namespaced (user/gemma4-...)
    bare = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
    return any(bare.startswith(prefix) for prefix in OLLAMA_GEMMA4_MODEL_PREFIXES)


def _resolve_ollama_num_predict(model: str, max_image_edge: int) -> int:
    """Choose a model-aware token budget for Ollama responses."""
    if _is_qwen_ollama_model(model):
        if max_image_edge <= OLLAMA_QWEN_SMALL_EDGE_THRESHOLD:
            return OLLAMA_QWEN_NUM_PREDICT_SMALL_EDGE
        return OLLAMA_QWEN_NUM_PREDICT_LARGE_EDGE
    if _is_gemma4_ollama_model(model):
        return OLLAMA_GEMMA4_NUM_PREDICT
    return OLLAMA_DEFAULT_NUM_PREDICT


_THINKING_PREAMBLE_RE = re.compile(
    r"(?i)^(?:"
    r"here[\u2019']?s a (?:thinking|step-by-step|reasoning)[^.]*\.\s*\d*\.?\s*"
    r"|(?:let me|i will|i'll|i need to) (?:analyze|think|evaluate|assess|review|examine)[^.]*\.\s*"
    r"|(?:step|thought|thinking)[\s:]+\d+[.:]\s*"
    r")"
)


def _sanitize_vision_one_line(raw_value: object, *, fallback_text: str = "") -> str:
    """Clean up one-line summaries from partially formatted model output."""
    line = str(raw_value or "").strip()
    if not line:
        first_sentence = re.split(r"(?<=[.!?])\s+", fallback_text.strip(), maxsplit=1)[0].strip()
        line = first_sentence if first_sentence else "Vision scoring summary"

    line = re.sub(r"\s+", " ", line).strip().strip("`")
    if line.startswith(("'", '"')):
        line = line[1:].strip()
    if line.endswith(("'", '"')):
        line = line[:-1].strip()

    # Strip model thinking-chain preambles (e.g. Gemma 4's "Here's a thinking process...")
    stripped = _THINKING_PREAMBLE_RE.sub("", line).strip()
    if len(stripped) >= 20:
        line = stripped
    elif _THINKING_PREAMBLE_RE.match(line):
        line = ""

    return line[:220] if line else "Vision scoring summary"


def _normalize_ollama_vision_payload(vision: dict, *, fallback_text: str = "") -> dict:
    """Normalize parsed vision payload to stable score ranges and text."""
    if not isinstance(vision, dict):
        raise ValueError("Ollama vision response must be a JSON object")
    candidate = dict(vision)
    candidate["one_line"] = _sanitize_vision_one_line(
        candidate.get("one_line", ""), fallback_text=fallback_text
    )
    # Ollama responses historically treat missing or malformed individual
    # criteria as neutral, and the total is derived from the six criteria.
    # Keep that parser contract while sending the resulting payload through
    # the shared finite/range validator.
    malformed_score = False
    for key in VISION_SCORE_KEYS:
        raw = candidate.get(key, 5)
        try:
            value = float(raw) if not isinstance(raw, bool) else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            candidate[key] = 5
            malformed_score = True
        else:
            candidate[key] = value
    candidate.setdefault("total", sum(candidate[key] for key in VISION_SCORE_KEYS))
    if malformed_score:
        candidate["total"] = sum(
            max(0, min(10, round(float(candidate[key])))) for key in VISION_SCORE_KEYS
        )
    return normalize_vision_payload(candidate, fallback_text=fallback_text)


def _extract_json_payload(raw_text: str) -> str:
    """Extract JSON body from a model response that may include code fences."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


def _parse_ollama_message_json_with_mode(body: str, parsed: dict) -> tuple[dict, str]:
    """Parse Ollama message and return both normalized payload and parse mode."""
    message = parsed.get("message") if isinstance(parsed, dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"Ollama response did not include message object: {body[:300]}")

    content = str(message.get("content") or "").strip()
    thinking = str(message.get("thinking") or "").strip()
    for source, text in (("content", content), ("thinking", thinking)):
        if not text:
            continue
        try:
            parsed_json = json.loads(_extract_json_payload(text))
            return _normalize_ollama_vision_payload(
                parsed_json, fallback_text=text
            ), f"json-{source}"
        except Exception:
            continue

    for source, text in (("content", content), ("thinking", thinking)):
        fallback = _parse_ollama_plaintext_scores(text)
        if fallback is not None:
            fallback["total"] = sum(fallback[key] for key in VISION_SCORE_KEYS)
            return _normalize_ollama_vision_payload(fallback, fallback_text=text), f"plain-{source}"

    for source, text in (("content", content), ("thinking", thinking)):
        fallback = _ollama_neutral_fallback_from_text(text)
        if fallback is not None:
            return _normalize_ollama_vision_payload(
                fallback, fallback_text=text
            ), f"neutral-{source}"

    raise RuntimeError(
        f"Ollama response did not include parseable JSON in message content/thinking: {body[:300]}"
    )


def _parse_ollama_message_json(body: str, parsed: dict) -> dict:
    """Compatibility wrapper returning only the normalized parsed payload."""
    payload, _ = _parse_ollama_message_json_with_mode(body, parsed)
    return payload


def _parse_ollama_plaintext_scores(raw_text: str) -> Optional[dict]:
    """Fallback parser for rubric-like plain text when model ignores JSON format."""
    text = (raw_text or "").strip()
    if not text:
        return None

    key_patterns = {
        "subject_clarity": r"subject[\s_-]*clarity",
        "lighting": r"lighting",
        "color_pop": r"color[\s_-]*pop",
        "emotion": r"emotion",
        "scroll_stop": r"scroll[\s_-]*stop",
        "crop_4x5": r"crop[\s_-]*4x5",
    }

    def _extract_score(label_pattern: str) -> Optional[float]:
        patterns = [
            rf"(?is){label_pattern}\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
            rf"(?is){label_pattern}[^0-9]{{0,48}}([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
            rf"(?is){label_pattern}\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"(?is){label_pattern}[^0-9]{{0,48}}([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        return None

    scores: dict[str, int] = {}
    for key, label_pattern in key_patterns.items():
        score = _extract_score(label_pattern)
        if score is None:
            continue
        scores[key] = int(max(0, min(10, round(score))))

    if len(scores) < 3:
        return None

    present_values = list(scores.values())
    fill_value = int(round(sum(present_values) / len(present_values))) if present_values else 5
    fill_value = max(0, min(10, fill_value))
    for key in key_patterns:
        scores.setdefault(key, fill_value)

    total_match = re.search(r"(?is)\btotal\s*[:=-]?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if total_match:
        total = int(max(0, min(60, round(float(total_match.group(1))))))
    else:
        total = sum(scores[k] for k in key_patterns)

    one_line = ""
    one_line_match = re.search(r"(?is)\bone[\s_-]*line\s*[:=-]\s*(.+)", text)
    if one_line_match:
        one_line = one_line_match.group(1).strip().splitlines()[0]
    if not one_line:
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        one_line = first_sentence[:180] if first_sentence else "Vision scoring summary"

    return {
        "subject_clarity": scores["subject_clarity"],
        "lighting": scores["lighting"],
        "color_pop": scores["color_pop"],
        "emotion": scores["emotion"],
        "scroll_stop": scores["scroll_stop"],
        "crop_4x5": scores["crop_4x5"],
        "total": total,
        "one_line": one_line,
    }


def _ollama_neutral_fallback_from_text(raw_text: str) -> Optional[dict]:
    """Last-resort fallback when model returns prose without structured numeric output."""
    text = (raw_text or "").strip()
    if not text:
        return None

    values: list[float] = []
    for match in re.finditer(r"(?is)\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text):
        try:
            values.append(float(match.group(1)))
        except Exception:
            continue
    if values:
        values.sort()
        mid = len(values) // 2
        base = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2.0
        score = int(max(0, min(10, round(base))))
    else:
        score = 5

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    one_line = first_sentence[:180] if first_sentence else "Vision response prose; neutral fallback"
    total = int(max(0, min(60, score * 6)))

    return {
        "subject_clarity": score,
        "lighting": score,
        "color_pop": score,
        "emotion": score,
        "scroll_stop": score,
        "crop_4x5": score,
        "total": total,
        "one_line": one_line,
    }


def _encode_image_for_ollama(
    image_path: Path,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> str:
    """Encode image as base64, downscaling/compressing to reduce Ollama payload size."""
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")


def score_with_ollama(
    image_path: Path,
    base_url: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    use_yolo_context: bool = True,
    prompt: Optional[str] = None,
    timeout_seconds: int = 300,
    max_image_edge: int = 1024,
    jpeg_quality: int = 80,
    keep_alive: str = "10m",
    detect_subject=None,
    encode_image=None,
    open_url=urlopen,
) -> dict:
    """Score a single image using Ollama's vision API (/api/chat)."""
    yolo_context = ""
    if use_yolo_context and detect_subject is not None:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                detection = detect_subject(img, debug=False)
                if detection:
                    x, y, w, h, class_name, conf = detection
                    img_h, img_w = img.shape[:2]
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    size_ratio = (w * h) / (img_w * img_h)
                    h_pos = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
                    v_pos = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
                    position = (
                        f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "centered"
                    )
                    size_desc = (
                        "large" if size_ratio > 0.3 else "medium" if size_ratio > 0.1 else "small"
                    )
                    yolo_context = (
                        f"\n\n**Detected Subject**: {class_name} "
                        f"({position}, {size_desc}, confidence: {conf:.0%})"
                    )
        except Exception:
            pass

    encoder = encode_image or _encode_image_for_ollama
    image_data = encoder(
        image_path,
        max_edge=max_image_edge,
        jpeg_quality=jpeg_quality,
    )

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    base_prompt = prompt or build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    if yolo_context:
        base_prompt = base_prompt + yolo_context

    _is_gemma4 = _is_gemma4_ollama_model(model)

    def _send_ollama_request(
        *, active_prompt: str, response_format: object, num_predict: int
    ) -> tuple[str, dict]:
        # Gemma 4: omit `think` entirely — setting think=false breaks `format` (Ollama bug #15260)
        # Gemma 4: use temperature=0.3 per Google's recommendations for more reliable output
        payload: dict = {
            "model": model,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0.3 if _is_gemma4 else 0, "num_predict": num_predict},
        }
        # Gemma 4: never use the `format` parameter — it causes tier-collapse (all criteria score
        # the same value) regardless of prompt variant. All prompts already instruct JSON output
        # so parsing still works without schema enforcement.
        if not _is_gemma4:
            payload["format"] = response_format
        messages: list[dict] = []
        if _is_gemma4:
            messages.append({"role": "system", "content": OLLAMA_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": active_prompt, "images": [image_data]})
        payload["messages"] = messages
        if not _is_gemma4:
            payload["think"] = False
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with open_url(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            details = ""
            try:
                details = error.read().decode("utf-8", errors="ignore")
            except Exception:
                details = ""
            raise RuntimeError(
                f"Ollama request failed ({error.code}) at {endpoint}: {details[:300]}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Ollama connection failed for {endpoint}: {error}") from error
        return body, json.loads(body)

    use_structured_profile = _is_qwen_ollama_model(model) or _is_gemma4
    if use_structured_profile:
        account_context = _extract_account_context_from_prompt(base_prompt)
        if _is_gemma4:
            primary_prompt = build_vision_prompt(account_context)
        else:
            primary_prompt = build_ollama_compact_json_prompt(account_context)
        if yolo_context:
            primary_prompt = primary_prompt + yolo_context
        primary_format: object = OLLAMA_STRICT_JSON_SCHEMA
    else:
        primary_prompt = base_prompt
        primary_format = "json"

    num_predict = _resolve_ollama_num_predict(model, max_image_edge=max_image_edge)
    body, parsed = _send_ollama_request(
        active_prompt=primary_prompt,
        response_format=primary_format,
        num_predict=num_predict,
    )
    payload, parse_mode = _parse_ollama_message_json_with_mode(body, parsed)

    # Retry once when first pass degraded into plain/neutral fallback output.
    if parse_mode.startswith("plain-") or parse_mode.startswith("neutral-"):
        retry_context = _extract_account_context_from_prompt(base_prompt)
        if _is_gemma4:
            retry_prompt = build_vision_prompt(retry_context)
        else:
            retry_prompt = build_ollama_compact_json_prompt(retry_context)
        if yolo_context:
            retry_prompt = retry_prompt + yolo_context
        retry_body, retry_parsed = _send_ollama_request(
            active_prompt=retry_prompt,
            response_format=OLLAMA_STRICT_JSON_SCHEMA,
            num_predict=num_predict,
        )
        retry_payload, retry_mode = _parse_ollama_message_json_with_mode(retry_body, retry_parsed)
        if retry_mode.startswith("json-"):
            return retry_payload
        return retry_payload

    return payload


def _is_retryable_ollama_error(error: Exception) -> bool:
    text = str(error).lower()
    retryable_markers = [
        "timed out",
        "timeout",
        "connection failed",
        "connection reset",
        "temporarily unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(marker in text for marker in retryable_markers)


def _is_model_not_found_error(error: Exception) -> bool:
    text = str(error).lower()
    return "not_found_error" in text and "model" in text


def _claude_setup_hint(error: Exception) -> str:
    """Return a practical setup hint for common Claude initialization failures."""
    text = str(error).lower()
    if "no module named" in text or "import" in text:
        return (
            "Install Claude dependency in your active environment:\n"
            "  python -m pip install -e '.[claude]'"
        )
    if _is_model_not_found_error(error):
        return (
            "Claude model not found for your account.\n"
            f"Try: --claude-model {DEFAULT_CLAUDE_MODEL}\n"
            "or set ANTHROPIC_MODEL in your environment/.env."
        )
    return (
        "Claude initialization failed. Verify dependency and API key setup.\n"
        'Run: python -c "import anthropic; print(anthropic.__version__)".'
    )


def _ollama_setup_hint(error: Exception) -> str:
    """Return a practical setup hint for common Ollama initialization failures."""
    return (
        "Ollama setup failed. Verify PICKINSTA_OLLAMA_BASE_URL points to a running Ollama server,\n"
        "and that PICKINSTA_OLLAMA_MODEL is already pulled on that server (for example: qwen2.5vl:7b).\n"
        "You can increase client timeout with PICKINSTA_OLLAMA_TIMEOUT_SEC.\n"
        f"Original error: {error}"
    )
