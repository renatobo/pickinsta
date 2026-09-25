"""Claude vision transport with lazy optional dependency loading."""

import base64
import io
import json
from pathlib import Path
from typing import Optional

import cv2
from PIL import Image

from pickinsta.config import DEFAULT_ACCOUNT_CONTEXT, DEFAULT_CLAUDE_MODEL
from pickinsta.vision.prompts import build_vision_prompt

CLAUDE_SCORING_MAX_EDGE = 1024
CLAUDE_SCORING_JPEG_QUALITY = 75


def score_with_claude(
    image_path: Path,
    api_key: Optional[str] = None,
    model: str = DEFAULT_CLAUDE_MODEL,
    client=None,
    use_yolo_context: bool = True,
    prompt: Optional[str] = None,
    detect_subject=None,
) -> dict:
    """
    Score a single image using Claude's vision API.

    Args:
        image_path: Path to image to score
        api_key: Claude API key
        model: Claude model to use
        client: Optional pre-initialized Claude client
        use_yolo_context: If True, detect subjects with YOLO and enhance prompt
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

    # Optional: Detect subjects with YOLO to enhance Claude's context
    yolo_context = ""
    if use_yolo_context and detect_subject is not None:
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                detection = detect_subject(img, debug=False)
                if detection:
                    x, y, w, h, class_name, conf = detection
                    img_h, img_w = img.shape[:2]

                    # Calculate relative position
                    center_x = (x + w / 2) / img_w
                    center_y = (y + h / 2) / img_h
                    size_ratio = (w * h) / (img_w * img_h)

                    # Describe position
                    h_pos = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
                    v_pos = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"
                    position = (
                        f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "centered"
                    )

                    # Describe size
                    size_desc = (
                        "large" if size_ratio > 0.3 else "medium" if size_ratio > 0.1 else "small"
                    )

                    yolo_context = f"\n\n**Detected Subject**: {class_name} ({position}, {size_desc}, confidence: {conf:.0%})"
        except Exception:
            # Silently fail - YOLO context is optional
            pass

    # Downsize for scoring — 1024px is sufficient for composition evaluation
    # and significantly reduces API token cost vs sending full 1920px images.
    with Image.open(image_path) as img:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        longest = max(w, h)
        if longest > CLAUDE_SCORING_MAX_EDGE:
            scale = CLAUDE_SCORING_MAX_EDGE / longest
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=CLAUDE_SCORING_JPEG_QUALITY, optimize=True)
        image_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    media_type = "image/jpeg"

    # Enhance prompt with YOLO detection context if available
    enhanced_prompt = prompt or build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    if yolo_context:
        enhanced_prompt = enhanced_prompt + yolo_context

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": enhanced_prompt},
                ],
            }
        ],
    )

    content = getattr(response, "content", None)
    if not isinstance(content, (list, tuple)) or not content:
        raise RuntimeError("Claude response did not include content")
    raw = getattr(content[0], "text", None)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("Claude response did not include text content")
    try:
        payload = json.loads(_extract_json_payload(raw))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Claude response did not include valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Claude response JSON must be an object")
    return payload


def _extract_json_payload(raw_text: str) -> str:
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end > start else raw
