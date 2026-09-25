"""Prompt contracts shared by remote vision scorers."""

import re

from pickinsta.config import DEFAULT_ACCOUNT_CONTEXT

VISION_PROMPT_TEMPLATE = """Score this motorcycle photo for Instagram cover potential.

Context: {account_context}.

Rate each criterion 1-10 using these professional composition guidelines:

1. SUBJECT_CLARITY: Is the motorcycle/rider the clear focal point? Does it stand out
   from the background at thumbnail size? Subject-to-background sharpness ratio ≥3:1
   is good; busy backgrounds that compete for attention score low.

2. LIGHTING: Quality of light — golden hour or dramatic low-sun light is ideal.
   Penalize if chrome/metallic highlights are blown out, or if > 2% of pixels are
   clipped black/white. Mean luminance should feel balanced (not flat midday).

3. COLOR_POP: Evaluate color harmony (complementary/analogous/triadic schemes).
   Does the bike's color contrast with the background? Orange bike on blue sky = high;
   matching bike-and-background colors = low. Moderate, consistent saturation preferred.

4. EMOTION: Does the image convey motion, tension, power, or aspiration? Low camera
   angles (tank/axle level) make bikes look powerful; standing eye-height shots score
   lower. 3/4 view is the most flattering standard angle.

5. SCROLL_STOP: Would this image stop fast-scrolling on Instagram? Consider: dramatic
   composition, strong leading lines, clean negative space, and clear visual hierarchy.

6. CROP_4x5: Can this be cropped to 3:4 portrait (1080x1440) while maintaining good
   composition? Consider:
   - Is the subject placed near a rule-of-thirds power point (not dead center)?
   - Is there adequate lead room (60-70% of space ahead of the motorcycle's facing direction)?
   - Would cropping to portrait cut off wheels, handlebars, or exhaust?
   - Would the subject remain well-composed in Instagram's 3:4 grid thumbnail?

BRAND BONUS: This is a Ducati-focused account. If you can identify the motorcycle as a Ducati
(by logo, livery, bodywork shape, or distinctive features like trellis frame, desmo, Panigale
fairings, etc.), add 2 bonus points to SUBJECT_CLARITY and EMOTION (max 10 each).
All other brands or unidentifiable bikes score normally — do NOT penalize them.

Return ONLY valid JSON, no markdown:
{{"subject_clarity": N, "lighting": N, "color_pop": N, "emotion": N, "scroll_stop": N, "crop_4x5": N, "total": N, "one_line": "why this works or doesn't"}}"""

OLLAMA_COMPACT_JSON_PROMPT_TEMPLATE = """Evaluate this motorcycle photo for Instagram cover potential.

Context: {account_context}.

Return ONLY a JSON object with keys:
- subject_clarity
- lighting
- color_pop
- emotion
- scroll_stop
- crop_4x5
- total
- one_line

Rules:
- Score each criterion as an integer from 0 to 10.
- total must equal the sum of the 6 criterion scores (0 to 60).
- one_line must be exactly one concise sentence describing this specific image.
- BRAND BONUS: This is a Ducati-focused account. If the motorcycle is identifiably a Ducati, add 2 bonus points to subject_clarity and emotion (max 10 each). All other brands or unidentifiable bikes score normally — do NOT penalize them.
"""


OLLAMA_SYSTEM_PROMPT = (
    "You are a professional motorsport photographer and Instagram content curator. "
    "You evaluate motorcycle photos for visual quality, composition, and social media impact. "
    "You always score each criterion independently based on what you observe — "
    "different aspects of a photo can have very different quality levels."
)

OLLAMA_STRICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subject_clarity": {"type": "integer", "minimum": 0, "maximum": 10},
        "lighting": {"type": "integer", "minimum": 0, "maximum": 10},
        "color_pop": {"type": "integer", "minimum": 0, "maximum": 10},
        "emotion": {"type": "integer", "minimum": 0, "maximum": 10},
        "scroll_stop": {"type": "integer", "minimum": 0, "maximum": 10},
        "crop_4x5": {"type": "integer", "minimum": 0, "maximum": 10},
        "total": {"type": "integer", "minimum": 0, "maximum": 60},
        "one_line": {"type": "string"},
    },
    "required": [
        "subject_clarity",
        "lighting",
        "color_pop",
        "emotion",
        "scroll_stop",
        "crop_4x5",
        "total",
        "one_line",
    ],
    "additionalProperties": False,
}

VISION_SCORE_KEYS = (
    "subject_clarity",
    "lighting",
    "color_pop",
    "emotion",
    "scroll_stop",
    "crop_4x5",
)


def build_vision_prompt(account_context: str) -> str:
    """Build the Claude vision prompt with account-specific context."""
    context = account_context.strip() or DEFAULT_ACCOUNT_CONTEXT
    return VISION_PROMPT_TEMPLATE.format(account_context=context)


def build_ollama_compact_json_prompt(account_context: str) -> str:
    """Build compact strict-json prompt for Ollama models."""
    context = account_context.strip() or DEFAULT_ACCOUNT_CONTEXT
    return OLLAMA_COMPACT_JSON_PROMPT_TEMPLATE.format(account_context=context)


def extract_account_context_from_prompt(prompt_text: str) -> str:
    """Best-effort extraction of account context from a full vision prompt."""
    text = (prompt_text or "").strip()
    if not text:
        return DEFAULT_ACCOUNT_CONTEXT

    match = re.search(
        r"(?is)\bcontext:\s*(.+?)(?:\.\s*rate each criterion|\.\s*return only|\n\n|$)",
        text,
    )
    if match:
        context = match.group(1).strip().rstrip(".").strip().strip('"').strip("'")
        if context:
            return context
    return DEFAULT_ACCOUNT_CONTEXT
