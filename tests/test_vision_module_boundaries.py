"""Direct contracts for extracted vision modules."""

import json

from pickinsta import ig_image_selector as selector
from pickinsta.vision import ollama, prompts


def test_selector_prompt_exports_match_extracted_contract() -> None:
    context = "A track-focused Ducati account"
    assert selector.build_vision_prompt(context) == prompts.build_vision_prompt(context)
    assert selector.build_ollama_compact_json_prompt(
        context
    ) == prompts.build_ollama_compact_json_prompt(context)


def test_ollama_parser_normalizes_json_from_thinking() -> None:
    payload = {
        "message": {
            "content": "",
            "thinking": json.dumps(
                {
                    "subject_clarity": 9,
                    "lighting": 8,
                    "color_pop": 7,
                    "emotion": 8,
                    "scroll_stop": 9,
                    "crop_4x5": 8,
                    "total": 49,
                    "one_line": "Strong motorcycle action frame.",
                }
            ),
        }
    }
    result = ollama._parse_ollama_message_json(json.dumps(payload), payload)
    assert result["total"] == 49
    assert result["crop_4x5"] == 8


def test_selector_parser_alias_is_exact_extracted_function() -> None:
    assert selector._parse_ollama_plaintext_scores is ollama._parse_ollama_plaintext_scores
    assert selector._normalize_ollama_vision_payload is ollama._normalize_ollama_vision_payload
