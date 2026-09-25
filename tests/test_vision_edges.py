"""Failure-path contracts for extracted vision infrastructure."""

import base64
import io
import json
import sys
import types
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
from PIL import Image

from pickinsta import ig_image_selector as selector
from pickinsta.models import ImageScore
from pickinsta.pipeline import vision_scoring
from pickinsta.vision import claude, ollama, prompts


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True
        return False

    def read(self):
        return self.body


def _image(path: Path, size: tuple[int, int] = (24, 12)) -> Path:
    Image.new("RGBA", size, (200, 20, 30, 128)).save(path)
    return path


def _vision(**overrides) -> dict:
    result = {
        "subject_clarity": 8,
        "lighting": 7,
        "color_pop": 6,
        "emotion": 8,
        "scroll_stop": 9,
        "crop_4x5": 7,
        "total": 45,
        "one_line": "Strong frame.",
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([], "did not include content"),
        ([object()], "did not include text"),
        ([types.SimpleNamespace(text="not json")], "valid JSON"),
        ([types.SimpleNamespace(text="[]")], "must be an object"),
    ],
)
def test_claude_rejects_malformed_content_without_optional_import(
    tmp_path, monkeypatch, content, message
) -> None:
    image_path = _image(tmp_path / "claude.png")
    client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=lambda **_kwargs: types.SimpleNamespace(content=content)
        )
    )
    monkeypatch.setitem(sys.modules, "anthropic", None)

    with pytest.raises(RuntimeError, match=message):
        claude.score_with_claude(image_path, client=client, use_yolo_context=False)


def test_claude_closes_image_and_ignores_yolo_context_failure(tmp_path, monkeypatch) -> None:
    image_path = _image(tmp_path / "claude.png", (2048, 1024))
    observed: dict = {}

    def create(**kwargs):
        observed.update(kwargs)
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=json.dumps(_vision()))])

    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(claude.cv2, "imread", lambda _path: np.zeros((10, 10, 3), dtype=np.uint8))

    result = claude.score_with_claude(
        image_path,
        client=client,
        detect_subject=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("yolo")),
    )

    assert result["total"] == 45
    sent = observed["messages"][0]["content"]
    assert "Detected Subject" not in sent[1]["text"]
    decoded = Image.open(io.BytesIO(base64.b64decode(sent[0]["source"]["data"])))
    assert max(decoded.size) == claude.CLAUDE_SCORING_MAX_EDGE
    with image_path.open("ab") as writable:
        writable.write(b"")


def test_claude_adds_detected_subject_context(tmp_path, monkeypatch) -> None:
    image_path = _image(tmp_path / "claude.png")
    observed = {}
    client = types.SimpleNamespace(
        messages=types.SimpleNamespace(
            create=lambda **kwargs: (
                observed.update(kwargs)
                or types.SimpleNamespace(
                    content=[types.SimpleNamespace(text=json.dumps(_vision()))]
                )
            )
        )
    )
    monkeypatch.setattr(claude.cv2, "imread", lambda _path: np.zeros((100, 200, 3)))
    claude.score_with_claude(
        image_path,
        client=client,
        prompt="Custom prompt",
        detect_subject=lambda *_args, **_kwargs: (0, 0, 40, 40, "motorcycle", 0.9),
    )
    prompt = observed["messages"][0]["content"][1]["text"]
    assert "motorcycle (top-left, small, confidence: 90%)" in prompt


def test_prompt_defaults_context_and_extracts_supported_families() -> None:
    assert "Ducati" in prompts.build_vision_prompt("  ")
    assert "Ducati" in prompts.build_ollama_compact_json_prompt("")
    assert prompts.extract_account_context_from_prompt("")
    assert (
        prompts.extract_account_context_from_prompt("Context: Track riders. Return ONLY JSON")
        == "Track riders"
    )
    assert prompts.extract_account_context_from_prompt("unrelated")


@pytest.mark.parametrize(
    ("model", "edge", "expected"),
    [
        ("qwen3-vl:8b", 512, 650),
        ("qwen2.5-vl:7b", 1024, 750),
        ("owner/gemma4:e4b", 1024, 350),
        ("llava:latest", 1024, 220),
    ],
)
def test_ollama_routes_model_token_profiles(model, edge, expected) -> None:
    assert ollama._resolve_ollama_num_predict(model, edge) == expected


def test_ollama_image_encoder_resizes_and_falls_back_to_raw_bytes(tmp_path) -> None:
    image_path = _image(tmp_path / "large.png", (100, 50))
    encoded = ollama._encode_image_for_ollama(image_path, max_edge=20, jpeg_quality=70)
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (20, 10)

    raw_path = tmp_path / "broken.jpg"
    raw_path.write_bytes(b"not-an-image")
    assert (
        base64.b64decode(ollama._encode_image_for_ollama(raw_path, max_edge=20, jpeg_quality=70))
        == b"not-an-image"
    )


def test_ollama_http_error_includes_endpoint_and_bounded_body(tmp_path) -> None:
    image_path = _image(tmp_path / "image.png")
    error = HTTPError("http://host/api/chat", 503, "down", {}, io.BytesIO(b"busy"))

    with pytest.raises(RuntimeError, match=r"503.*http://host/api/chat: busy"):
        ollama.score_with_ollama(
            image_path,
            "http://host/",
            encode_image=lambda *_args, **_kwargs: "b64",
            open_url=lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
        )


def test_ollama_url_error_is_classified_retryable(tmp_path) -> None:
    image_path = _image(tmp_path / "image.png")
    with pytest.raises(RuntimeError, match="connection failed") as caught:
        ollama.score_with_ollama(
            image_path,
            "http://host",
            encode_image=lambda *_args, **_kwargs: "b64",
            open_url=lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("reset")),
        )
    assert ollama._is_retryable_ollama_error(caught.value)
    assert not ollama._is_retryable_ollama_error(RuntimeError("invalid response"))


@pytest.mark.parametrize("body", [b"not-json", b"\xff\xfe", b"[]"])
def test_ollama_rejects_invalid_http_response_bodies(tmp_path, body) -> None:
    image_path = _image(tmp_path / "image.png")
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError, RuntimeError, AttributeError)):
        ollama.score_with_ollama(
            image_path,
            "http://host",
            encode_image=lambda *_args, **_kwargs: "b64",
            open_url=lambda *_args, **_kwargs: FakeResponse(body),
        )


def test_ollama_parser_modes_and_normalization_edges() -> None:
    payload, mode = ollama._parse_ollama_message_json_with_mode(
        "body",
        {"message": {"content": "", "thinking": "Lighting: 11/10; emotion: 4/10; crop 4x5: 6/10"}},
    )
    assert mode == "plain-thinking"
    assert payload["lighting"] == 10
    assert set(prompts.VISION_SCORE_KEYS) <= payload.keys()

    normalized = ollama._normalize_ollama_vision_payload(
        {
            "lighting": "bad",
            "emotion": -3,
            "scroll_stop": 20,
            "total": 2,
            "one_line": "Here's a thinking process. Great motorcycle composition.",
        }
    )
    assert normalized["lighting"] == 5
    assert normalized["emotion"] == 0
    assert normalized["scroll_stop"] == 10
    assert normalized["total"] == 30
    assert "thinking" not in normalized["one_line"].lower()


def test_ollama_parser_rejects_missing_message_and_empty_text() -> None:
    with pytest.raises(RuntimeError, match="message object"):
        ollama._parse_ollama_message_json("{}", {})
    with pytest.raises(RuntimeError, match="parseable JSON"):
        ollama._parse_ollama_message_json("body", {"message": {}})
    assert ollama._parse_ollama_plaintext_scores("lighting: 8/10") is None
    assert ollama._ollama_neutral_fallback_from_text("") is None


def test_shared_vision_schema_rejects_non_finite_and_bounds_scores() -> None:
    from pickinsta.vision.schema import normalize_vision_payload

    payload = {key: 99 for key in prompts.VISION_SCORE_KEYS}
    normalized = normalize_vision_payload(payload)
    assert all(normalized[key] == 10 for key in prompts.VISION_SCORE_KEYS)
    assert normalized["total"] == 60
    payload["lighting"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalize_vision_payload(payload)


def test_ollama_gemma_payload_omits_format_and_think_and_uses_system_prompt(tmp_path) -> None:
    image_path = _image(tmp_path / "image.png")
    observed = {}

    def open_url(request, **_kwargs):
        observed.update(json.loads(request.data))
        return FakeResponse(json.dumps({"message": {"content": json.dumps(_vision())}}).encode())

    ollama.score_with_ollama(
        image_path,
        "http://host",
        model="owner/gemma4:e4b",
        prompt="Context: Track riders.\n\n",
        encode_image=lambda *_args, **_kwargs: "b64",
        open_url=open_url,
    )
    assert "format" not in observed
    assert "think" not in observed
    assert observed["messages"][0]["role"] == "system"
    assert "Track riders" in observed["messages"][1]["content"]


def test_ollama_plaintext_first_pass_retries_and_returns_second_result(tmp_path) -> None:
    image_path = _image(tmp_path / "image.png")
    responses = iter(
        [
            {"message": {"content": "Good framing without numeric scores."}},
            {"message": {"content": json.dumps(_vision(total=48))}},
        ]
    )
    calls = []

    def open_url(request, **_kwargs):
        calls.append(json.loads(request.data))
        return FakeResponse(json.dumps(next(responses)).encode())

    result = ollama.score_with_ollama(
        image_path,
        "http://host",
        model="llava",
        encode_image=lambda *_args, **_kwargs: "b64",
        open_url=open_url,
    )
    assert result["total"] == 48
    assert len(calls) == 2
    assert isinstance(calls[1]["format"], dict)


def test_batch_vision_empty_input_does_not_initialize_collaborators() -> None:
    assert vision_scoring.batch_vision_score([], collaborators=object()) == []


def test_batch_ollama_all_failure_opens_circuit_breaker(tmp_path, monkeypatch) -> None:
    candidates = []
    for index in range(5):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"image")
        candidates.append(ImageScore(path=path, source_path=path, technical={"composite": 0.5}))

    monkeypatch.setattr(
        selector, "urlopen", lambda *_args, **_kwargs: FakeResponse(b'{"models":[]}')
    )
    monkeypatch.setattr(selector, "resolve_ollama_base_url", lambda search_dir=None: "http://host")
    monkeypatch.setattr(selector, "resolve_ollama_model", lambda search_dir=None: "llava")
    monkeypatch.setattr(selector, "resolve_ollama_keep_alive", lambda search_dir=None: "1m")
    monkeypatch.setattr(selector, "resolve_ollama_concurrency", lambda: 1)
    monkeypatch.setattr(selector, "resolve_ollama_max_retries", lambda: 0)
    monkeypatch.setattr(selector, "resolve_ollama_circuit_breaker_errors", lambda: 2)
    monkeypatch.setattr(
        selector,
        "score_with_ollama",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad response")),
    )

    ranked = selector.batch_vision_score(candidates, scorer="ollama")

    assert all(item.final_score == 0.15 for item in ranked)
    assert sum("circuit breaker active" in item.one_line.lower() for item in ranked) == 3


def test_batch_claude_preloaded_cache_notifies_observer_without_api_call(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cached.jpg"
    path.write_bytes(b"image")
    item = ImageScore(path=path, source_path=path, technical={"composite": 0.5})
    cached = _vision(total=54, crop_4x5=9)
    observations = []

    class Messages:
        def create(self, **_kwargs):
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok")])

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(
            Anthropic=lambda **_kwargs: types.SimpleNamespace(messages=Messages())
        ),
    )
    monkeypatch.setattr(selector, "resolve_anthropic_api_key", lambda search_dir=None: "key")
    monkeypatch.setattr(selector, "resolve_claude_model", lambda cli_model=None: "model")
    monkeypatch.setattr(selector, "_claude_model_candidates", lambda _preferred: ["model"])
    monkeypatch.setattr(selector, "_file_sha256", lambda _path: "sha")
    monkeypatch.setattr(
        selector,
        "score_with_claude",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    ranked = selector.batch_vision_score(
        [item],
        scorer="claude",
        preloaded_vision_cache={path: cached},
        cache_observer=observations.append,
    )

    assert observations == [True]
    assert ranked[0].vision == cached
    assert ranked[0].one_line == "Strong frame."
