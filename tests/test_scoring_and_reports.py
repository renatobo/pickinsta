import json
from pathlib import Path
import sys
import types

import pickinsta.ig_image_selector as selector
from pickinsta.ig_image_selector import ImageScore


def test_write_markdown_report_contains_expected_sections_and_escapes(tmp_path) -> None:
    report_path = tmp_path / "selection_report.md"
    selected_report = [
        {
            "rank": 1,
            "filename": "bike|shot.jpg",
            "final_score": 0.9123,
            "technical_composite": 0.7,
            "vision_total": 54,
            "one_line": "High impact\nGreat lead room",
            "output": "01_bike.jpg",
        }
    ]
    analyzed_items = [
        ImageScore(
            path=Path("work_1.jpg"),
            source_path=Path("source_1.jpg"),
            technical={"composite": 0.7},
            vision={
                "total": 54,
                "subject_clarity": 9,
                "lighting": 8,
                "color_pop": 8,
                "emotion": 9,
                "scroll_stop": 10,
                "crop_4x5": 10,
            },
            final_score=0.9123,
            one_line="Great frame",
        )
    ]

    selector.write_markdown_report(
        report_path,
        input_folder=Path("input"),
        output_folder=Path("selected"),
        scorer="claude",
        top_n=1,
        selected_report=selected_report,
        analyzed_items=analyzed_items,
    )

    content = report_path.read_text(encoding="utf-8")
    assert "# pickinsta Selection Report" in content
    assert "## Top Selected Outputs" in content
    assert "## Claude Analysis (All Images Analyzed)" in content
    assert "bike\\|shot.jpg" in content
    assert "High impact Great lead room" in content


def test_batch_technical_score_sorts_results_and_skips_failures(tmp_path, monkeypatch) -> None:
    good_a = tmp_path / "a.jpg"
    good_b = tmp_path / "b.jpg"
    bad = tmp_path / "bad.jpg"
    for p in (good_a, good_b, bad):
        p.write_bytes(b"x")

    scores = {"a.jpg": 0.3, "b.jpg": 0.8}

    def fake_score_technical(path: Path):
        if path.name == "bad.jpg":
            raise ValueError("cannot score")
        return {"composite": scores[path.name]}

    source_map = {good_a: Path("source_a.jpg"), good_b: Path("source_b.jpg"), bad: Path("source_bad.jpg")}
    monkeypatch.setattr(selector, "score_technical", fake_score_technical)

    ranked = selector.batch_technical_score([good_a, bad, good_b], source_map=source_map)

    assert [item.path.name for item in ranked] == ["b.jpg", "a.jpg"]
    assert ranked[0].source_path == Path("source_b.jpg")
    assert ranked[1].source_path == Path("source_a.jpg")


def test_batch_vision_score_clip_uses_weighted_formula_and_ranks(monkeypatch) -> None:
    candidates = [
        ImageScore(path=Path("one.jpg"), technical={"composite": 0.4}),
        ImageScore(path=Path("two.jpg"), technical={"composite": 0.7}),
    ]

    monkeypatch.setattr(selector, "resolve_optional_hf_token", lambda search_dir=None: None)
    monkeypatch.setattr(selector, "load_clip_model", lambda: ("fake-model", "fake-processor"))

    def fake_score_with_clip(path: Path, model=None, processor=None):
        assert model == "fake-model"
        assert processor == "fake-processor"
        if path.name == "one.jpg":
            return {"total": 54, "one_line": "Excellent"}
        return {"total": 36, "one_line": "Average"}

    monkeypatch.setattr(selector, "score_with_clip", fake_score_with_clip)

    ranked = selector.batch_vision_score(candidates, scorer="clip")

    assert ranked[0].path.name == "one.jpg"
    assert ranked[0].final_score == 0.4 * 0.3 + (54 / 60.0) * 0.7
    assert ranked[1].final_score == 0.7 * 0.3 + (36 / 60.0) * 0.7
    assert ranked[0].one_line == "Excellent"


def test_batch_vision_score_clip_falls_back_to_technical_on_model_error(monkeypatch) -> None:
    candidates = [
        ImageScore(path=Path("x.jpg"), technical={"composite": 0.62}),
        ImageScore(path=Path("y.jpg"), technical={"composite": 0.41}),
    ]

    monkeypatch.setattr(selector, "resolve_optional_hf_token", lambda search_dir=None: None)

    def fail_load_clip_model():
        raise RuntimeError("model init failed")

    monkeypatch.setattr(selector, "load_clip_model", fail_load_clip_model)

    ranked = selector.batch_vision_score(candidates, scorer="clip")

    assert [item.path.name for item in ranked] == ["x.jpg", "y.jpg"]
    assert ranked[0].final_score == 0.62
    assert ranked[1].final_score == 0.41
    assert "CLIP unavailable" in ranked[0].one_line


def test_batch_vision_score_claude_uses_resolved_anthropic_key(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "candidate.jpg"
    image_file.write_bytes(b"image-bytes")
    candidate = ImageScore(path=image_file, source_path=image_file, technical={"composite": 0.5})

    observed = {"api_key": None, "model": None}

    class FakeMessages:
        def create(self, *, model, max_tokens, messages):
            observed["model"] = model
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok")])

    class FakeAnthropicClient:
        def __init__(self, *, api_key):
            observed["api_key"] = api_key
            self.messages = FakeMessages()

    fake_anthropic_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    monkeypatch.setattr(selector, "resolve_anthropic_api_key", lambda search_dir=None: "sk-test-key")
    monkeypatch.setattr(selector, "resolve_claude_model", lambda cli_model=None: "claude-sonnet-4-5")
    monkeypatch.setattr(selector, "_claude_model_candidates", lambda preferred: [preferred])
    monkeypatch.setattr(selector, "_file_sha256", lambda _path: "sha")
    monkeypatch.setattr(selector, "load_claude_score_from_file_cache", lambda **_kwargs: None)
    monkeypatch.setattr(selector, "save_claude_score_to_file_cache", lambda **_kwargs: None)
    monkeypatch.setattr(
        selector,
        "score_with_claude",
        lambda *_args, **_kwargs: {"total": 48, "crop_4x5": 9, "one_line": "Claude verified"},
    )

    ranked = selector.batch_vision_score([candidate], scorer="claude")

    assert observed["api_key"] == "sk-test-key"
    assert observed["model"] == "claude-sonnet-4-5"
    assert ranked[0].final_score == 0.5 * 0.3 + (48 / 60.0) * 0.7
    assert ranked[0].one_line == "Claude verified"


def test_batch_vision_score_claude_applies_strong_crop_gate(tmp_path, monkeypatch) -> None:
    low_crop = tmp_path / "low.jpg"
    high_crop = tmp_path / "high.jpg"
    low_crop.write_bytes(b"low")
    high_crop.write_bytes(b"high")

    candidates = [
        ImageScore(path=low_crop, source_path=low_crop, technical={"composite": 0.5}),
        ImageScore(path=high_crop, source_path=high_crop, technical={"composite": 0.5}),
    ]

    class FakeMessages:
        def create(self, *, model, max_tokens, messages):
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="ok")])

    class FakeAnthropicClient:
        def __init__(self, *, api_key):
            self.messages = FakeMessages()

    fake_anthropic_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_module)

    monkeypatch.setattr(selector, "resolve_anthropic_api_key", lambda search_dir=None: "sk-test-key")
    monkeypatch.setattr(selector, "resolve_claude_model", lambda cli_model=None: "claude-sonnet-4-5")
    monkeypatch.setattr(selector, "_claude_model_candidates", lambda preferred: [preferred])
    monkeypatch.setattr(selector, "_file_sha256", lambda _path: "sha")
    monkeypatch.setattr(selector, "load_claude_score_from_file_cache", lambda **_kwargs: None)
    monkeypatch.setattr(selector, "save_claude_score_to_file_cache", lambda **_kwargs: None)

    def fake_score_with_claude(path: Path, **_kwargs):
        if path.name == "low.jpg":
            return {"total": 48, "crop_4x5": 4, "one_line": "Low crop confidence"}
        return {"total": 48, "crop_4x5": 9, "one_line": "High crop confidence"}

    monkeypatch.setattr(selector, "score_with_claude", fake_score_with_claude)

    ranked = selector.batch_vision_score(candidates, scorer="claude")

    base = 0.5 * 0.3 + (48 / 60.0) * 0.7
    assert ranked[0].path.name == "high.jpg"
    assert ranked[0].final_score == base
    assert ranked[1].path.name == "low.jpg"
    assert ranked[1].final_score == base * 0.15


def test_batch_vision_score_ollama_uses_resolved_server_and_model(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "candidate.jpg"
    image_file.write_bytes(b"image-bytes")
    candidate = ImageScore(path=image_file, source_path=image_file, technical={"composite": 0.5})

    observed = {"base_url": None, "model": None}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b'{"models":[]}'

    monkeypatch.setattr(selector, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(selector, "resolve_ollama_base_url", lambda search_dir=None: "http://remote:11434")
    monkeypatch.setattr(selector, "resolve_ollama_model", lambda search_dir=None: "qwen2.5vl:7b")
    monkeypatch.setattr(selector, "resolve_ollama_keep_alive", lambda search_dir=None: "10m")
    monkeypatch.setattr(selector, "resolve_account_context", lambda search_dir=None: "motorcycle account")

    def fake_score_with_ollama(
        path: Path,
        *,
        base_url,
        model,
        use_yolo_context,
        prompt,
        timeout_seconds,
        max_image_edge,
        jpeg_quality,
        keep_alive,
    ):
        observed["base_url"] = base_url
        observed["model"] = model
        assert use_yolo_context is False
        assert "motorcycle account" in prompt
        assert path == image_file
        assert timeout_seconds >= 300
        assert max_image_edge == 1024
        assert jpeg_quality == 80
        assert keep_alive == "10m"
        return {"total": 48, "crop_4x5": 9, "one_line": "Ollama verified"}

    monkeypatch.setattr(selector, "score_with_ollama", fake_score_with_ollama)

    ranked = selector.batch_vision_score([candidate], scorer="ollama")

    assert observed["base_url"] == "http://remote:11434"
    assert observed["model"] == "qwen2.5vl:7b"
    assert ranked[0].final_score == 0.5 * 0.3 + (48 / 60.0) * 0.7
    assert ranked[0].one_line == "Ollama verified"


def test_batch_vision_score_ollama_retries_and_succeeds(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "retry.jpg"
    image_file.write_bytes(b"image-bytes")
    candidate = ImageScore(path=image_file, source_path=image_file, technical={"composite": 0.4})

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            return b'{"models":[]}'

    attempts = {"n": 0}

    monkeypatch.setattr(selector, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(selector, "resolve_ollama_base_url", lambda search_dir=None: "http://remote:11434")
    monkeypatch.setattr(selector, "resolve_ollama_model", lambda search_dir=None: "qwen2.5vl:7b")
    monkeypatch.setattr(selector, "resolve_ollama_keep_alive", lambda search_dir=None: "10m")
    monkeypatch.setattr(selector, "resolve_account_context", lambda search_dir=None: "motorcycle account")
    monkeypatch.setattr(selector, "resolve_ollama_concurrency", lambda: 1)
    monkeypatch.setattr(selector, "resolve_ollama_max_retries", lambda: 2)
    monkeypatch.setattr(selector, "resolve_ollama_retry_backoff_seconds", lambda: 0.01)
    monkeypatch.setattr(selector, "resolve_ollama_circuit_breaker_errors", lambda: 6)
    monkeypatch.setattr(selector.time, "sleep", lambda _sec: None)

    def flaky_score(*_args, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Ollama request failed (500)")
        return {"total": 42, "crop_4x5": 8, "one_line": "Recovered after retry"}

    monkeypatch.setattr(selector, "score_with_ollama", flaky_score)

    ranked = selector.batch_vision_score([candidate], scorer="ollama")

    assert attempts["n"] == 2
    expected = (0.4 * 0.3 + (42 / 60.0) * 0.7) * 1.0
    assert ranked[0].final_score == expected
    assert ranked[0].one_line == "Recovered after retry"


def test_score_with_ollama_parses_json_from_thinking_when_content_is_empty(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "thinking.jpg"
    image_file.write_bytes(b"fake-image")
    monkeypatch.setattr(selector, "_encode_image_for_ollama", lambda *_args, **_kwargs: "b64")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {
                "model": "qwen3-vl:8b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": '{"total": 47, "crop_4x5": 9, "one_line": "Strong action frame"}',
                },
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(selector, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    result = selector.score_with_ollama(
        image_path=image_file,
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b",
        use_yolo_context=False,
    )

    assert result["total"] == 47
    assert result["crop_4x5"] == 9
    assert "Strong action frame" in result["one_line"]


def test_score_with_ollama_sets_think_false_in_payload(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "payload.jpg"
    image_file.write_bytes(b"fake-image")
    monkeypatch.setattr(selector, "_encode_image_for_ollama", lambda *_args, **_kwargs: "b64")
    observed = {"think": None}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {"message": {"role": "assistant", "content": '{"total": 40, "crop_4x5": 8}'}}
            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        observed_payload = json.loads(request.data.decode("utf-8"))
        observed["think"] = observed_payload.get("think")
        return FakeResponse()

    monkeypatch.setattr(selector, "urlopen", fake_urlopen)

    selector.score_with_ollama(
        image_path=image_file,
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b",
        use_yolo_context=False,
    )

    assert observed["think"] is False


def test_score_with_ollama_parses_plaintext_thinking_rubric(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "thinking-plain.jpg"
    image_file.write_bytes(b"fake-image")
    monkeypatch.setattr(selector, "_encode_image_for_ollama", lambda *_args, **_kwargs: "b64")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {
                "model": "qwen3-vl:8b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": (
                        "1. SUBJECT_CLARITY: strong separation, score 8/10\n"
                        "2. LIGHTING: balanced highlights 7/10\n"
                        "3. COLOR_POP: vibrant reds and blues 8/10\n"
                        "4. EMOTION: dynamic lean angle 7/10\n"
                        "5. SCROLL_STOP: eye-catching action 8/10\n"
                        "6. CROP_4x5: good portrait crop viability 9/10\n"
                        "TOTAL: 47\n"
                    ),
                },
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(selector, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    result = selector.score_with_ollama(
        image_path=image_file,
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b",
        use_yolo_context=False,
    )

    assert result["subject_clarity"] == 8
    assert result["lighting"] == 7
    assert result["color_pop"] == 8
    assert result["emotion"] == 7
    assert result["scroll_stop"] == 8
    assert result["crop_4x5"] == 9
    assert result["total"] == 47


def test_score_with_ollama_uses_neutral_fallback_for_prose_only_thinking(tmp_path, monkeypatch) -> None:
    image_file = tmp_path / "thinking-prose.jpg"
    image_file.write_bytes(b"fake-image")
    monkeypatch.setattr(selector, "_encode_image_for_ollama", lambda *_args, **_kwargs: "b64")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def read(self):
            payload = {
                "model": "qwen3-vl:8b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": (
                        "Got it, let's break down each criterion for this motorcycle photo. "
                        "The rider is a clear focal point and the background is not too distracting."
                    ),
                },
            }
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(selector, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    result = selector.score_with_ollama(
        image_path=image_file,
        base_url="http://127.0.0.1:11434",
        model="qwen3-vl:8b",
        use_yolo_context=False,
    )

    assert result["subject_clarity"] == 5
    assert result["lighting"] == 5
    assert result["color_pop"] == 5
    assert result["emotion"] == 5
    assert result["scroll_stop"] == 5
    assert result["crop_4x5"] == 5
    assert result["total"] == 30
