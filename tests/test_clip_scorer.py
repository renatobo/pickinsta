import sys
import types
from pathlib import Path

import numpy as np
import pytest

import pickinsta.clip_scorer as clip_scorer


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray([values], dtype=float)

    def softmax(self, *, dim):
        assert dim == 1
        return self

    def numpy(self):
        return self.values


class NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False


def install_fake_torch(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(no_grad=NoGrad))


def test_load_clip_model_uses_expected_pretrained_model(monkeypatch) -> None:
    calls = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            calls.append(("model", name))
            return "model"

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, name):
            calls.append(("processor", name))
            return "processor"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(CLIPModel=FakeModel, CLIPProcessor=FakeProcessor),
    )

    assert clip_scorer.load_clip_model() == ("model", "processor")
    assert calls == [
        ("model", "openai/clip-vit-large-patch14"),
        ("processor", "openai/clip-vit-large-patch14"),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModuleNotFoundError("No module named 'transformers'"), "pip install -e '.[clip]'"),
        (ImportError("cannot import name CLIPModel"), "pip install -e '.[clip]'"),
        (RuntimeError("HTTPS certificate verification failed"), "network/TLS"),
        (RuntimeError("offline connection refused"), "network/TLS"),
        (RuntimeError("incompatible tensor ABI"), "compatible versions"),
    ],
)
def test_clip_setup_hint_classifies_common_failures(error, expected) -> None:
    assert expected in clip_scorer._clip_setup_hint(error)


def test_score_with_clip_returns_expected_score_and_closes_image(tmp_path, monkeypatch) -> None:
    install_fake_torch(monkeypatch)
    image_path = tmp_path / "candidate.jpg"
    image_path.write_bytes(b"not read by fake")
    events = []

    class FakeImage:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            events.append(("enter", self.label))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("close", self.label))

        def convert(self, mode):
            assert mode == "RGB"
            return FakeImage("rgb")

    monkeypatch.setattr(clip_scorer.Image, "open", lambda path: FakeImage(str(path)))

    def processor(*, text, images, return_tensors, padding):
        assert len(text) == 6
        assert images.label == "rgb"
        assert return_tensors == "pt"
        assert padding is True
        return {"pixel_values": "pixels"}

    def model(**inputs):
        assert inputs == {"pixel_values": "pixels"}
        return types.SimpleNamespace(
            logits_per_image=FakeTensor([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])
        )

    result = clip_scorer.score_with_clip(image_path, model=model, processor=processor)

    assert result == {
        "subject_clarity": 12,
        "lighting": 9,
        "color_pop": 6,
        "emotion": 6,
        "scroll_stop": 7,
        "crop_4x5": 5,
        "total": 42,
        "one_line": "CLIP score: 0.100 (cinematic=0.30, action=0.20, striking=0.15)",
    }
    assert events == [
        ("enter", str(image_path)),
        ("enter", "rgb"),
        ("close", "rgb"),
        ("close", str(image_path)),
    ]


def test_score_with_clip_loads_model_when_either_dependency_is_missing(monkeypatch) -> None:
    install_fake_torch(monkeypatch)
    monkeypatch.setattr(
        clip_scorer, "load_clip_model", lambda: ("loaded-model", "loaded-processor")
    )

    class FakeImage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def convert(self, mode):
            return self

    monkeypatch.setattr(clip_scorer.Image, "open", lambda _path: FakeImage())

    def loaded_processor(**_kwargs):
        return {}

    def loaded_model(**_kwargs):
        return types.SimpleNamespace(logits_per_image=FakeTensor([0.2] * 4 + [0.1] * 2))

    monkeypatch.setattr(clip_scorer, "load_clip_model", lambda: (loaded_model, loaded_processor))
    assert clip_scorer.score_with_clip(Path("x.jpg"), model=object())["total"] == 42
    assert clip_scorer.score_with_clip(Path("x.jpg"), processor=object())["total"] == 42


@pytest.mark.parametrize("failure_stage", ["processor", "model"])
def test_score_with_clip_propagates_failures_and_closes_images(monkeypatch, failure_stage) -> None:
    install_fake_torch(monkeypatch)
    closed = []

    class FakeImage:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            closed.append(self.label)

        def convert(self, mode):
            return FakeImage("rgb")

    monkeypatch.setattr(clip_scorer.Image, "open", lambda _path: FakeImage("source"))

    def processor(**_kwargs):
        if failure_stage == "processor":
            raise ValueError("malformed image input")
        return {}

    def model(**_kwargs):
        if failure_stage == "model":
            raise RuntimeError("model inference failed")
        raise AssertionError("model should not run")

    expected = ValueError if failure_stage == "processor" else RuntimeError
    with pytest.raises(expected):
        clip_scorer.score_with_clip(Path("broken.jpg"), model=model, processor=processor)

    assert closed == ["rgb", "source"]
