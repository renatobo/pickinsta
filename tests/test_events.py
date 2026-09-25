from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pickinsta.events import Event, EventLevel, console_event, emit, event_sink


def test_default_renderer_preserves_human_message(capsys) -> None:
    emit("test.message", "  exact human output", stage="test", count=2)
    assert capsys.readouterr().out == "  exact human output\n"


def test_context_local_sink_captures_structured_event_without_rendering(capsys) -> None:
    captured: list[Event] = []
    with event_sink(captured.append):
        result = emit(
            "resize.complete",
            "prepared",
            level=EventLevel.INFO,
            stage="resize",
            count=3,
        )

    assert captured == [result]
    assert result.code == "resize.complete"
    assert result.stage == "resize"
    assert result.fields == {"count": 3}
    assert capsys.readouterr().out == ""


def test_events_are_immutable_and_sensitive_fields_are_redacted() -> None:
    event = Event(
        code="credential",
        message="loaded",
        fields={"api_key": "private", "access_token": "private", "path": ".env"},
    )
    assert event.fields == {
        "api_key": "[REDACTED]",
        "access_token": "[REDACTED]",
        "path": ".env",
    }
    with pytest.raises(TypeError):
        event.fields["path"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        event.code = "changed"  # type: ignore[misc]


def test_broken_observer_falls_back_to_human_renderer(capsys) -> None:
    def broken(_event: Event) -> None:
        raise RuntimeError("observer failed")

    with event_sink(broken):
        emit("test.safe", "pipeline continues")

    assert capsys.readouterr().out == "pipeline continues\n"


def test_nested_sinks_restore_previous_context() -> None:
    outer: list[Event] = []
    inner: list[Event] = []
    with event_sink(outer.append):
        emit("outer.first", "one")
        with event_sink(inner.append):
            emit("inner", "two")
        emit("outer.last", "three")
    assert [event.code for event in outer] == ["outer.first", "outer.last"]
    assert [event.code for event in inner] == ["inner"]


@pytest.mark.parametrize(
    ("message", "code", "level"),
    [
        ("ready", "cropping.console.message", EventLevel.INFO),
        ("  ⚠ degraded", "cropping.console.warning", EventLevel.WARNING),
        ("❌ failed", "cropping.console.error", EventLevel.ERROR),
    ],
)
def test_console_event_adds_stage_severity_metadata(message, code, level) -> None:
    captured: list[Event] = []
    with event_sink(captured.append):
        console_event("cropping", message)
    assert captured[0].code == code
    assert captured[0].level is level
    assert captured[0].stage == "cropping"
    assert captured[0].message == message
