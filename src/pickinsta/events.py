"""Small structured event API with a backwards-compatible console renderer."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class EventLevel(StrEnum):
    """Severity understood by structured event consumers."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


_SENSITIVE_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def _safe_fields(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    safe = {
        str(key): "[REDACTED]"
        if any(marker in str(key).lower() for marker in _SENSITIVE_MARKERS)
        else value
        for key, value in fields.items()
    }
    return MappingProxyType(safe)


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable machine-readable event with its already-formatted human message."""

    code: str
    message: str
    level: EventLevel = EventLevel.INFO
    stage: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _safe_fields(self.fields))


EventSink = Callable[[Event], None]


def render_event(event: Event) -> None:
    """Render an event exactly as legacy pipeline ``print`` calls did."""
    print(event.message)


_sink: ContextVar[EventSink] = ContextVar("pickinsta_event_sink", default=render_event)


def emit(
    code: str,
    message: str,
    *,
    level: EventLevel = EventLevel.INFO,
    stage: str | None = None,
    **fields: Any,
) -> Event:
    """Publish an event; a broken observer must never break image processing."""
    event = Event(code=code, message=message, level=level, stage=stage, fields=fields)
    sink = _sink.get()
    try:
        sink(event)
    except Exception:
        if sink is not render_event:
            render_event(event)
    return event


def console_event(stage: str, message: object) -> Event:
    """Turn a legacy one-argument console line into a structured stage event."""
    rendered = str(message)
    stripped = rendered.lstrip()
    if stripped.startswith("❌"):
        level, outcome = EventLevel.ERROR, "error"
    elif "⚠" in stripped:
        level, outcome = EventLevel.WARNING, "warning"
    else:
        level, outcome = EventLevel.INFO, "message"
    return emit(f"{stage}.console.{outcome}", rendered, level=level, stage=stage)


@contextmanager
def event_sink(sink: EventSink) -> Iterator[None]:
    """Install a task/thread-local sink for the duration of a context."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
