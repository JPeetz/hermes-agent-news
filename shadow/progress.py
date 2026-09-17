"""Secret-safe progress events for bounded shadow model calls.

Progress is deliberately a tiny stderr-only observer.  It never receives a
prompt, response, header, exception, or arbitrary metadata dictionary.  The
background heartbeat makes a blocked synchronous HTTP call observable while
leaving stdout available for a machine-readable command result.
"""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import time
from typing import Any


DEFAULT_HEARTBEAT_SECONDS = 30.0
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,159}$")
# Provider/model identifiers commonly contain ``@``, ``/`` and ``+``. Keep
# those safe characters visible while redacting controls, whitespace, and
# other arbitrary text instead of preventing the inference itself.
_SAFE_MODEL = re.compile(r"^[A-Za-z0-9@/+_.:-]{1,160}$")
REDACTED_MODEL = "redacted"
_EVENTS = frozenset({"start", "heartbeat", "done"})
_STATUSES = frozenset({
    "started",
    "heartbeat",
    "complete",
    "success",
    "invalid_response",
    "transport_error",
    "error",
    "failed",
    "incomplete",
    "budget_exhausted",
})
_ALLOWED_FIELDS = frozenset({
    "event",
    "role",
    "stage",
    "request_index",
    "attempt",
    "item_count",
    "model",
    "elapsed_seconds",
    "status",
    "input_tokens",
    "output_tokens",
})
_OUTPUT_LOCK = threading.Lock()


def _safe_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_TEXT.fullmatch(value):
        raise ValueError(f"{field} must be a bounded safe identifier")
    return value


def _nonnegative_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field)


def _safe_model(value: Any) -> str:
    if isinstance(value, str) and _SAFE_MODEL.fullmatch(value):
        return value
    return REDACTED_MODEL


class Progress:
    """Emit bounded start/heartbeat/completion events to stderr.

    The class is usable directly with ``start``/``finish`` or as either a
    synchronous or asynchronous context manager.  ``finish`` is idempotent;
    callers can safely use it from both an exception path and a ``finally``
    block.
    """

    def __init__(
        self,
        *,
        role: str,
        stage: str,
        request_index: int | None = None,
        attempt: int | None = None,
        item_count: int | None = None,
        model: str,
        interval_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        stream: Any | None = None,
    ) -> None:
        self.role = _safe_text(role, "role")
        self.stage = _safe_text(stage, "stage")
        self.request_index = _optional_nonnegative_int(request_index, "request_index")
        if attempt is not None:
            self.attempt = _nonnegative_int(attempt, "attempt", minimum=1)
        else:
            self.attempt = None
        self.item_count = _optional_nonnegative_int(item_count, "item_count")
        self.model = _safe_model(model)
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        self.interval_seconds = float(interval_seconds)
        self.stream = stream if stream is not None else sys.stderr
        if not callable(getattr(self.stream, "write", None)) or not callable(
            getattr(self.stream, "flush", None)
        ):
            raise ValueError("stream must provide write and flush")
        self._started_at: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_started = False
        self._state_lock = threading.Lock()
        self._finished = False

    @property
    def started(self) -> bool:
        with self._state_lock:
            return self._started_at is not None

    @property
    def finished(self) -> bool:
        with self._state_lock:
            return self._finished

    def _event(self, event: str, *, status: str, **metrics: int | float | None) -> None:
        if event not in _EVENTS:
            raise ValueError("unsupported progress event")
        if status not in _STATUSES:
            raise ValueError("unsupported progress status")
        with self._state_lock:
            started_at = self._started_at
        elapsed = 0.0 if started_at is None else max(0.0, time.monotonic() - started_at)
        payload: dict[str, Any] = {
            "event": event,
            "role": self.role,
            "stage": self.stage,
            "model": self.model,
            "elapsed_seconds": round(elapsed, 3),
            "status": status,
        }
        for field, value in (
            ("request_index", self.request_index),
            ("attempt", self.attempt),
            ("item_count", self.item_count),
        ):
            if value is not None:
                payload[field] = value
        for field, value in metrics.items():
            if field not in {"input_tokens", "output_tokens"}:
                raise ValueError("unsupported progress metric")
            if value is not None:
                payload[field] = _nonnegative_int(value, field)
        if set(payload) - _ALLOWED_FIELDS:
            raise ValueError("progress event contains an unsupported field")
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        # Logging must never change the outcome of a model call.  In
        # particular, a closed stderr pipe cannot strand a heartbeat thread or
        # replace a real provider failure with an I/O error.
        try:
            with _OUTPUT_LOCK:
                self.stream.write(line + "\n")
                self.stream.flush()
        except Exception:
            return

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            with self._state_lock:
                if self._finished or self._started_at is None:
                    return
            self._event("heartbeat", status="heartbeat")

    def start(self) -> "Progress":
        with self._state_lock:
            if self._started_at is not None:
                return self
            self._started_at = time.monotonic()
            self._finished = False
        self._event("start", status="started")
        try:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="shadow-progress-heartbeat",
                daemon=True,
            )
            self._thread.start()
            self._thread_started = True
        except Exception:
            # Progress is an observer. A thread/resource failure must not
            # abort a model request after its budget reservation was made.
            self._thread = None
            self._thread_started = False
            self._stop.set()
        return self

    def finish(
        self,
        *,
        status: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> "Progress":
        with self._state_lock:
            if self._started_at is None or self._finished:
                return self
        try:
            if (
                not isinstance(status, str)
                or status not in _STATUSES
                or status in {"started", "heartbeat"}
            ):
                raise ValueError("finish status must be terminal")
            _optional_nonnegative_int(input_tokens, "input_tokens")
            _optional_nonnegative_int(output_tokens, "output_tokens")
        except ValueError:
            # Invalid terminal metadata must not strand a heartbeat thread.
            # Stop it and emit only a safe generic error event before
            # re-raising the programming error to the caller.
            self._stop_and_emit_error()
            raise
        with self._state_lock:
            self._finished = True
        self._stop.set()
        thread = self._thread
        if self._thread_started and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval_seconds + 0.1))
        self._event(
            "done",
            status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return self

    def _stop_and_emit_error(self) -> None:
        with self._state_lock:
            if self._started_at is None or self._finished:
                return
            self._finished = True
        self._stop.set()
        thread = self._thread
        if self._thread_started and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval_seconds + 0.1))
        self._event("done", status="error")

    def __enter__(self) -> "Progress":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.finish(status="error" if exc_type is not None else "complete")
        return False

    async def __aenter__(self) -> "Progress":
        return self.start()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.finish(status="error" if exc_type is not None else "complete")
        return False


__all__ = ["DEFAULT_HEARTBEAT_SECONDS", "Progress", "REDACTED_MODEL"]
