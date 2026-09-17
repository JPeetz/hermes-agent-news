"""Tests for secret-safe, stderr-only shadow progress events."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shadow.progress import Progress  # noqa: E402


ALLOWED_FIELDS = {
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
}


def _events(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _wait_for_heartbeat(stream: io.StringIO, *, timeout: float = 0.3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(event.get("event") == "heartbeat" for event in _events(stream)):
            return
        time.sleep(0.005)
    raise AssertionError("heartbeat was not emitted before timeout")


class ProgressTests(unittest.TestCase):
    def test_start_heartbeat_and_done_are_flushed_to_stderr_only(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            progress = Progress(
                role="judge",
                stage="input_adjudication",
                request_index=2,
                attempt=1,
                item_count=16,
                model="deepseek-v4.1-flash",
                interval_seconds=0.01,
                stream=stderr,
            ).start()
            _wait_for_heartbeat(stderr)
            progress.finish(status="success", input_tokens=281, output_tokens=12730)

        events = _events(stderr)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(events[0]["event"], "start")
        self.assertTrue(any(event["event"] == "heartbeat" for event in events))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(events[-1]["input_tokens"], 281)
        self.assertEqual(events[-1]["output_tokens"], 12730)
        for event in events:
            self.assertLessEqual(set(event), ALLOWED_FIELDS)
            self.assertEqual(event["model"], "deepseek-v4.1-flash")
            self.assertIsInstance(event["elapsed_seconds"], float)
            self.assertNotIn("prompt", json.dumps(event))
            self.assertNotIn("Bearer", json.dumps(event))

    def test_finish_is_idempotent_and_stops_the_heartbeat_on_error(self):
        stderr = io.StringIO()
        progress = Progress(
            role="judge",
            stage="input_adjudication",
            request_index=0,
            attempt=1,
            item_count=1,
            model="deepseek-v4.1-flash",
            interval_seconds=0.01,
            stream=stderr,
        ).start()
        _wait_for_heartbeat(stderr)
        progress.finish(status="error")
        count_after_finish = len(_events(stderr))
        progress.finish(status="success", input_tokens=1, output_tokens=1)
        time.sleep(0.04)
        events = _events(stderr)
        self.assertEqual(len(events), count_after_finish)
        self.assertEqual(sum(event["event"] == "done" for event in events), 1)
        self.assertEqual(events[-1]["status"], "error")

    def test_context_manager_logs_error_without_exception_text(self):
        stderr = io.StringIO()
        stdout = io.StringIO()
        secret = "synthetic-progress-secret"
        with redirect_stdout(stdout):
            with self.assertRaises(RuntimeError):
                with Progress(
                    role="judge",
                    stage="input_adjudication",
                    request_index=0,
                    attempt=1,
                    item_count=1,
                    model="deepseek-v4.1-flash",
                    interval_seconds=0.01,
                    stream=stderr,
                ):
                    raise RuntimeError(f"Bearer {secret}")
        encoded = stderr.getvalue()
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn('"status":"error"', encoded)
        self.assertNotIn(secret, encoded)
        self.assertNotIn("Bearer", encoded)

    def test_async_context_manager_uses_the_same_contract(self):
        async def exercise():
            stderr = io.StringIO()
            async with Progress(
                role="judge",
                stage="output_comparison",
                request_index=0,
                attempt=1,
                item_count=2,
                model="deepseek-v4.1-flash",
                interval_seconds=0.01,
                stream=stderr,
            ):
                _wait_for_heartbeat(stderr)
                await asyncio.sleep(0)
            return _events(stderr)

        events = asyncio.run(exercise())
        self.assertEqual(events[0]["event"], "start")
        self.assertTrue(any(event["event"] == "heartbeat" for event in events))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["status"], "complete")

    def test_metadata_is_validated_before_any_event(self):
        with self.assertRaises(ValueError):
            Progress(
                role="judge",
                stage="input\nforged",
                request_index=0,
                attempt=1,
                item_count=1,
                model="deepseek-v4.1-flash",
            )
        with self.assertRaises(ValueError):
            progress = Progress(
                role="judge",
                stage="input",
                request_index=0,
                attempt=1,
                item_count=1,
                model="deepseek-v4.1-flash",
                stream=io.StringIO(),
            )
            progress.start().finish(status="success", input_tokens=True)
        self.assertTrue(progress.finished)

    def test_heartbeat_thread_start_failure_does_not_abort_or_strand_request(self):
        stderr = io.StringIO()
        with patch(
            "shadow.progress.threading.Thread.start",
            side_effect=RuntimeError("synthetic thread failure"),
        ):
            progress = Progress(
                role="judge",
                stage="input",
                request_index=0,
                attempt=1,
                item_count=1,
                model="deepseek-v4.1-flash",
                stream=stderr,
            ).start()
        progress.finish(status="complete")
        events = _events(stderr)
        self.assertEqual([event["event"] for event in events], ["start", "done"])
        self.assertTrue(progress.finished)

    def test_provider_model_characters_are_allowed_and_unsafe_model_is_redacted(self):
        safe_stream = io.StringIO()
        safe = Progress(
            role="judge",
            stage="input",
            request_index=0,
            attempt=1,
            item_count=1,
            model="provider/@deepseek+v1",
            stream=safe_stream,
        ).start()
        safe.finish(status="complete")
        self.assertEqual(_events(safe_stream)[0]["model"], "provider/@deepseek+v1")

        unsafe_stream = io.StringIO()
        secret = "Bearer synthetic-progress-secret"
        unsafe = Progress(
            role="judge",
            stage="input",
            request_index=0,
            attempt=1,
            item_count=1,
            model=secret + "\n",
            stream=unsafe_stream,
        ).start()
        unsafe.finish(status="complete")
        encoded = unsafe_stream.getvalue()
        self.assertNotIn(secret, encoded)
        self.assertTrue(all(event["model"] == "redacted" for event in _events(unsafe_stream)))


if __name__ == "__main__":
    unittest.main()
