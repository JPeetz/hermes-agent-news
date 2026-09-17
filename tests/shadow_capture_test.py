"""Offline contracts for the bounded shadow capture/replay layer.

These tests intentionally use fake responses and local files only.  The
repository owner runs the focused suite; this file is supplied as the offline
acceptance fixture and is not executed by the implementation worker.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from shadow.capture import CaptureSession
from shadow.contracts import seal_bundle, sha256_json
from shadow.evidence import EvidenceStore, ReplayIntegrityError
from shadow.replay_context import ReplayContext


class FakeResponse:
    def __init__(self, body: bytes = b"<html><time>2026-09-16</time></html>", status_code: int = 200, url: str = "https://news.example/article"):
        self.content = body
        self.status_code = status_code
        self.url = url
        self.reason = "OK"
        self.encoding = "utf-8"
        self.headers = {"Content-Type": "text/html; charset=utf-8", "ETag": "abc"}


def _load_finalizer():
    path = Path(__file__).parents[1] / "scripts" / "shadow" / "finalize_bundle.py"
    spec = importlib.util.spec_from_file_location("shadow_finalize_bundle_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ShadowCaptureTest(unittest.TestCase):
    def test_capture_is_opt_in_and_secret_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            session = CaptureSession(Path(temp) / "bundle", "2026-09-17", enabled=True)
            self.assertTrue(session.capture_config({
                "llm": {"route_id": "main", "model": "incumbent", "api_key": "do-not-write"},
                "providers": {"api_key": "do-not-write"},
                "timeout_seconds": 30,
            }))
            config = json.loads((Path(temp) / "bundle/context/effective-config.json").read_text())
            encoded = json.dumps(config)
            self.assertNotIn("do-not-write", encoded)
            self.assertIn("timeout_seconds", encoded)

            disabled = CaptureSession(Path(temp) / "disabled", enabled=False)
            self.assertFalse(disabled.capture_config({"model": "x"}))
            self.assertFalse((Path(temp) / "disabled").exists())

    def test_history_is_date_bounded_and_records_search_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "web" / "data"
            for day in ("2026-09-16", "2026-08-03", "2026-07-31", "2026-09-17"):
                (source / day).mkdir(parents=True)
                for category in CaptureSession.CATEGORIES:
                    (source / day / f"{category}.json").write_text("[]")
                (source / day / "search-documents.json").write_text("[]")
            (source / "search-documents.json").write_text("[]")
            session = CaptureSession(Path(temp) / "bundle", "2026-09-17", enabled=True)
            self.assertTrue(session.capture_history(source, search_documents_used=True))
            files = [p.relative_to(Path(temp) / "bundle").as_posix() for p in (Path(temp) / "bundle").rglob("*") if p.is_file()]
            self.assertIn("history/2026-09-16/news.json", files)
            self.assertIn("history/search-documents.json", files)
            self.assertNotIn("history/2026-09-17/news.json", files)
            self.assertNotIn("history/2026-07-31/news.json", files)

    def test_evidence_round_trip_and_failed_requests_are_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bundle"
            store = EvidenceStore(root, mode="capture")
            response = FakeResponse()
            wrapped = store.wrap_safe_get(lambda url, timeout=12, max_redirects=5: response)
            observed = wrapped("https://news.example/article", timeout=7, max_redirects=3)
            self.assertEqual(observed.content, response.content)
            store.record_failure("https://news.example/missing", RuntimeError("timeout token=secret"))
            replay = EvidenceStore(root, mode="replay")
            frozen = replay.safe_get("https://news.example/article", timeout=1, max_redirects=0)
            self.assertEqual(frozen.text, response.content.decode())
            with self.assertRaises(ReplayIntegrityError):
                replay.safe_get("https://news.example/missing")
            with self.assertRaises(ReplayIntegrityError):
                replay.safe_get("https://news.example/not-captured")

    def test_replay_context_refuses_future_history_and_network_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bundle"
            root.mkdir()
            records = [{"id": "a", "title": "AI", "source": "s", "snippet": "AI..."}]
            digest = sha256_json(records)
            (root / "relevance").mkdir()
            (root / "relevance/input.json").write_text(json.dumps({
                "schema_version": "news-relevance-input/v1", "records": records,
                "ordered_ids": ["a"], "input_sha256": digest,
                "system_prompt": "system", "user_message": "user",
            }))
            (root / "relevance/incumbent-decision.json").write_text(json.dumps({
                "input_sha256": digest,
                "decisions": [{"id": "a", "decision": "keep", "effective_keep": True}],
            }))
            (root / "input-manifest.json").write_text("{}")
            manifest = seal_bundle(root, {
                "source": {}, "report_date": "2026-09-17", "coverage": {},
                "capabilities": {"filter_replay": True, "pipeline_replay": False},
                "missing_dependencies": [], "versions": {},
                "eligibility": {"healthy": True}, "publication": {"status": "unverified"},
            })
            context = ReplayContext(root)
            self.assertEqual(context.read_filter_input()["ordered_ids"], ["a"])
            with self.assertRaises(ReplayIntegrityError):
                context.read_history("2026-09-17", category="news")
            with self.assertRaises(ReplayIntegrityError):
                context.safe_get("https://never-fetch.example/article")

    def test_finalize_binds_postpush_commit_and_output_hashes(self):
        finalizer = _load_finalizer()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "bundle"
            (root / "original").mkdir(parents=True)
            (root / "relevance").mkdir()
            records = [{"id": "a", "title": "AI", "source": "s", "snippet": "AI..."}]
            digest = sha256_json(records)
            (root / "relevance/input.json").write_text(json.dumps({
                "records": records, "ordered_ids": ["a"], "input_sha256": digest,
                "system_prompt": "system", "user_message": "user",
            }))
            (root / "relevance/incumbent-decision.json").write_text(json.dumps({
                "input_sha256": digest,
                "decisions": [{"id": "a", "decision": "keep", "effective_keep": True}],
            }))
            (root / "input-manifest.json").write_text(json.dumps({"report_date": "2026-09-17", "coverage": {}}))
            for name in finalizer.OUTPUT_NAMES:
                value = {"date": "2026-09-17"}
                if name != "summary.json":
                    value["category"] = name[:-5]
                (root / "original" / name).write_text(json.dumps(value))
            sha = "a" * 40
            output_hashes = {
                name: hashlib.sha256((root / "original" / name).read_bytes()).hexdigest()
                for name in finalizer.OUTPUT_NAMES
            }
            manifest = finalizer.finalize_bundle(root, run_id=1, run_attempt=1,
                                                 execution_sha=sha, output_commit=sha,
                                                 report_date="2026-09-17", healthy=True,
                                                 expected_output_hashes=output_hashes)
            self.assertEqual(manifest["publication"]["status"], "published")
            self.assertEqual(manifest["source"]["output_commit"], sha)


if __name__ == "__main__":
    unittest.main()
