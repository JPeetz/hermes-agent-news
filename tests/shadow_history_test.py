"""Offline historical evidence contracts. Run explicitly by the repository owner."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from shadow.contracts import verify_bundle
from shadow.github import GitHubSource, lineage_from_logs
from shadow.history import (HistoricalEvidenceError, import_legacy_bundle, map_selected_ids,
                            read_diagnostics_archive, recover_filter, reconstruct_text, source_health, _coverage)
from shadow.rendering import filter_system_prompt, build_fenced_user_message, render_filter_input


ITEMS = [
    {"id": "01234567abcd", "title": "OpenAI release", "content": "New model", "source": "Example"},
    {"id": "abcdef012345", "title": "AI policy", "content": "Safety regulation", "source": "Example"},
]


def span(call_id="c001", *, start=0, outcome="ok", selected=None):
    nonce = "1234567890abcdef"
    system = filter_system_prompt(ITEMS[0]["id"], nonce)
    messages = "[USER]\n" + build_fenced_user_message(render_filter_input(ITEMS), nonce)
    text = json.dumps({"ai_article_ids": [ITEMS[0]["id"]] if selected is None else selected})
    return {"id": call_id, "request_id": start + 1, "caller": "news_analyzer.filter",
            "provider_model": "observed-model", "provider_id": "provider", "start_ms": start,
            "end_ms": start + 10, "outcome": outcome, "stop_reason": "end_turn" if outcome == "ok" else None,
            "truncated": False, "dropped_deltas": 0, "text_chars": len(text),
            "prompt_truncated": False, "prompt_chars": len(system) + len(messages),
            "prompt": {"system": system, "messages": messages},
            "deltas": {"t": [start, start + 1], "kind": [0, 1], "text": ["Ignore this reasoning", text]},
            "input_tokens": 25, "output_tokens": 15, "context": {"attempt": 1}}


def archive(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name, value in entries:
            zipped.writestr(name, json.dumps(value) if not isinstance(value, bytes) else value)
    return buffer.getvalue()


class HistoricalRecoveryTest(unittest.TestCase):
    def test_last_successful_retry_and_failed_usage_are_distinct(self):
        failed, success = span(outcome="failed"), span("c026", start=500)
        failed["deltas"]["text"][1] = '{"ai_article_ids": ['
        inputs, decision = recover_filter(ITEMS, [success, failed])
        self.assertEqual(decision["final_call_id"], "c026")
        self.assertEqual(decision["selected_ids"], [ITEMS[0]["id"]])
        self.assertEqual(len(decision["attempts"]), 2)
        self.assertTrue(decision["attempts"][0]["usage_partial"])
        self.assertEqual(decision["attempts"][0]["output_tokens"], 15)
        self.assertEqual(len(inputs["records"]), 2)

    def test_legacy_prose_around_final_json_is_supported_without_reasoning_ids(self):
        call = span()
        call["deltas"]["text"][0] = '{"ai_article_ids": ["fabricated-in-reasoning"]}'
        final = "Relevant articles follow.\n```json\n" + call["deltas"]["text"][1] + "\n```\nDone."
        call["deltas"]["text"][1] = final
        call["text_chars"] = len(final)
        _, decision = recover_filter(ITEMS, [call])
        self.assertEqual(decision["selected_ids"], [ITEMS[0]["id"]])
        call["deltas"]["text"][1] += '\n{"ai_article_ids": []}'
        call["text_chars"] = len(call["deltas"]["text"][1])
        with self.assertRaises(HistoricalEvidenceError):
            recover_filter(ITEMS, [call])

    def test_failed_final_attempt_cannot_reuse_earlier_success(self):
        with self.assertRaises(HistoricalEvidenceError):
            recover_filter(ITEMS, [span(), span("c026", start=500, outcome="failed")])

    def test_shortened_id_requires_unique_incumbent_mapping(self):
        self.assertEqual(map_selected_ids(["01234567"], [i["id"] for i in ITEMS]), [ITEMS[0]["id"]])
        for raw, ids in [(["unknown"], ["01234567abcd"]),
                         (["01234567"], ["01234567abcd", "01234567efgh"]),
                         (["01234567", "01234567"], ["01234567abcd"]),
                         (["01234567", "01234567abcd"], ["01234567abcd"])]:
            with self.subTest(raw=raw), self.assertRaises(HistoricalEvidenceError):
                map_selected_ids(raw, ids)

    def test_text_only_complete_stream_is_authoritative(self):
        expected = span()
        self.assertEqual(json.loads(reconstruct_text(expected))["ai_article_ids"], [ITEMS[0]["id"]])
        for change in ({"truncated": True}, {"dropped_deltas": 1}, {"text_chars": 999}, {"stop_reason": "max_tokens"}):
            with self.subTest(change=change), self.assertRaises(HistoricalEvidenceError):
                reconstruct_text({**expected, **change})

    def test_survivor_only_gathering_cannot_reconstruct_input(self):
        with self.assertRaises(HistoricalEvidenceError):
            recover_filter(ITEMS[:1], [span()])

    def test_capture_and_rubric_tampering_fail_parity(self):
        for key in ("system", "messages"):
            changed = span()
            changed["prompt"][key] += "tampered"
            changed["prompt_chars"] += len("tampered")
            with self.subTest(key=key), self.assertRaises(HistoricalEvidenceError):
                recover_filter(ITEMS, [changed])
        with self.assertRaises(HistoricalEvidenceError):
            recover_filter(list(reversed(ITEMS)), [span()])

    def test_empty_keyword_input_is_a_valid_no_model_decision(self):
        inputs, decision = recover_filter([{"id": "one", "title": "Ocean", "content": "Waves", "source": "Nature"}], [])
        self.assertEqual(inputs["ordered_ids"], [])
        self.assertEqual(decision["decisions"], [])
        self.assertIsNone(decision["model"])

    def test_dst_coverage_uses_historical_local_date(self):
        spring = _coverage("2026-03-09", {})
        self.assertTrue(spring["start"].endswith("-05:00"))
        self.assertTrue(spring["end"].endswith("-04:00"))
        autumn = _coverage("2026-11-02", {})
        self.assertTrue(autumn["start"].endswith("-04:00"))
        self.assertTrue(autumn["end"].endswith("-05:00"))

    def test_nested_partial_reddit_cannot_be_hidden_by_old_success_status(self):
        gathering = {"categories": {"reddit": [{"id": "present"}]},
                     "collection_status": {cat: {"status": "success"} for cat in ("news", "research", "social", "reddit")}}
        gathering["collection_status"]["reddit"]["steps"] = [{"name": "r/example", "status": "partial"}]
        health = source_health(gathering, {})
        self.assertFalse(health["healthy"])
        self.assertFalse(health["reddit_empty"])
        self.assertEqual(len(health["degraded_steps"]), 1)

    def test_filter_capability_survives_degraded_or_unpublished_cohort(self):
        gathering = {"categories": {"news": ITEMS, "reddit": [{"id": "reddit"}]},
                     "collection_status": {cat: {"status": "partial" if cat == "research" else "success"}
                                           for cat in ("news", "research", "social", "reddit")}}
        payload = archive([("checkpoints/2026-09-17/gathering.json", gathering),
                           ("checkpoints/2026-09-17/analysis.json", {"_replay": {"spans": [span()], "cost_calls": []}})])
        metadata = {"run_id": 35192960377, "run_attempt": 1, "execution_sha": "a" * 40, "conclusion": "success"}
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "bundle"
            result = import_legacy_bundle(payload, metadata, target)
            self.assertTrue(result["capabilities"]["filter_replay"])
            self.assertFalse(result["capabilities"]["pipeline_replay"])
            self.assertFalse(result["eligibility"]["healthy"])
            self.assertFalse(result["eligibility"]["published"])
            verify_bundle(target)
            with self.assertRaises(ValueError):
                verify_bundle(target, require_healthy=True)
            with self.assertRaises(HistoricalEvidenceError):
                import_legacy_bundle(payload, metadata, target)


class ArchiveAndLineageTest(unittest.TestCase):
    def test_archive_paths_and_duplicate_aliases_rejected(self):
        for entries in [([("../evil.json", {})]),
                        ([("/absolute", {})]),
                        ([("checkpoints/2026-09-17/gathering.json", {}),
                          ("data/checkpoints/2026-09-17/gathering.json", {})])]:
            with self.subTest(entries=entries), self.assertRaises(HistoricalEvidenceError):
                read_diagnostics_archive(archive(entries))

    def test_archive_allowlist_and_size_bound(self):
        payload = archive([("config/providers.yaml", b"secret"),
                           ("checkpoints/2026-09-17/gathering.json", {})])
        self.assertNotIn("config/providers.yaml", read_diagnostics_archive(payload))
        with patch("shadow.history.MAX_MEMBER_BYTES", 1), self.assertRaises(HistoricalEvidenceError):
            read_diagnostics_archive(payload)

    def test_logs_cannot_take_checkout_or_push_from_model_output(self):
        sha = "1" * 40
        logs = (f"job\tRun pipeline\t2026-09-17T07:00:00Z git log -1 --format=%H\n"
                f"job\tRun pipeline\t2026-09-17T07:00:01Z {sha}\n"
                f"job\tCheck out flyryan main\t2026-09-17T07:00:00Z [command]/usr/bin/git log -1 --format=%H\n"
                f"job\tCheck out flyryan main\t2026-09-17T07:00:01Z '{sha}'\n"
                "job\tPush generated commit\t2026-09-17T08:00:00Z 1111111..abcdef0 HEAD -> main\n")
        lineage = lineage_from_logs(logs)
        self.assertEqual(lineage["execution_sha"], sha)
        self.assertEqual(lineage["output_commit"], "abcdef0")

    def test_artifacts_do_not_cross_attempt_boundaries(self):
        class FakeSource(GitHubSource):
            def run(self, run_id, attempt=None):
                return {"id": run_id, "run_attempt": 2 if attempt is None else attempt,
                        "run_started_at": "2026-09-17T08:00:00Z", "updated_at": "2026-09-17T09:00:00Z"}
            def artifacts(self, run_id):
                return [{"id": 1, "created_at": "2026-09-17T07:30:00Z"},
                        {"id": 2, "created_at": "2026-09-17T08:30:00Z"}]
        first_attempt = {"id": 123, "run_attempt": 1, "run_started_at": "2026-09-17T07:00:00Z", "updated_at": "2026-09-17T07:45:00Z"}
        self.assertEqual([a["id"] for a in FakeSource().attempt_artifacts(first_attempt)], [1])


if __name__ == "__main__":
    unittest.main()
