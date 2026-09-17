"""The decision replay preserves actual calls, timing and raw results."""

import copy
import gzip
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.cost_tracker import CostTracker
from agents.replay_recorder import DELTA_TEXT, ReplayRecorder
from agents.replay_taxonomy import agent_ids, resolve_call
from generators.replay_generator import ReplayGenerator


class JevReplayTests(unittest.TestCase):
    def capture(self):
        self.now = 1_000_000.0
        recorder = ReplayRecorder(clock=lambda: self.now)
        recorder.begin_run("2026-09-18")
        request = {"state": {"article_0000": {"title": "Model announcement"}}, "questions": {}}
        raw = {"answers": {"r_0000": {"choice": "relevant", "confidence": 0.7}}}
        call_id = recorder.start_call(None, {
            "caller": "jev.filter.batch_0", "provider_id": "typesafe",
            "provider_model": "jev-1.13.0", "interaction_type": "decision",
            "decision_item_count": 1, "decision_question_count": 2,
            "messages_text": json.dumps(request),
        })
        recorder.mark_started(call_id)
        self.now += 1.641
        recorder.record_delta(call_id, DELTA_TEXT, json.dumps({
            "schema_version": "jev-relevance-replay/v1", "articles": [], "raw_response": raw,
        }))
        recorder.finish_call(call_id, response=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=432, output_tokens=41), stop_reason="complete",
        ), context_update={"estimated_cost_usd": .0002, "usage_measured": True,
                           "decision_items_kept": 1, "authorization": "must not propagate"})
        return recorder.snapshot(), raw, request

    def build_calls(self, recorder):
        return ReplayGenerator("/tmp/unused-jev-replay")._build_calls(
            {}, recorder, CostTracker(), 1_000_000.0,
            [{"id": "phase-2", "start_ms": 0, "end_ms": 3000}],
        )

    def test_single_response_keeps_exact_timing_and_native_usage(self):
        recorder, raw, request = self.capture()
        calls = self.build_calls(recorder)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual((call["start_ms"], call["end_ms"]), (0, 1641))
        self.assertEqual(call["agent_id"], "jev")
        self.assertEqual(call["decision_question_count"], 2)
        self.assertEqual((call["input_tokens"], call["output_tokens"]), (432, 41))
        self.assertIsNone(call["first_token_ms"])
        self.assertIsNone(call["effort"])
        self.assertFalse(call["billed"])
        self.assertEqual(call["cost_usd_estimated"], .0002)
        span = recorder["calls"][0]
        self.assertNotIn("authorization", span["context"])
        self.assertEqual(span["deltas"]["t"], [1641])
        self.assertEqual(json.loads(span["deltas"]["text"][0])["raw_response"], raw)
        self.assertEqual(json.loads(span["prompt"]["messages"]), request)

    def test_decisions_survive_stream_pruning_without_partial_json(self):
        recorder, raw, _ = self.capture()
        calls = self.build_calls(recorder)
        # Force every compression rung, including the final protected-call set.
        gen = ReplayGenerator("/tmp/unused-jev-replay", max_stream_bytes=1)
        blob, _ = gen._build_stream(recorder, calls, "2026-09-18")
        delta = json.loads(gzip.decompress(blob))["calls"][calls[0]["id"]]
        self.assertEqual(json.loads("".join(delta["text"]))["raw_response"], raw)
        self.assertEqual(delta["t"], [1641])
        self.assertTrue(calls[0]["has_stream"])

    def test_failed_batch_does_not_inherit_claude_price_or_other_batch(self):
        recorder, _, _ = self.capture()
        successful = recorder["calls"][0]
        failed = {**successful, "id": "failed", "outcome": "failed", "end_ms": 1,
                  "input_tokens": None, "output_tokens": None, "deltas": {},
                  "context": {**successful["context"], "estimated_cost_usd": None}}
        recorder["calls"] = [failed, successful]
        calls = self.build_calls(recorder)
        failure = next(call for call in calls if call["id"] == "failed")
        self.assertEqual(failure["recovered_by"], successful["id"])
        self.assertNotIn("cost_usd_estimated", failure)
        self.assertNotIn("input_tokens_estimated", failure)

    def test_restored_decision_preserves_response_and_time_across_id_collision(self):
        recorder, raw, _ = self.capture()
        original = copy.deepcopy(recorder)
        restored = {"t0_epoch": 1_000_000.0, "spans": recorder["calls"], "cost_calls": []}
        # A later live process starts its own call counter at c001.
        live = copy.deepcopy(recorder)
        live["calls"][0]["caller"] = "jev.filter.batch_1"
        phases = [{"id": "phase-2", "start_ms": 5000, "end_ms": 8000}]
        gen = ReplayGenerator("/tmp/unused-jev-restore-review")
        costs, merged, count = gen._merge_restored_calls(
            restored, {}, live, [{"start_time": 1_000_000.0, "duration": 3.0}],
            [True], phases, 1_000_000.0,
        )
        calls = gen._build_calls(costs, merged, CostTracker(), 1_000_000.0, phases)
        self.assertEqual(len({call["id"] for call in calls}), 2)
        self.assertEqual(count, 1)
        recovered = next(call for call in calls if call["id"] != "c001")
        self.assertEqual(recovered["interaction_type"], "decision")
        self.assertEqual((recovered["start_ms"], recovered["end_ms"]), (5000, 6641))
        self.assertEqual(recovered["input_tokens"], 432)
        blob, _ = gen._build_stream(merged, calls, "2026-09-18")
        outputs = json.loads(gzip.decompress(blob))["calls"]
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[recovered["id"]]["t"], [6641])
        self.assertEqual(json.loads(outputs[recovered["id"]]["text"][0])["raw_response"], raw)
        self.assertEqual(recorder, original)

    def test_cumulative_checkpoint_preserves_decision_receipt_epoch(self):
        from agents.orchestrator import MainOrchestrator
        recorder, raw, _ = self.capture()
        original = copy.deepcopy(recorder)
        bundle = {"t0_epoch": 1_000_000.0, "spans": recorder["calls"], "cost_calls": []}
        expected_receipt_epoch = 1_000_001.641
        for new_origin in (1_000_020.0, 1_000_040.0):
            orchestrator = MainOrchestrator.__new__(MainOrchestrator)
            orchestrator._restored_replay = bundle
            live = SimpleNamespace(snapshot=lambda: {"t0_epoch": new_origin, "calls": []})
            tracker = SimpleNamespace(get_json_report=lambda: {"calls": []})
            with patch("agents.orchestrator.get_recorder", return_value=live), \
                    patch("agents.orchestrator.get_tracker", return_value=tracker):
                bundle = orchestrator._export_replay_bundle()
            span = bundle["spans"][0]
            self.assertAlmostEqual(bundle["t0_epoch"] + span["end_ms"] / 1000, expected_receipt_epoch)
            self.assertAlmostEqual(bundle["t0_epoch"] + span["deltas"]["t"][0] / 1000, expected_receipt_epoch)
            self.assertEqual(json.loads(span["deltas"]["text"][0])["raw_response"], raw)
            self.assertEqual(span["context"]["interaction_type"], "decision")
        self.assertEqual(recorder, original)

    def test_raw_response_secret_blocks_output_without_logging_secret(self):
        secret = "apikey_" + "a" * 32 + "_" + "b" * 64
        gen = ReplayGenerator("/tmp/unused-jev-replay")
        index = {"run": {"stream_available": True}, "calls": [{"has_stream": True}]}
        blob = gzip.compress(json.dumps({"raw_response": secret}).encode())
        with self.assertLogs("generators.replay_generator", level="ERROR") as logged:
            updated, output, _ = gen._gate_artifacts(index, blob, None)
        self.assertIsNone(output)
        self.assertFalse(updated["run"]["stream_available"])
        self.assertNotIn(secret, "".join(logged.output))

    def test_taxonomy_keeps_jev_distinct_from_news_writer(self):
        self.assertIn("jev", agent_ids())
        self.assertEqual(resolve_call("jev.filter.batch_1").agent_id, "jev")
        self.assertEqual(resolve_call("news_analyzer.batch_1").agent_id, "news_analyzer")


if __name__ == "__main__":
    unittest.main()
