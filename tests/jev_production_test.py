"""Mock-only production Jev selection, transport observation, and replay contracts."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from agents.analyzers.news_analyzer import NewsAnalyzer
from agents.jev_relevance import JevRelevanceFilter
from agents.replay_recorder import ReplayRecorder
from shadow.typesafe import TypeSafeAdapter, TypeSafeConfig
from tests.shadow_relevance_test import FakeAsyncClient, FakeResponse, _records, _typesafe_payload
from tests.shadow_pipeline_test import _item, _AsyncClient


class JevProductionTests(unittest.IsolatedAsyncioTestCase):
    async def test_choice_controls_keep_and_raw_unused_noul_is_preserved(self):
        payload = _typesafe_payload()
        payload["answers"]["r_0000"]["confidence"] = 0
        payload["answers"]["r_0001"]["confidence"] = 0
        recorder = ReplayRecorder()
        service = JevRelevanceFilter(api_key="private-test-key", recorder=recorder,
            http_client=FakeAsyncClient([FakeResponse(payload)]))
        result = await service.evaluate(_records())
        self.assertEqual([row["effective_keep"] for row in result["decisions"]], [True, False, True])
        self.assertEqual(result["decisions"][2]["fallback_reason"], "insufficient_evidence")
        calls = recorder.snapshot()["calls"]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["context"]["decision_item_count"], 3)
        self.assertEqual(call["context"]["decision_question_count"], 6)
        self.assertEqual(call["delta_events"], 1)
        self.assertEqual(call["input_tokens"], 120)
        self.assertEqual(call["output_tokens"], 30)
        self.assertEqual(call["outcome"], "ok")
        replay = json.loads(call["deltas"]["text"][0])
        self.assertEqual(replay["raw_response"], payload)
        self.assertEqual(replay["articles"][0]["critical_probability"], .95)
        self.assertIsNone(replay["articles"][1]["critical_probability"])
        self.assertIsNone(replay["articles"][2]["critical_probability"])
        self.assertNotIn("private-test-key", json.dumps(calls))
        self.assertEqual(json.loads(call["prompt"]["messages"])["state"]["article_0000"], _records()[0])

    async def test_http_retries_are_separate_real_attempts(self):
        recorder = ReplayRecorder()
        responses = [FakeResponse({"error": "busy"}, 503), FakeResponse(_typesafe_payload())]
        service = JevRelevanceFilter(api_key="test-key", recorder=recorder,
            http_client=FakeAsyncClient(responses),
            config=TypeSafeConfig(backoff_initial_seconds=0, backoff_max_seconds=0))
        result = await service.evaluate(_records())
        calls = recorder.snapshot()["calls"]
        self.assertEqual(result["status"], "complete")
        self.assertEqual([call["context"]["attempt"] for call in calls], [1, 2])
        self.assertEqual([call["outcome"] for call in calls], ["failed", "ok"])
        self.assertEqual([call["delta_events"] for call in calls], [1, 1])
        self.assertEqual(calls[0]["caller"], calls[1]["caller"])
        self.assertLessEqual(calls[0]["end_ms"], calls[1]["start_ms"])

    async def test_transport_failure_has_no_invented_output(self):
        recorder = ReplayRecorder()
        service = JevRelevanceFilter(api_key="test-key", recorder=recorder,
            http_client=FakeAsyncClient([httpx.ReadTimeout("private-test-key")]),
            config=TypeSafeConfig(max_attempts=1))
        result = await service.evaluate(_records())
        self.assertTrue(all(row["effective_keep"] for row in result["decisions"]))
        call = recorder.snapshot()["calls"][0]
        self.assertEqual(call["outcome"], "failed")
        self.assertEqual(call["delta_events"], 0)
        self.assertIsNone(call["input_tokens"])
        self.assertNotIn("private-test-key", json.dumps(call))

    async def test_missing_key_retains_without_fabricated_attempt(self):
        recorder = ReplayRecorder()
        result = await JevRelevanceFilter(api_key="", recorder=recorder).evaluate(_records())
        self.assertTrue(all(row["fallback_reason"] == "missing_api_key" for row in result["decisions"]))
        self.assertEqual(recorder.snapshot()["calls"], [])

    async def test_invalid_response_retains_and_records_raw(self):
        raw = {"model": "wrong-model", "answers": {},
               "usage": {"input_tokens": 37, "output_tokens": 11}}
        recorder = ReplayRecorder()
        result = await JevRelevanceFilter(api_key="key", recorder=recorder,
            http_client=FakeAsyncClient([FakeResponse(raw)])).evaluate(_records())
        self.assertTrue(all(row["effective_keep"] for row in result["decisions"]))
        call = recorder.snapshot()["calls"][0]
        self.assertEqual(call["outcome"], "failed")
        self.assertEqual(json.loads(call["deltas"]["text"][0])["raw_response"], raw)
        self.assertEqual(call["input_tokens"], 37)
        self.assertEqual(call["output_tokens"], 11)

    async def test_observer_cannot_mutate_transport_or_decisions(self):
        client = FakeAsyncClient([FakeResponse(_typesafe_payload())])
        def bad_observer(event):
            event.get("body", {}).clear()
            event.get("decisions", []).clear()
            raise RuntimeError("broken replay")
        result = await TypeSafeAdapter(api_key="key", http_client=client,
            attempt_observer=bad_observer).evaluate(_records())
        self.assertEqual(len(result["decisions"]), 3)
        self.assertEqual(len(client.calls[0]["json"]["questions"]), 6)

    async def test_production_bounded_evidence_and_keyword_filter(self):
        items = [_item("a" * 20, "OpenAI " + "a" * 400, "content " * 90),
                 _item("b" * 20, "Weather", "Cold tomorrow")]
        client = _AsyncClient("{}")
        with patch.dict("os.environ", {"NEWS_RELEVANCE_PROVIDER": "typesafe"}):
            analyzer = NewsAnalyzer(async_client=client)
        capture = []
        async def evaluate(records):
            capture.extend(records)
            return {"status": "complete", "decisions": [{"id": row["id"], "effective_keep": False}
                                                        for row in records]}
        with patch("agents.jev_relevance.JevRelevanceFilter") as filter_class:
            filter_class.return_value.evaluate.side_effect = evaluate
            await analyzer.analyze(items)
        self.assertEqual(len(capture), 1)
        self.assertEqual(set(capture[0]), {"id", "title", "source", "snippet"})
        self.assertEqual(capture[0]["title"], analyzer._clip_context_text(items[0].title, 300))
        self.assertEqual(capture[0]["snippet"], analyzer._clip_context_text(items[0].content, 300) + "...")
        self.assertEqual(client.calls, [])

    async def test_default_legacy_and_replay_isolation(self):
        with patch.dict("os.environ", {"NEWS_RELEVANCE_PROVIDER": "llm"}):
            self.assertEqual(NewsAnalyzer(async_client=_AsyncClient("{}")).news_relevance_provider, "llm")
        with patch.dict("os.environ", {"NEWS_RELEVANCE_PROVIDER": "typesafe"}):
            analyzer = NewsAnalyzer(async_client=_AsyncClient("{}"), replay_context=SimpleNamespace())
            self.assertEqual(analyzer.news_relevance_provider, "llm")
            with self.assertRaisesRegex(ValueError, "require a replay_context"):
                NewsAnalyzer(async_client=_AsyncClient("{}"), relevance_strategy=object())

    async def test_replay_capture_disabled_does_not_change_decisions(self):
        with patch.dict("os.environ", {"LLM_REPLAY_CAPTURE": "false"}):
            recorder = ReplayRecorder()
        result = await JevRelevanceFilter(api_key="key", recorder=recorder,
            http_client=FakeAsyncClient([FakeResponse(_typesafe_payload())])).evaluate(_records())
        self.assertEqual([row["effective_keep"] for row in result["decisions"]], [True, False, True])
        self.assertEqual(recorder.snapshot()["calls"], [])

    async def test_raw_credential_echo_is_redacted(self):
        recorder = ReplayRecorder()
        raw = {"error": "echo-private-key"}
        await JevRelevanceFilter(api_key="echo-private-key", recorder=recorder,
            http_client=FakeAsyncClient([FakeResponse(raw, 401)])).evaluate(_records())
        call = recorder.snapshot()["calls"][0]
        output = json.loads(call["deltas"]["text"][0])
        self.assertIsNone(output["raw_response"])
        self.assertTrue(output["raw_response_redacted"])
        self.assertNotIn("echo-private-key", json.dumps(call))

    async def test_output_is_recorded_only_when_response_arrives(self):
        now = [1000.0]
        recorder = ReplayRecorder(clock=lambda: now[0])
        class Client:
            async def post(self, url, **kwargs):
                active = recorder.snapshot()["calls"][0]
                self_at_send = active["delta_events"]
                assert self_at_send == 0
                now[0] += 1.5
                return FakeResponse(_typesafe_payload())
        await JevRelevanceFilter(api_key="key", recorder=recorder, http_client=Client()).evaluate(_records())
        call = recorder.snapshot()["calls"][0]
        self.assertEqual(call["start_ms"], 0)
        self.assertEqual(call["deltas"]["t"], [1500])
        self.assertEqual(call["end_ms"], 1500)

    async def test_concurrent_chunks_do_not_share_replay_calls(self):
        class Client:
            async def post(self, url, **kwargs):
                await asyncio.sleep(0)
                payload = _typesafe_payload()
                payload["answers"] = {k: v for k, v in payload["answers"].items() if k.endswith("0000")}
                return FakeResponse(payload)
        recorder = ReplayRecorder()
        result = await JevRelevanceFilter(api_key="key", recorder=recorder, http_client=Client(),
            config=TypeSafeConfig(batch_size=1)).evaluate(_records())
        calls = recorder.snapshot()["calls"]
        self.assertEqual(len(calls), 3)
        self.assertEqual(len({call["caller"] for call in calls}), 3)
        self.assertTrue(all(call["outcome"] == "ok" for call in calls))
        self.assertEqual(len(result["decisions"]), 3)


    async def test_production_uses_recall_first_v5_policy_with_unchanged_mechanics(self):
        from pathlib import Path
        from agents import jev_relevance
        from shadow.settings import load_policy
        root = Path(jev_relevance.__file__).resolve().parents[1]
        v4 = load_policy(root / "config/shadow/news-relevance-v4-frozen.json")
        v5 = load_policy(jev_relevance.POLICY_PATH)
        self.assertEqual(jev_relevance.POLICY_PATH.name, "news-relevance-v5-frozen.json")
        self.assertEqual(JevRelevanceFilter(api_key="").policy, v5)
        self.assertTrue(v4["frozen"] and v5["frozen"])
        self.assertEqual(v4["version"], "news-relevance-v4-frozen")
        self.assertEqual(v5["version"], "news-relevance-v5-frozen")
        self.assertGreater(v5["prospective_start_date"], v4["prospective_start_date"])
        instruction = v5["relevance_question"]["instructions"]["instruction"]
        self.assertIn("Use irrelevant only when AI is absent or merely incidental.", instruction)
        # Only the Choice wording and its identifying metadata change; model, batching,
        # budgets, fallback, the importance Noul and the three Choice labels do not.
        changed = {"version", "prospective_start_date", "supersedes", "rationale",
                   "rubric_version", "relevance_question"}
        self.assertEqual({k: v for k, v in v4.items() if k not in changed},
                         {k: v for k, v in v5.items() if k not in changed})
        self.assertEqual(v5["supersedes"], v4["version"])
        self.assertEqual({**v4["relevance_question"], "instructions": None},
                         {**v5["relevance_question"], "instructions": None})
        self.assertEqual(v5["relevance_question"]["criteria"],
                         {"relevant": None, "irrelevant": None, "insufficient_evidence": None})
        self.assertEqual(v4["relevance_question"]["instructions"]["article"],
                         v5["relevance_question"]["instructions"]["article"])

if __name__ == "__main__":
    unittest.main()
