"""Offline regression checks for the production/replay pipeline seams.

The focused suite is intentionally mock-only.  The implementation worker only
parses this module; the repository owner runs it after the shadow bundle is
reviewed.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from agents import orchestrator as orchestrator_module
from agents.analyzers.news_analyzer import NewsAnalyzer
from agents.analyzers.reddit_analyzer import RedditAnalyzer
from agents.base import AnalyzedItem, CollectedItem
from agents.orchestrator import MainOrchestrator
from shadow.replay_context import ReplayIntegrityError


DATE = "2026-09-17"


def _item(article_id: str, title: str, content: str) -> CollectedItem:
    return CollectedItem(
        id=article_id,
        title=title,
        content=content,
        url=f"https://source.example/{article_id}",
        author="author",
        published="2026-09-16T12:00:00Z",
        source="Example Feed",
        source_type="rss",
    )


class _Response:
    model = "incumbent-model"
    usage = {"input_tokens": 11, "output_tokens": 7}
    stop_reason = "end_turn"
    thinking = ""

    def __init__(self, content: str):
        self.content = content


class _AsyncClient:
    def __init__(self, content: str):
        self.response = _Response(content)
        self.calls = []

    async def call_with_thinking(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _CaptureSpy:
    def __init__(self):
        self.inputs = []
        self.decisions = []

    def record_relevance_input(self, **kwargs):
        self.inputs.append(kwargs)

    def record_relevance_decision(self, **kwargs):
        self.decisions.append(kwargs)


class NewsFilterContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_selection_and_complete_capture_request(self):
        kept = _item("a" * 20, "OpenAI ships GPT-5", "A frontier model release.")
        rejected = _item("b" * 20, "Local weather update", "No relevant development.")
        response = '{"ai_article_ids":["' + kept.id[:16] + '"]}'
        client = _AsyncClient(response)
        capture = _CaptureSpy()
        analyzer = NewsAnalyzer(
            async_client=client, target_date=DATE, capture=capture,
        )

        result = await analyzer._filter_with_llm([kept, rejected])

        self.assertEqual([item.id for item in result], [kept.id])
        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        system = call["system"]
        user = call["messages"][0]["content"]
        self.assertIn("FRONTIER AI newsletter", system)
        self.assertIn("Return the IDs of articles relevant to frontier AI", system)
        self.assertIn(kept.title, user)
        self.assertIn(rejected.title, user)
        self.assertIn(kept.id[:16], user)

        self.assertEqual(len(capture.inputs), 1)
        captured = capture.inputs[0]
        self.assertEqual(captured["system_prompt"], system)
        self.assertEqual(captured["user_message"], user)
        canonical = [
            {key: row[key] for key in ("id", "title", "source", "snippet")}
            for row in captured["records"]
        ]
        self.assertEqual(captured["input_sha256"], analyzer._filter_input_hash(canonical))
        self.assertEqual(captured["renderer"], "news_filter_v1")
        self.assertEqual(capture.decisions[0]["mapped_kept_ids"], [kept.id])
        self.assertEqual(capture.decisions[0]["model"], "incumbent-model")
        self.assertEqual(capture.decisions[0]["usage"], {"input_tokens": 11, "output_tokens": 7})
        self.assertNotIn("request_id", capture.decisions[0])

    async def test_optional_capture_preserves_incumbent_selection(self):
        kept = _item("c" * 20, "Anthropic releases Claude", "A model announcement.")
        response = '{"ai_article_ids":["' + kept.id[:16] + '"]}'
        plain_client = _AsyncClient(response)
        captured_client = _AsyncClient(response)
        plain = NewsAnalyzer(async_client=plain_client, target_date=DATE)
        captured = NewsAnalyzer(
            async_client=captured_client, target_date=DATE, capture=_CaptureSpy(),
        )

        plain_result = await plain._filter_with_llm([kept])
        captured_result = await captured._filter_with_llm([kept])

        self.assertEqual([item.id for item in plain_result], [item.id for item in captured_result])
        self.assertEqual(plain_client.calls[0]["caller"], captured_client.calls[0]["caller"])
        self.assertEqual(plain_client.calls[0]["profile"], captured_client.calls[0]["profile"])

    async def test_precomputed_ids_bypass_incumbent_call(self):
        kept = _item("d" * 20, "Google ships a model", "A frontier release.")
        rejected = _item("e" * 20, "Unrelated item", "No model news.")
        client = _AsyncClient('{"ai_article_ids":[]}')
        analyzer = NewsAnalyzer(
            async_client=client,
            target_date=DATE,
            replay_context=SimpleNamespace(),
            precomputed_exact_kept_ids=[kept.id],
        )

        result = await analyzer._filter_with_llm([kept, rejected])

        self.assertEqual([item.id for item in result], [kept.id])
        self.assertEqual(client.calls, [])


class ReplayConstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_replay_skips_sources_catalog_and_hero(self):
        replay = SimpleNamespace(frozen_report_date=DATE, grounding_context="frozen grounding")
        with mock.patch.object(orchestrator_module, "NewsGatherer") as news, \
                mock.patch.object(orchestrator_module, "ResearchGatherer") as research, \
                mock.patch.object(orchestrator_module, "SocialGatherer") as social, \
                mock.patch.object(orchestrator_module, "RedditGatherer") as reddit, \
                mock.patch.object(orchestrator_module, "initialize_hero_generator") as hero, \
                mock.patch.object(orchestrator_module, "HERO_GENERATOR_AVAILABLE", True), \
                mock.patch.object(orchestrator_module, "AnthropicClient"), \
                mock.patch.object(orchestrator_module, "AsyncAnthropicClient"), \
                mock.patch.object(orchestrator_module, "EcosystemContextManager") as catalog:
            orchestrator = MainOrchestrator(target_date=DATE, replay_context=replay)

        self.assertEqual(orchestrator.gatherers, {})
        for source in (news, research, social, reddit):
            source.assert_not_called()
        hero.assert_not_called()
        catalog.return_value.initialize.assert_not_called()

    async def test_replay_phase_0_uses_frozen_grounding_without_catalog_refresh(self):
        replay = SimpleNamespace(frozen_report_date=DATE, grounding_context="frozen grounding")
        with mock.patch.object(orchestrator_module, "AnthropicClient"), \
                mock.patch.object(orchestrator_module, "AsyncAnthropicClient"):
            orchestrator = MainOrchestrator(target_date=DATE, replay_context=replay)
        orchestrator._load_replay_gathering = mock.Mock(side_effect=RuntimeError("stop after phase 0"))
        tracker = SimpleNamespace(start=mock.Mock())
        recorder = SimpleNamespace(begin_run=mock.Mock())
        with mock.patch.object(orchestrator.ecosystem_manager, "initialize", new=mock.AsyncMock()) as catalog:
            with mock.patch.object(orchestrator_module, "reset_tracker", return_value=tracker), \
                    mock.patch.object(orchestrator_module, "get_recorder", return_value=recorder):
                with self.assertRaisesRegex(RuntimeError, "stop after phase 0"):
                    await orchestrator.run()
        catalog.assert_not_awaited()


class RedditRankingCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reduce_uses_corrected_summary_and_exact_ranked_ids(self):
        # The production incident contained an initial placeholder ranking,
        # followed by a complete correction in the same successful response.
        items = [
            AnalyzedItem(
                item=_item(f"{index:012x}", f"OpenAI discussion {index}", "Community evidence."),
                summary=f"Analyzed discussion {index}.", importance_score=95 - index,
                reasoning="Substantive community discussion.", themes=[],
            )
            for index in range(1, 12)
        ]
        corrected_ids = [item.item.id for item in reversed(items[1:])]
        corrected_summary = "**r/LocalLLaMA** discussed open models and agent security."
        draft = {"top_10": ["10a..."] + [item.item.id for item in items[:9]],
                 "category_summary": "placeholder"}
        corrected = {"top_10": corrected_ids, "category_summary": corrected_summary}
        valid_correction = json.dumps(corrected)
        corrections = {
            "valid": valid_correction,
            "trailing_comma": valid_correction[:-1] + ",}",
            "missing_comma": valid_correction.replace('], "category_summary"', '] "category_summary"'),
        }
        for variant, correction in corrections.items():
            with self.subTest(variant=variant):
                content = (
                    f"```json\n{json.dumps(draft)}\n```\n"
                    "Wait, let me correct that output.\n"
                    f"```json\n{correction}\n```"
                )
                client = _AsyncClient(content)
                analyzer = RedditAnalyzer(async_client=client, target_date="2026-09-18")
                with mock.patch("agents.staleness_checker.StalenessChecker") as checker:
                    checker.return_value.process_items = mock.AsyncMock(return_value=0)
                    report = await analyzer._reduce_phase(items, [], [], "Saved batch reasoning.")
                    checker.return_value.process_items.assert_awaited_once()

                self.assertEqual(report.category_summary, corrected_summary)
                self.assertEqual([item.item.id for item in report.top_items], corrected_ids)
                self.assertEqual(report.all_items, items)
                self.assertEqual(report.degradations, [])
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(client.calls[0]["caller"], "reddit_analyzer.reduce_rank")


class ReplayIntegrityPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_evidence_is_rethrown_by_reduce_broad_catch(self):
        item = _item("f" * 20, "A model story", "A relevant article.")
        analyzed = AnalyzedItem(
            item=item, summary="summary", importance_score=80,
            reasoning="reason", themes=[],
        )
        analyzer = NewsAnalyzer(
            async_client=_AsyncClient("{}"), target_date=DATE,
            replay_context=SimpleNamespace(),
        )
        with mock.patch("agents.staleness_checker.StalenessChecker") as checker:
            checker.return_value.process_items = mock.AsyncMock(
                side_effect=ReplayIntegrityError("missing frozen evidence")
            )
            with self.assertRaisesRegex(ReplayIntegrityError, "missing frozen evidence"):
                await analyzer._reduce_phase([analyzed], [], [], "")


if __name__ == "__main__":
    unittest.main()
