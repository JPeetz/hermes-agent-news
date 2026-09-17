"""Offline contract tests for the bounded relevance adapters.

These tests use injected fake HTTP clients only and never call a paid model.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from shadow.budget import BudgetLimits, RequestBudget
from shadow.contracts import sha256_json
from shadow.incumbent import IncumbentAdapter, OpenAIChatConfig
from shadow.typesafe import (
    DEFAULT_TYPESAFE_MODEL,
    TypeSafeAdapter,
    TypeSafeConfig,
)


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200, headers: dict[str, str] | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"x-request-id": "req-1"}

    def json(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


class FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _records():
    return [
        {"id": "article-keep", "title": "OpenAI ships a frontier model", "source": "Example", "snippet": "A model release."},
        {"id": "article-reject", "title": "Local weather forecast", "source": "Example", "snippet": "No AI connection."},
        {"id": "article-abstain", "title": "AI company changes office policy", "source": "Example", "snippet": "The bounded excerpt is ambiguous."},
    ]


def _typesafe_payload(*, model: str = DEFAULT_TYPESAFE_MODEL, include_third: bool = True):
    answers = {
        "r_0000": {"type": "noul", "noul": 0.95},
        "s_0000": {"type": "noul", "noul": 0.99},
        "r_0001": {"type": "noul", "noul": 0.04},
        "s_0001": {"type": "noul", "noul": 0.99},
    }
    if include_third:
        answers.update({
            "r_0002": {"type": "noul", "noul": 0.50},
            "s_0002": {"type": "noul", "noul": 0.99},
        })
    return {
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


def _frozen_input(records=None):
    records = records or _records()
    return {
        "schema_version": "news-relevance-input/v1",
        "records": records,
        "ordered_ids": [record["id"] for record in records],
        "input_sha256": sha256_json(records),
        "system_prompt": "Frozen incumbent system prompt.",
        "user_message": "Frozen incumbent user message with source data.",
    }


class TypeSafeAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_exact_questions_and_conservative_decisions(self):
        client = FakeAsyncClient([FakeResponse(_typesafe_payload())])
        adapter = TypeSafeAdapter(http_client=client)
        result = await adapter.evaluate(
            _records(),
            policy={
                "schema_version": "news-shadow-policy/v1",
                "version": "policy-test-1",
                "model": DEFAULT_TYPESAFE_MODEL,
                "reject_max": 0.10,
                "keep_min": 0.80,
                "sufficiency_min": 0.90,
                "questions_per_article": 2,
                "chunk_size": 16,
                "concurrency": 2,
                "max_attempts": 3,
                "timeout_seconds": 30,
            },
            config=TypeSafeConfig(backoff_initial_seconds=0, backoff_max_seconds=0),
            api_key="typesafe-secret",
        )

        self.assertEqual([row["decision"] for row in result["decisions"]], ["keep", "reject", "abstain"])
        self.assertEqual([row["effective_keep"] for row in result["decisions"]], [True, False, True])
        self.assertEqual(client.calls[0]["url"], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(client.calls[0]["json"]["model"], DEFAULT_TYPESAFE_MODEL)
        self.assertEqual(client.calls[0]["json"]["state"]["articles"], _records())
        self.assertEqual(sorted(client.calls[0]["json"]["questions"]), [
            "r_0000", "r_0001", "r_0002", "s_0000", "s_0001", "s_0002",
        ])
        self.assertNotIn("confidence", client.calls[0]["json"]["questions"]["r_0000"])
        self.assertEqual(result["requests"][0]["returned_model"], DEFAULT_TYPESAFE_MODEL)
        self.assertNotIn("typesafe-secret", json.dumps(result))

    async def test_missing_question_abstains_only_the_affected_article(self):
        payload = _typesafe_payload(include_third=False)
        payload["answers"].pop("s_0001")
        client = FakeAsyncClient([FakeResponse(payload)])
        result = await TypeSafeAdapter(http_client=client).evaluate(
            _records()[:2],
            policy={"version": "policy-test-2"},
            api_key="secret",
        )
        self.assertEqual(result["decisions"][0]["decision"], "keep")
        self.assertEqual(result["decisions"][1]["decision"], "abstain")
        self.assertTrue(result["decisions"][1]["effective_keep"])
        self.assertEqual(result["decisions"][1]["fallback_reason"], "abstain_error")
        self.assertIn("invalid_answer:article-reject", result["degradations"])

    async def test_each_retry_attempt_is_budgeted_and_transport_failure_retains_superset(self):
        client = FakeAsyncClient([
            FakeResponse({"error": {"message": "busy"}}, status_code=529),
            FakeResponse(_typesafe_payload(include_third=False)),
        ])
        budget = RequestBudget(BudgetLimits(max_requests=2, max_input_tokens=100_000, max_output_tokens=100_000))
        result = await TypeSafeAdapter(http_client=client).evaluate(
            _records()[:2],
            policy={"version": "policy-test-3", "max_attempts": 2},
            config=TypeSafeConfig(max_attempts=2, backoff_initial_seconds=0, backoff_max_seconds=0),
            api_key="secret",
            budget=budget,
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(budget.snapshot()["requests"], 2)
        self.assertEqual(result["decisions"][0]["decision"], "keep")
        self.assertEqual(len(result["requests"][0]["attempts"]), 2)

    async def test_retryable_non_json_status_is_retried_before_schema_validation(self):
        client = FakeAsyncClient([
            FakeResponse(ValueError("gateway body is not JSON"), status_code=529),
            FakeResponse(_typesafe_payload(include_third=False)),
        ])
        result = await TypeSafeAdapter(http_client=client).evaluate(
            _records()[:2],
            policy={"version": "policy-test-3b", "max_attempts": 2},
            config=TypeSafeConfig(max_attempts=2, backoff_initial_seconds=0, backoff_max_seconds=0),
            api_key="secret",
        )
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["decisions"][0]["decision"], "keep")

    async def test_returned_model_mismatch_fails_closed(self):
        client = FakeAsyncClient([FakeResponse(_typesafe_payload(model="jev-latest"))])
        result = await TypeSafeAdapter(http_client=client).evaluate(
            _records()[:1], policy={"version": "policy-test-4"}, api_key="secret"
        )
        self.assertEqual(result["decisions"][0]["decision"], "abstain")
        self.assertTrue(result["decisions"][0]["effective_keep"])
        self.assertIn("unexpected_returned_model", result["degradations"])


class IncumbentAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_replays_exact_prompts_and_maps_unambiguous_prefix(self):
        records = _records()[:2]
        client = FakeAsyncClient([FakeResponse({
            "id": "chat-1",
            "model": "incumbent-model",
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"ai_article_ids":["article-keep"]}'},
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        })])
        config = OpenAIChatConfig(api_key="incumbent-secret", base_url="https://route.example/v1", model="incumbent-model")
        frozen = _frozen_input(records)
        result = await IncumbentAdapter(config=config, http_client=client).evaluate(frozen)

        self.assertEqual([row["decision"] for row in result["decisions"]], ["keep", "reject"])
        body = client.calls[0]["json"]
        self.assertEqual(client.calls[0]["url"], "https://route.example/v1/chat/completions")
        self.assertEqual(body["messages"][0]["content"], frozen["system_prompt"])
        self.assertEqual(body["messages"][1]["content"], frozen["user_message"])
        self.assertEqual(body["model"], "incumbent-model")
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertNotIn("cache", body)
        self.assertEqual(result["requests"][0]["returned_model"], "incumbent-model")
        self.assertEqual(result["requests"][0]["attempts"][0]["response_id"], "chat-1")
        self.assertEqual(result["requests"][0]["attempts"][0]["request_id"], "req-1")
        self.assertNotIn("incumbent-secret", json.dumps(result))

    async def test_rdsec_control_bypasses_cache_without_changing_frozen_messages(self):
        client = FakeAsyncClient([FakeResponse({"id": "chat-fresh", "model": "deepseek-v4.1-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ai_article_ids":["article-keep"]}'}}]})])
        frozen = _frozen_input(_records()[:2])
        config = OpenAIChatConfig(api_key="test", base_url="https://api.rdsec.trendmicro.com/prod/aiendpoint/v1",
                                  model="deepseek-v4.1-flash")
        result = await IncumbentAdapter(config, http_client=client).evaluate(frozen)
        body = client.calls[0]["json"]
        self.assertEqual(body["cache"], {"no-cache": True, "no-store": True})
        self.assertEqual(body["messages"][0]["content"], frozen["system_prompt"])
        self.assertEqual(body["messages"][1]["content"], frozen["user_message"])
        self.assertEqual(result["status"], "complete")

    async def test_ambiguous_prefix_fails_closed_to_superset(self):
        records = [
            {"id": "abcdef1234567890", "title": "A", "source": "S", "snippet": "A..."},
            {"id": "abcdef12fedcba98", "title": "B", "source": "S", "snippet": "B..."},
        ]
        client = FakeAsyncClient([FakeResponse({
            "model": "incumbent-model",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ai_article_ids":["abcdef12"]}'}}],
        })])
        result = await IncumbentAdapter(http_client=client).evaluate(
            _frozen_input(records),
            config=OpenAIChatConfig(base_url="https://route.example/v1", model="incumbent-model"),
            api_key="secret",
        )
        self.assertEqual([row["decision"] for row in result["decisions"]], ["abstain", "abstain"])
        self.assertTrue(all(row["effective_keep"] for row in result["decisions"]))
        self.assertIn("ambiguous_ids", result["degradations"])

    async def test_missing_terminal_response_fails_closed(self):
        client = FakeAsyncClient([FakeResponse({
            "model": "incumbent-model",
            "choices": [{"finish_reason": None, "message": {"content": '{"ai_article_ids":[]}'}}],
        })])
        result = await IncumbentAdapter(http_client=client).evaluate(
            _frozen_input(_records()[:1]),
            config=OpenAIChatConfig(base_url="https://route.example/v1", model="incumbent-model"),
            api_key="secret",
        )
        self.assertEqual(result["decisions"][0]["decision"], "abstain")
        self.assertIn("missing_terminal_response", result["degradations"])

    def test_config_public_dict_never_contains_secret(self):
        config = OpenAIChatConfig(api_key="do-not-serialize", base_url="https://route.example/v1", model="incumbent-model")
        self.assertNotIn("api_key", config.public_dict())
        self.assertNotIn("do-not-serialize", json.dumps(config.public_dict()))


if __name__ == "__main__":
    unittest.main()
