"""Offline contract tests for the bounded relevance adapters.

These tests use injected fake HTTP clients only and never call a paid model.
"""

from __future__ import annotations

import json
import io
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from shadow.budget import BudgetLimits, RequestBudget
from shadow.contracts import sha256_json
from shadow.contracts import BundleValidationError
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
        {"id": "article-borderline", "title": "AI company changes office policy", "source": "Example", "snippet": "The bounded excerpt is ambiguous."},
    ]


def _choice_answer(label="relevant"):
    probabilities = {key: .02 for key in ("relevant", "irrelevant", "insufficient_evidence")}
    probabilities[label] = .96
    return {"type": "choice", "choice": label, "probabilities": probabilities, "confidence": .92}


def _typesafe_payload(*, model: str = DEFAULT_TYPESAFE_MODEL, include_third: bool = True):
    answers = {
        "r_0000": _choice_answer("relevant"),
        "c_0000": {"type": "noul", "noul": .95},
        "r_0001": _choice_answer("irrelevant"),
        "c_0001": {"type": "noul", "noul": .99},
    }
    if include_third:
        answers.update({
            "r_0002": _choice_answer("insufficient_evidence"),
            "c_0002": {"type": "noul", "noul": .99},
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
    async def test_choice_controls_relevance_without_a_probability_or_critical_gate(self):
        for probability in (0, .5, .9, 1):
            with self.subTest(probability=probability):
                choice = {"type": "choice", "choice": "relevant", "confidence": .10,
                          "probabilities": {"relevant": .4, "irrelevant": .3, "insufficient_evidence": .3}}
                payload = {"model": DEFAULT_TYPESAFE_MODEL, "answers": {
                    "r_0000": choice, "c_0000": {"type": "noul", "noul": probability},
                }}
                result = await TypeSafeAdapter(http_client=FakeAsyncClient([FakeResponse(payload)])).evaluate(
                    _records()[:1], api_key="test"
                )
                row = result["decisions"][0]
                self.assertEqual(row["decision"], "keep")
                self.assertEqual(row["relevance"], "relevant")
                self.assertEqual(row["probabilities"], choice["probabilities"])
                self.assertEqual(row["confidence"], .10)
                self.assertEqual(row["critical_probability"], probability)
                self.assertIsNone(row["fallback_reason"])

    async def test_invalid_used_critical_noul_reports_error_without_replacing_valid_choice(self):
        for probability in (None, True, "0.9", -0.1, 1.1, float("nan")):
            with self.subTest(probability=probability):
                payload = {"model": DEFAULT_TYPESAFE_MODEL, "answers": {
                    "r_0000": _choice_answer(), "c_0000": {"type": "noul", "noul": probability},
                }}
                result = await TypeSafeAdapter(http_client=FakeAsyncClient([FakeResponse(payload)])).evaluate(
                    _records()[:1], api_key="test"
                )
                row = result["decisions"][0]
                self.assertIsNone(row["fallback_reason"])
                self.assertEqual(row["relevance"], "relevant")
                self.assertEqual(row["decision"], "keep")
                self.assertIsNone(row["critical_probability"])
                self.assertIn("invalid_critical_answer:article-keep", result["degradations"])
                self.assertTrue(row["effective_keep"])

    async def test_unused_critical_noul_is_discarded_even_if_missing_or_invalid(self):
        for label in ("irrelevant", "insufficient_evidence"):
            for critical in (None, {"type": "noul", "noul": 1}, {"type": "noul", "noul": "bad"}):
                with self.subTest(label=label, critical=critical):
                    answers = {"r_0000": _choice_answer(label)}
                    if critical is not None:
                        answers["c_0000"] = critical
                    payload = {"model": DEFAULT_TYPESAFE_MODEL, "answers": answers}
                    result = await TypeSafeAdapter(http_client=FakeAsyncClient([FakeResponse(payload)])).evaluate(
                        _records()[:1], api_key="test"
                    )
                    self.assertEqual(result["status"], "complete")
                    self.assertEqual(result["degradations"], [])
                    row = result["decisions"][0]
                    self.assertEqual(row["relevance"], label)
                    self.assertIsNone(row["critical_probability"])
                    self.assertEqual(row["decision"], "reject" if label == "irrelevant" else "abstain")

    async def test_malformed_choice_retains_item_with_explicit_error(self):
        for change in ({"choice": "other"}, {"confidence": None}, {"probabilities": {"relevant": .9}},
                       {"probabilities": {"relevant": True, "irrelevant": 0, "insufficient_evidence": 0}},
                       {"probabilities": {"relevant": .01, "irrelevant": .98, "insufficient_evidence": .01}}):
            with self.subTest(change=change):
                payload = {"model": DEFAULT_TYPESAFE_MODEL, "answers": {
                    "r_0000": {**_choice_answer(), **change}, "c_0000": {"type": "noul", "noul": .99}}}
                result = await TypeSafeAdapter(http_client=FakeAsyncClient([FakeResponse(payload)])).evaluate(
                    _records()[:1], api_key="test"
                )
                row = result["decisions"][0]
                self.assertEqual(row["fallback_reason"], "abstain_error")
                self.assertIsNone(row["relevance"])
                self.assertIsNone(row["critical_probability"])
                self.assertTrue(row["effective_keep"])

    async def test_legacy_evidence_gate_policy_is_not_silently_reinterpreted(self):
        client = FakeAsyncClient([])
        with self.assertRaises(BundleValidationError):
            await TypeSafeAdapter(http_client=client).evaluate(
                _records(), policy={"sufficiency_min": .9}, api_key="test"
            )
        self.assertEqual(client.calls, [])

    async def test_exact_questions_and_inclusion_decisions(self):
        client = FakeAsyncClient([FakeResponse(_typesafe_payload())])
        adapter = TypeSafeAdapter(http_client=client)
        result = await adapter.evaluate(
            _records(),
            policy={
                "schema_version": "news-shadow-policy/v3",
                "version": "policy-test-1",
                "model": DEFAULT_TYPESAFE_MODEL,
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
        self.assertEqual([row["relevance"] for row in result["decisions"]],
                         ["relevant", "irrelevant", "insufficient_evidence"])
        self.assertEqual([row["critical_probability"] for row in result["decisions"]], [.95, None, None])
        self.assertEqual(client.calls[0]["url"], "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(client.calls[0]["json"]["model"], DEFAULT_TYPESAFE_MODEL)
        self.assertEqual(client.calls[0]["json"]["state"], {
            "article_0000": _records()[0],
            "article_0001": _records()[1],
            "article_0002": _records()[2],
        })
        self.assertEqual(sorted(client.calls[0]["json"]["questions"]), [
            "c_0000", "c_0001", "c_0002", "r_0000", "r_0001", "r_0002",
        ])
        self.assertEqual([row["probabilities"] for row in result["decisions"]],
                         [_choice_answer(label)["probabilities"] for label in ("relevant", "irrelevant", "insufficient_evidence")])
        self.assertTrue(all(row["fallback_reason"] is None for row in result["decisions"]))
        question = client.calls[0]["json"]["questions"]["r_0000"]
        self.assertIsInstance(question["instructions"], dict)
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"relevant", "irrelevant", "insufficient_evidence"})
        self.assertEqual(client.calls[0]["json"]["questions"]["c_0000"]["type"], "noul")
        self.assertNotIn("confidence", client.calls[0]["json"]["questions"]["r_0000"])
        self.assertEqual(result["requests"][0]["returned_model"], DEFAULT_TYPESAFE_MODEL)
        self.assertNotIn("typesafe-secret", json.dumps(result))

    async def test_article_variable_binding_survives_reordering_and_chunk_boundaries(self):
        class ResolvingClient:
            async def post(self, url, **kwargs):
                body = kwargs["json"]
                answers = {}
                for key, question in body["questions"].items():
                    # Resolve the exact variable in each question, as opposed
                    # to assuming that question IDs supply model-visible scope.
                    variables = re.findall(r"`([^`]+)`", json.dumps(question["instructions"]))
                    self_test.assertEqual(len(variables), 1)
                    article = body["state"][variables[0]]
                    label = {
                        "article-keep": "relevant", "article-reject": "irrelevant", "article-borderline": "insufficient_evidence",
                    }[article["id"]]
                    answers[key] = _choice_answer(label) if question["type"] == "choice" else {"type": "noul", "noul": .99}
                return FakeResponse({"model": DEFAULT_TYPESAFE_MODEL, "answers": answers,
                                     "usage": {"input_tokens": 100, "output_tokens": 20}})

        self_test = self
        for records in (_records(), list(reversed(_records()))):
            result = await TypeSafeAdapter(http_client=ResolvingClient()).evaluate(
                records, policy={"version": "binding-test", "chunk_size": 2}, api_key="test"
            )
            self.assertEqual({row["id"]: row["decision"] for row in result["decisions"]}, {
                "article-keep": "keep", "article-reject": "reject", "article-borderline": "abstain",
            })

    async def test_missing_question_abstains_only_the_affected_article(self):
        payload = _typesafe_payload(include_third=False)
        payload["answers"].pop("r_0001")
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
        progress, stdout = io.StringIO(), io.StringIO()
        with redirect_stderr(progress), redirect_stdout(stdout):
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
        self.assertEqual(stdout.getvalue(), "")
        self.assertRegex(progress.getvalue(), r'"input_tokens"\s*:\s*20')
        self.assertRegex(progress.getvalue(), r'"output_tokens"\s*:\s*5')
        self.assertNotIn("incumbent-secret", progress.getvalue())
        self.assertNotIn(frozen["system_prompt"], progress.getvalue())
        self.assertNotIn(frozen["user_message"], progress.getvalue())

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
