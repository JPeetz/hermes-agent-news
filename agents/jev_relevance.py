"""Production news relevance using the reviewed, bounded Jev policy.

Choice alone controls retention. The replay observer records a single typed
response per actual HTTP attempt and never influences the filtering result.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shadow.budget import BudgetLimits, RequestBudget
from shadow.typesafe import TypeSafeAdapter, TypeSafeConfig
from .replay_recorder import DELTA_TEXT, get_recorder

POLICY_PATH = Path(__file__).resolve().parents[1] / "config/shadow/news-relevance-v4-frozen.json"


class JevRelevanceError(RuntimeError):
    """Secret-free failure marker for a recorded provider attempt."""


def _production_decision(decision: dict[str, Any]) -> dict[str, Any]:
    result = dict(decision)
    if result.get("relevance") == "insufficient_evidence":
        result["fallback_reason"] = "insufficient_evidence"
    if result.get("relevance") != "relevant":
        result["critical_probability"] = None
    return result


class JevRelevanceFilter:
    def __init__(self, *, api_key: str | None = None, http_client=None,
                 recorder=None, config: TypeSafeConfig | None = None):
        self._api_key = os.getenv("TYPESAFE_API_KEY", "") if api_key is None else api_key
        self._recorder = get_recorder() if recorder is None else recorder
        self._calls = {}
        self.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        self.config = config or TypeSafeConfig(
            model=self.policy["model"], batch_size=self.policy["chunk_size"],
            max_concurrency=self.policy["concurrency"],
            max_attempts=self.policy["max_attempts"],
            timeout_seconds=self.policy["timeout_seconds"],
            max_http_attempts=self.policy["candidate_budget"]["max_requests"],
        )
        self.adapter = TypeSafeAdapter(
            config=self.config, api_key=self._api_key, http_client=http_client,
            attempt_observer=self._observe_attempt,
        )

    def _observe_attempt(self, event: dict[str, Any]) -> None:
        # The adapter isolates this callback and deep-copies its input, so a
        # broken recorder cannot fail inference or modify a request/decision.
        key = (event["chunk_index"], event["attempt"])
        if event["event"] == "start":
            count = len(event["records"])
            call_id = self._recorder.start_call(None, {
                "caller": f"jev.filter.batch_{event['chunk_index']}",
                "provider_id": "typesafe", "provider_model": event["model"],
                "interaction_type": "decision", "decision_item_count": count,
                "decision_question_count": 2 * count, "attempt": event["attempt"],
                "messages_text": json.dumps(event["body"], ensure_ascii=False),
            })
            self._calls[key] = call_id
            self._recorder.mark_started(call_id)
            return
        call_id = self._calls.pop(key, None)
        if event["event"] == "error":
            self._recorder.finish_call(call_id, error=JevRelevanceError(event["error"]),
                                       context_update={"usage_measured": False})
            return
        decisions = [_production_decision(row) for row in event["decisions"]]
        articles = [{**record, **{field: decision.get(field) for field in (
            "relevance", "probabilities", "confidence", "critical_probability",
            "effective_keep", "fallback_reason",
        )}} for record, decision in zip(event["records"], decisions)]
        raw_response = event["raw_response"]
        # Preserve the exact parsed response under normal operation. If a
        # misbehaving provider echoes our credential, omit the raw object.
        raw_redacted = bool(self._api_key and self._api_key in json.dumps(raw_response))
        payload = {"schema_version": "jev-relevance-replay/v1", "articles": articles,
                   "raw_response": None if raw_redacted else raw_response}
        if raw_redacted:
            payload["raw_response_redacted"] = True
        self._recorder.record_delta(call_id, DELTA_TEXT, json.dumps(payload, ensure_ascii=False))
        usage = event["usage"]
        retained = sum(bool(row.get("fallback_reason")) for row in decisions)
        self._recorder.finish_call(
            call_id, response=SimpleNamespace(usage=SimpleNamespace(**usage)),
            error=JevRelevanceError(event["error"]) if event.get("error") else None,
            context_update={
                "estimated_cost_usd": event["estimated_cost_usd"],
                "usage_measured": all(usage.get(k) is not None for k in ("input_tokens", "output_tokens")),
                "decision_items_kept": sum(row["effective_keep"] for row in decisions) - retained,
                "decision_items_excluded": sum(not row["effective_keep"] for row in decisions),
                "decision_items_retained": retained,
            },
        )

    async def evaluate(self, records: list[dict[str, str]]) -> dict[str, Any]:
        result = await self.adapter.evaluate(
            records, policy=self.policy,
            budget=RequestBudget(BudgetLimits(**self.policy["candidate_budget"])),
        )
        result["decisions"] = [_production_decision(row) for row in result["decisions"]]
        return result
