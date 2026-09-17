"""Isolated replay adapter for the incumbent OpenAI-compatible news filter.

This module replays the frozen filter prompt through a caller-supplied route.
It intentionally does not import the production Anthropic client or inspect
``ANTHROPIC_MODEL``/any other environment variable.  The returned model and a
terminal chat-completions response are required before any selection is
accepted.  Prefix IDs retain the incumbent's historical behavior, while an
ambiguous prefix fails closed to the full superset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .contracts import (
    BundleValidationError as _BundleValidationError,
    sha256_json as _shared_sha256_json,
    validate_input as _shared_validate_input,
)
from .budget import BudgetExceeded as _BudgetExceeded
from .progress import Progress


@dataclass(frozen=True)
class OpenAIChatConfig:
    """Explicit route settings for one incumbent replay call.

    ``api_key`` is accepted for ergonomic use by the runner but intentionally
    excluded from ``public_dict`` and every returned diagnostic.
    """

    api_key: str | None = field(default=None, repr=False, compare=False)
    base_url: str = ""
    model: str = ""
    max_output_tokens: int = 4096
    timeout_seconds: float = 240.0
    max_attempts: int = 3
    max_http_attempts: int = 3
    # The production filter calls the OpenAI-chat route with the QUICK profile,
    # which maps to the documented ``reasoning: {effort: high}`` control for
    # non-Anthropic models.  Keep this explicit so a fresh control does not
    # silently inherit a provider default; ``None`` is an explicit opt-out for
    # a route whose preflight proves that it lacks this control.
    reasoning_effort: str | None = "high"
    backoff_initial_seconds: float = 0.5
    backoff_max_seconds: float = 5.0
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 409, 429, 500, 502, 503, 504, 529})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("incumbent base_url must be explicit and non-empty")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("incumbent base_url must be HTTPS without URL credentials or metadata")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("incumbent model must be explicit and non-empty")
        if self.max_output_tokens < 1:
            raise ValueError("incumbent max_output_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("incumbent timeout_seconds must be positive")
        if self.max_attempts < 1 or self.max_http_attempts < 1:
            raise ValueError("incumbent attempt limits must be positive")
        if self.reasoning_effort is not None and self.reasoning_effort not in {
            "max", "xhigh", "high", "medium", "low", "minimal", "none",
        }:
            raise ValueError("incumbent reasoning_effort is unsupported")
        if self.backoff_initial_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("incumbent backoff cannot be negative")
        for name in ("input_cost_per_million", "output_cost_per_million"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"incumbent {name} must be finite and nonnegative")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("api_key", None)
        result["retry_statuses"] = sorted(self.retry_statuses)
        return result

    def to_public_dict(self) -> dict[str, Any]:
        return self.public_dict()


class _AttemptGate:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.used = 0
        self._lock = asyncio.Lock()

    async def take(self) -> bool:
        async with self._lock:
            if self.used >= self.maximum:
                return False
            self.used += 1
            return True


class _UsageAccumulator:
    def __init__(self):
        self._input = 0
        self._output = 0
        self._cost = 0.0
        self._input_unknown = False
        self._output_unknown = False
        self._cost_unknown = False
        self.http_attempts = 0

    def add(self, usage: Mapping[str, Any] | None, cost: float | None) -> None:
        self.http_attempts += 1
        usage = usage or {}
        input_tokens = _optional_nonnegative_int(usage.get("prompt_tokens", usage.get("input_tokens")))
        output_tokens = _optional_nonnegative_int(usage.get("completion_tokens", usage.get("output_tokens")))
        if input_tokens is None:
            self._input_unknown = True
        else:
            self._input += input_tokens
        if output_tokens is None:
            self._output_unknown = True
        else:
            self._output += output_tokens
        if cost is None:
            self._cost_unknown = True
        else:
            self._cost += cost

    def finish(self) -> dict[str, Any]:
        return {
            "input_tokens": None if self._input_unknown else self._input,
            "output_tokens": None if self._output_unknown else self._output,
            "cost_usd": None if self._cost_unknown else self._cost,
            "request_count": 1,
            "http_attempts": self.http_attempts,
        }


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _hash_json(value: Any) -> str:
    try:
        return str(_shared_sha256_json(value))
    except Exception:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(data).hexdigest()


def _validation_error(message: str) -> Exception:
    return _BundleValidationError(message)


def _validate_frozen_input(frozen_input: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(frozen_input, Mapping):
        raise _validation_error("frozen incumbent input must be an object")
    # The shared validator checks exact record keys, ordering, bounds, and the
    # input hash.  Copy only after validation so callers cannot mutate a live
    # request while it is being sent.
    value = dict(frozen_input)
    try:
        checked = _shared_validate_input(value)
    except Exception:
        raise
    return {
        "system_prompt": checked["system_prompt"],
        "user_message": checked["user_message"],
        "records": [dict(record) for record in checked["records"]],
        "input_sha256": checked["input_sha256"],
        "ordered_ids": list(checked.get("ordered_ids", [record["id"] for record in checked["records"]])),
    }


def _endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _estimate_tokens(value: Any) -> int:
    try:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        body = str(value)
    return max(1, len(body.encode("utf-8")) if isinstance(body, str) else len(body))


def _cost(config: OpenAIChatConfig, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or config.input_cost_per_million is None:
        return None
    result = input_tokens * config.input_cost_per_million / 1_000_000.0
    if output_tokens is not None and config.output_cost_per_million is not None:
        result += output_tokens * config.output_cost_per_million / 1_000_000.0
    elif config.output_cost_per_million is not None:
        return None
    return result


def _usage(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("usage"), Mapping):
        return dict(payload["usage"])
    return {}


def _request_id(response: Any, payload: Any = None) -> str | None:
    headers = getattr(response, "headers", {})
    if isinstance(headers, Mapping):
        for key in ("x-request-id", "request-id", "X-Request-Id", "Request-Id"):
            value = headers.get(key)
            if isinstance(value, str) and value:
                return value[:256]
    if isinstance(payload, Mapping) and isinstance(payload.get("id"), str):
        return payload["id"][:256]
    return None


def _content_text(message: Any) -> str | None:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        pieces: list[str] = []
        for part in message:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        return "".join(pieces) if pieces else None
    return None


def _parse_selection(text: str) -> tuple[list[str] | None, str | None]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    # Preserve the incumbent's tolerance for a short explanatory prefix while
    # requiring exactly one authoritative JSON object.  A second object is an
    # ambiguous answer and must never be silently ignored.
    candidates: list[Mapping[str, Any]] = []
    decoder = json.JSONDecoder()
    for start, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(candidate[start:])
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping) and "ai_article_ids" in value:
            candidates.append(value)
    if len(candidates) != 1:
        return None, "invalid_json" if not candidates else "ambiguous_json"
    selected = candidates[0].get("ai_article_ids")
    if not isinstance(selected, list) or any(not isinstance(item, str) or not item for item in selected):
        return None, "missing_or_invalid_ai_article_ids"
    return selected, None


def _map_returned_ids(raw_ids: Sequence[str], records: Sequence[Mapping[str, str]]) -> tuple[set[str], list[str], list[str]]:
    full_ids = [record["id"] for record in records]
    selected: set[str] = set()
    unknown: list[str] = []
    ambiguous: list[str] = []
    seen_raw: set[str] = set()
    for raw in raw_ids:
        if raw in seen_raw:
            ambiguous.append(raw[:256])
            continue
        seen_raw.add(raw)
        if raw in full_ids:
            selected.add(raw)
            continue
        # The production filter commonly returns the first 16 characters.  It
        # also historically accepted shorter/differently truncated prefixes;
        # retain that mapping, but never choose one of several matches.
        prefix = raw[:8]
        matches = [
            full
            for full in full_ids
            if full.startswith(prefix) or full[:16].startswith(raw) or raw.startswith(full[:16])
        ]
        matches = sorted(set(matches))
        if len(matches) == 1:
            if matches[0] in selected:
                ambiguous.append(raw[:256])
            else:
                selected.add(matches[0])
        elif len(matches) > 1:
            ambiguous.append(raw[:256])
        else:
            unknown.append(raw[:256])
    return selected, unknown, ambiguous


def _decision(
    record_id: str,
    *,
    semantic: str,
    effective_keep: bool,
    model: str | None,
    request_hash: str | None,
    fallback_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "decision": semantic,
        "effective_keep": bool(effective_keep),
        "probabilities": {"relevance": None, "evidence_sufficient": None},
        "fallback_reason": fallback_reason,
        "model": model,
        "request_hash": request_hash,
    }


def _superset(
    records: Sequence[Mapping[str, str]],
    *,
    reason: str,
    model: str | None,
    request_hash: str | None,
) -> list[dict[str, Any]]:
    return [
        _decision(
            record["id"],
            semantic="abstain",
            effective_keep=True,
            model=model,
            request_hash=request_hash,
            fallback_reason=reason,
        )
        for record in records
    ]


def _budget_reserve(budget: Any, *, input_tokens: int, output_tokens: int, cost_usd: float | None) -> Any:
    if budget is None:
        return None
    return budget.reserve(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd)


def _budget_settle(
    budget: Any,
    reservation: Any,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
) -> None:
    if budget is None or reservation is None:
        return
    budget.settle(
        reservation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


class IncumbentAdapter:
    """Replay one frozen incumbent relevance request through chat completions."""

    def __init__(
        self,
        config: OpenAIChatConfig | None = None,
        api_key: str | None = None,
        *,
        http_client: Any | None = None,
        transport: Any | None = None,
        sleep: Callable[[float], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if http_client is not None and transport is not None:
            raise ValueError("supply only one injected incumbent HTTP client")
        self._http_client = http_client if http_client is not None else transport
        self._config = config
        self._api_key = api_key
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic

    async def evaluate(
        self,
        frozen_input: Mapping[str, Any],
        *,
        config: OpenAIChatConfig | None = None,
        budget: Any | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        frozen = _validate_frozen_input(frozen_input)
        config = config or self._config or OpenAIChatConfig()
        secret = api_key if api_key is not None else self._api_key
        if secret is None:
            secret = config.api_key
        records = frozen["records"]
        request_body = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": frozen["system_prompt"]},
                {"role": "user", "content": frozen["user_message"]},
            ],
            "max_tokens": config.max_output_tokens,
            "stream": False,
        }
        if config.reasoning_effort is not None:
            request_body["reasoning"] = {"effort": config.reasoning_effort}
        if urlsplit(config.base_url).hostname == "api.rdsec.trendmicro.com":
            # LiteLLM gateway controls: identical frozen prompts must receive
            # fresh responses when measuring control repeatability.
            request_body["cache"] = {"no-cache": True, "no-store": True}
        request_hash = _hash_json(request_body)
        empty_result = {
            "adapter": "incumbent",
            "status": "complete",
            "model": config.model or None,
            "requested_model": config.model or None,
            "actual_models": [],
            "input_sha256": frozen["input_sha256"],
            "decisions": [],
            "requests": [],
            "degradations": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "request_count": 0, "http_attempts": 0},
        }
        if not records:
            return empty_result
        if not isinstance(secret, str) or not secret.strip():
            return self._result(
                records,
                frozen=frozen,
                config=config,
                status="unavailable",
                decisions=_superset(records, reason="missing_api_key", model=None, request_hash=request_hash),
                request=None,
                degradations=["missing_api_key"],
                usage={"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "request_count": 0, "http_attempts": 0},
                actual_models=[],
            )

        gate = _AttemptGate(config.max_http_attempts)
        if self._http_client is not None:
            outcome = await self._request(
                self._http_client,
                records=records,
                frozen=frozen,
                body=request_body,
                request_hash=request_hash,
                config=config,
                api_key=secret,
                budget=budget,
                gate=gate,
            )
        else:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                outcome = await self._request(
                    client,
                    records=records,
                    frozen=frozen,
                    body=request_body,
                    request_hash=request_hash,
                    config=config,
                    api_key=secret,
                    budget=budget,
                    gate=gate,
                )
        return self._result(
            records,
            frozen=frozen,
            config=config,
            status=outcome["status"],
            decisions=outcome["decisions"],
            request=outcome["request"],
            degradations=outcome["degradations"],
            usage=outcome["usage"],
            actual_models=outcome["actual_models"],
        )

    @staticmethod
    def _result(
        records: Sequence[Mapping[str, str]],
        *,
        frozen: Mapping[str, Any],
        config: OpenAIChatConfig,
        status: str,
        decisions: list[dict[str, Any]],
        request: dict[str, Any] | None,
        degradations: list[str],
        usage: Mapping[str, Any],
        actual_models: list[str],
    ) -> dict[str, Any]:
        return {
            "adapter": "incumbent",
            "status": status,
            "model": config.model or None,
            "requested_model": config.model or None,
            "actual_models": actual_models,
            "input_sha256": frozen["input_sha256"],
            "decisions": decisions,
            "requests": [] if request is None else [request],
            "degradations": degradations,
            "usage": dict(usage),
        }

    async def _request(
        self,
        client: Any,
        *,
        records: Sequence[Mapping[str, str]],
        frozen: Mapping[str, Any],
        body: Mapping[str, Any],
        request_hash: str,
        config: OpenAIChatConfig,
        api_key: str,
        budget: Any,
        gate: _AttemptGate,
    ) -> dict[str, Any]:
        endpoint = _endpoint(config.base_url)
        estimate_input = _estimate_tokens(body)
        estimate_output = config.max_output_tokens
        estimate_cost = _cost(config, estimate_input, estimate_output)
        diagnostic: dict[str, Any] = {
            "request_hash": request_hash,
            "input_sha256": frozen["input_sha256"],
            "endpoint": endpoint,
            "requested_model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "cache_policy": body.get("cache"),
            "returned_model": None,
            "request_id": None,
            "attempts": [],
            "status": "pending",
            "terminal": False,
            "latency_ms": None,
            "elapsed_ms": None,
            "raw_selected_ids": [],
            "mapped_selected_ids": [],
            "unknown_ids": [],
            "ambiguous_ids": [],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        usage_accumulator = _UsageAccumulator()
        degradations: list[str] = []
        started = self._clock()
        last_error: str | None = None

        for attempt in range(1, config.max_attempts + 1):
            if not await gate.take():
                diagnostic["status"] = "budget_exhausted"
                diagnostic["error"] = "max_http_attempts"
                decisions = _superset(records, reason="abstain_error", model=None, request_hash=request_hash)
                return {
                    "status": "incomplete",
                    "decisions": decisions,
                    "request": diagnostic,
                    "degradations": ["max_http_attempts"],
                    "usage": usage_accumulator.finish(),
                    "actual_models": [],
                }
            reservation = None
            try:
                reservation = _budget_reserve(
                    budget,
                    input_tokens=estimate_input,
                    output_tokens=estimate_output,
                    cost_usd=estimate_cost,
                )
            except _BudgetExceeded:
                diagnostic["status"] = "budget_exhausted"
                diagnostic["error"] = "shared_budget_exhausted"
                return {
                    "status": "incomplete",
                    "decisions": _superset(records, reason="budget_exhausted", model=None, request_hash=request_hash),
                    "request": diagnostic,
                    "degradations": ["shared_budget_exhausted"],
                    "usage": usage_accumulator.finish(),
                    "actual_models": [],
                }
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "status": "started",
                "http_status": None,
                "request_id": None,
                "usage": {"input_tokens": None, "output_tokens": None},
                "cost_usd": None,
                "elapsed_ms": None,
                "latency_ms": None,
                "error": None,
            }
            response: Any = None
            payload: Any = None
            attempt_started = self._clock()
            progress = Progress(role="incumbent", stage="filter", request_index=1,
                                attempt=attempt, item_count=len(records), model=config.model).start()
            try:
                timeout = config.timeout_seconds
                if budget is not None:
                    remaining = getattr(budget, "remaining_seconds", None)
                    if isinstance(remaining, (int, float)):
                        timeout = min(timeout, max(0.001, float(remaining)))
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
                status_code = getattr(response, "status_code", None)
                attempt_record["http_status"] = status_code
                attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                try:
                    payload = response.json()
                    parse_error = None
                except Exception as exc:
                    parse_error = f"invalid_json:{type(exc).__name__}"
                raw_usage = _usage(payload)
                input_tokens = _optional_nonnegative_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")))
                output_tokens = _optional_nonnegative_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens")))
                cost_usd = _cost(config, input_tokens, output_tokens)
                usage_accumulator.add(raw_usage, cost_usd)
                attempt_record["usage"] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
                attempt_record["cost_usd"] = cost_usd
                attempt_record["request_id"] = _request_id(response, payload)
                response_id = payload.get("id") if isinstance(payload, Mapping) else None
                attempt_record["response_id"] = response_id[:256] if isinstance(response_id, str) else None
                _budget_settle(
                    budget,
                    reservation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
                returned_model = payload.get("model") if isinstance(payload, Mapping) else None
                if not isinstance(status_code, int) or not 200 <= status_code < 300:
                    reason = f"http_{status_code}" if status_code is not None else "http_error"
                    attempt_record["status"] = "transport_error"
                    attempt_record["error"] = reason
                    diagnostic["attempts"].append(attempt_record)
                    if isinstance(status_code, int) and status_code in config.retry_statuses and attempt < config.max_attempts:
                        await self._backoff(attempt, response, config, budget)
                        continue
                    diagnostic["status"] = "transport_error"
                    diagnostic["error"] = reason
                    degradations.append(reason)
                    last_error = reason
                    break
                diagnostic["returned_model"] = returned_model if isinstance(returned_model, str) else None
                diagnostic["request_id"] = attempt_record["request_id"]
                if parse_error:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = parse_error
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = parse_error
                    degradations.append("invalid_response")
                    break
                if not isinstance(returned_model, str) or not returned_model:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "missing_returned_model"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "missing_returned_model"
                    degradations.append("missing_returned_model")
                    break
                if returned_model != config.model:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "unexpected_returned_model"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "unexpected_returned_model"
                    degradations.append("unexpected_returned_model")
                    break
                if not isinstance(payload.get("choices"), list) or not payload["choices"]:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "missing_choices"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "missing_choices"
                    degradations.append("missing_choices")
                    break
                choice = payload["choices"][0]
                if not isinstance(choice, Mapping):
                    terminal = None
                    message = None
                else:
                    terminal = choice.get("finish_reason")
                    message = choice.get("message")
                if terminal not in ("stop", "end_turn"):
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "missing_terminal_response"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "missing_terminal_response"
                    degradations.append("missing_terminal_response")
                    break
                content = _content_text(message.get("content") if isinstance(message, Mapping) else None)
                if content is None:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "missing_message_content"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "missing_message_content"
                    degradations.append("missing_message_content")
                    break
                raw_ids, parse_error = _parse_selection(content)
                if parse_error or raw_ids is None:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = parse_error or "invalid_selection"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = attempt_record["error"]
                    degradations.append(attempt_record["error"])
                    break
                selected, unknown, ambiguous = _map_returned_ids(raw_ids, records)
                diagnostic["raw_selected_ids"] = [value[:256] for value in raw_ids[:1024]]
                diagnostic["mapped_selected_ids"] = sorted(selected)
                diagnostic["unknown_ids"] = unknown[:1024]
                diagnostic["ambiguous_ids"] = ambiguous[:1024]
                if ambiguous:
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = "ambiguous_ids"
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = "ambiguous_ids"
                    degradations.append("ambiguous_ids")
                    break
                if unknown:
                    error = "unknown_ids"
                    attempt_record["status"] = "invalid_response"
                    attempt_record["error"] = error
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "invalid_response"
                    diagnostic["error"] = error
                    degradations.append(error)
                    break
                decisions = [
                    _decision(
                        record["id"],
                        semantic="keep" if record["id"] in selected else "reject",
                        effective_keep=record["id"] in selected,
                        model=returned_model,
                        request_hash=request_hash,
                        fallback_reason=None,
                    )
                    for record in records
                ]
                attempt_record["status"] = "complete"
                diagnostic["attempts"].append(attempt_record)
                diagnostic["status"] = "complete"
                diagnostic["terminal"] = True
                diagnostic["latency_ms"] = round((self._clock() - started) * 1000.0, 3)
                diagnostic["elapsed_ms"] = diagnostic["latency_ms"]
                return {
                    "status": "degraded" if degradations else "complete",
                    "decisions": decisions,
                    "request": diagnostic,
                    "degradations": degradations,
                    "usage": usage_accumulator.finish(),
                    "actual_models": [returned_model],
                }
            except _BudgetExceeded:
                try:
                    _budget_settle(
                        budget,
                        reservation,
                        input_tokens=None,
                        output_tokens=None,
                        cost_usd=None,
                    )
                except Exception:
                    pass
                attempt_record["status"] = "budget_exhausted"
                attempt_record["error"] = "shared_budget_exhausted"
                attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                diagnostic["attempts"].append(attempt_record)
                diagnostic["status"] = "budget_exhausted"
                diagnostic["error"] = "shared_budget_exhausted"
                return {
                    "status": "incomplete",
                    "decisions": _superset(records, reason="budget_exhausted", model=None, request_hash=request_hash),
                    "request": diagnostic,
                    "degradations": ["shared_budget_exhausted"],
                    "usage": usage_accumulator.finish(),
                    "actual_models": [],
                }
            except (httpx.TimeoutException, httpx.TransportError, OSError) as exc:
                try:
                    _budget_settle(
                        budget,
                        reservation,
                        input_tokens=None,
                        output_tokens=None,
                        cost_usd=None,
                    )
                except Exception:
                    pass
                usage_accumulator.add({}, None)
                attempt_record["status"] = "transport_error"
                attempt_record["error"] = type(exc).__name__
                attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                diagnostic["attempts"].append(attempt_record)
                last_error = type(exc).__name__
                if attempt < config.max_attempts:
                    await self._backoff(attempt, response, config, budget)
                    continue
                degradations.append(last_error)
                break
            except Exception as exc:
                # Provider/schema/client errors are terminal for this frozen
                # request.  Keep only the exception type in diagnostics so a
                # secret-bearing transport message cannot reach artifacts.
                try:
                    _budget_settle(
                        budget,
                        reservation,
                        input_tokens=None,
                        output_tokens=None,
                        cost_usd=None,
                    )
                except Exception:
                    pass
                usage_accumulator.add({}, None)
                reason = type(exc).__name__
                attempt_record["status"] = "error"
                attempt_record["error"] = reason
                attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                diagnostic["attempts"].append(attempt_record)
                diagnostic["status"] = "error"
                diagnostic["error"] = reason
                degradations.append(reason)
                break
            finally:
                progress.finish(status="incomplete" if attempt_record["status"] == "started" else attempt_record["status"],
                                input_tokens=attempt_record["usage"].get("input_tokens"),
                                output_tokens=attempt_record["usage"].get("output_tokens"))

        diagnostic["status"] = diagnostic.get("status") if diagnostic.get("status") != "pending" else "transport_error"
        diagnostic["error"] = diagnostic.get("error") or last_error or "retry_exhausted"
        diagnostic["latency_ms"] = round((self._clock() - started) * 1000.0, 3)
        diagnostic["elapsed_ms"] = diagnostic["latency_ms"]
        return {
            "status": "incomplete" if diagnostic["status"] == "budget_exhausted" else "degraded",
            "decisions": _superset(records, reason="abstain_error", model=None, request_hash=request_hash),
            "request": diagnostic,
            "degradations": degradations or [diagnostic["error"]],
            "usage": usage_accumulator.finish(),
            "actual_models": [],
        }

    async def _backoff(self, attempt: int, response: Any, config: OpenAIChatConfig, budget: Any | None) -> None:
        delay: float | None = None
        headers = getattr(response, "headers", {})
        if isinstance(headers, Mapping):
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                try:
                    delay = max(0.0, float(raw))
                except (TypeError, ValueError):
                    delay = None
        if delay is None:
            delay = min(config.backoff_initial_seconds * (2 ** max(0, attempt - 1)), config.backoff_max_seconds)
        if budget is not None:
            remaining = getattr(budget, "remaining_seconds", None)
            if isinstance(remaining, (int, float)):
                delay = min(delay, max(0.0, float(remaining)))
        await self._sleep(delay)


__all__ = ["IncumbentAdapter", "OpenAIChatConfig"]
