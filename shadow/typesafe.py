"""Bounded TypeSafe relevance adapter used by the news shadow experiment.

The adapter deliberately owns only the transport and the translation between
the TypeSafe response and the shadow decision contract.  It does not read
environment variables, construct a production LLM client, or call any source
of article data.  Callers supply the frozen records, the policy, credentials,
and (when running under the experiment runner) the shared request budget.

TypeSafe's API returns a Noul probability directly.  Noul answers do not carry
the separate ``confidence`` field used by Choice and Score answers; this module
therefore preserves only the two probabilities returned by the service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .contracts import (
    BundleValidationError as _BundleValidationError,
    sha256_json as _shared_sha256_json,
    validate_records as _shared_validate_records,
)
from .budget import BudgetExceeded as _BudgetExceeded


DEFAULT_TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_TYPESAFE_MODEL = "jev-1.13.0"
TYPESAFE_INPUT_USD_PER_MILLION = 0.042


@dataclass(frozen=True)
class TypeSafeConfig:
    """Explicit, secret-free transport settings for the Jev adapter.

    ``api_key`` intentionally is not a config field.  The evaluator must pass
    it to :meth:`TypeSafeAdapter.evaluate` so it cannot accidentally be
    serialized in an effective-config or diagnostics record.
    """

    endpoint: str = DEFAULT_TYPESAFE_ENDPOINT
    model: str = DEFAULT_TYPESAFE_MODEL
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    max_http_attempts: int = 64
    batch_size: int = 16
    max_concurrency: int = 2
    backoff_initial_seconds: float = 0.5
    backoff_max_seconds: float = 5.0
    output_token_estimate: int = 256
    input_cost_per_million: float | None = TYPESAFE_INPUT_USD_PER_MILLION
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({408, 429, 500, 502, 503, 504, 529})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.strip():
            raise ValueError("TypeSafe endpoint must be an explicit non-empty URL")
        parsed = urlsplit(self.endpoint)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or
                parsed.password or parsed.query or parsed.fragment):
            raise ValueError("TypeSafe endpoint must be an HTTPS URL without URL credentials or metadata")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("TypeSafe model must be an explicit non-empty ID")
        if self.timeout_seconds <= 0:
            raise ValueError("TypeSafe timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("TypeSafe max_attempts must be at least one")
        if self.max_http_attempts < 1:
            raise ValueError("TypeSafe max_http_attempts must be at least one")
        if self.batch_size < 1:
            raise ValueError("TypeSafe batch_size must be positive")
        if self.max_concurrency < 1:
            raise ValueError("TypeSafe max_concurrency must be positive")
        if self.backoff_initial_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("TypeSafe retry backoff cannot be negative")
        if self.output_token_estimate < 1:
            raise ValueError("TypeSafe output_token_estimate must be positive")
        if self.input_cost_per_million is not None and self.input_cost_per_million < 0:
            raise ValueError("TypeSafe input cost cannot be negative")

    def public_dict(self) -> dict[str, Any]:
        """Return a diagnostics-safe representation with no credential field."""

        result = asdict(self)
        result["retry_statuses"] = sorted(self.retry_statuses)
        return result

    def to_public_dict(self) -> dict[str, Any]:
        """Compatibility name used by the other shadow client configs."""

        return self.public_dict()


DEFAULT_POLICY: dict[str, Any] = {
    "version": "news-relevance-v1-dev",
    "rubric_version": "frontier-news-v1",
    "reject_max": 0.10,
    "keep_min": 0.80,
    "sufficiency_min": 0.90,
    "batch_size": 16,
}


class _AttemptGate:
    """Bound attempts local to one adapter invocation.

    The shared ``RequestBudget`` accounts tokens and spend.  This small gate
    only enforces the adapter's maximum number of HTTP attempts and therefore
    does not duplicate the shared budget's accounting.
    """

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
    def __init__(self, *, no_requests: bool = False):
        self.request_count = 0
        self.http_attempts = 0
        self._input = 0
        self._output = 0
        self._cost = 0.0
        self._input_unknown = False
        self._output_unknown = False
        self._cost_unknown = False
        self._no_requests = no_requests

    def add_attempt(self, usage: Mapping[str, Any] | None, cost_usd: float | None) -> None:
        self.http_attempts += 1
        usage = usage or {}
        input_tokens = _optional_nonnegative_int(
            usage.get("input_tokens", usage.get("prompt_tokens"))
        )
        output_tokens = _optional_nonnegative_int(
            usage.get("output_tokens", usage.get("completion_tokens"))
        )
        if input_tokens is None:
            self._input_unknown = True
        else:
            self._input += input_tokens
        if output_tokens is None:
            self._output_unknown = True
        else:
            self._output += output_tokens
        if cost_usd is None:
            self._cost_unknown = True
        else:
            self._cost += cost_usd

    def add_request(self) -> None:
        self.request_count += 1

    def finish(self) -> dict[str, Any]:
        # A no-call empty input has an exact zero usage.  If a call happened but
        # omitted usage, None preserves the distinction between unknown and 0.
        if self._no_requests and self.http_attempts == 0:
            input_tokens: int | None = 0
            output_tokens: int | None = 0
            cost_usd: float | None = 0.0
        else:
            input_tokens = None if self._input_unknown else self._input
            output_tokens = None if self._output_unknown else self._output
            cost_usd = None if self._cost_unknown else self._cost
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "request_count": self.request_count,
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


def _finite_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        return None
    return value


def _hash_json(value: Any) -> str:
    try:
        return str(_shared_sha256_json(value))
    except Exception:
        # The fallback is only a guard for a malformed injected helper.  The
        # canonical shared helper is used in normal shadow runs.
        data = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


def _normalise_text(value: Any) -> str:
    """Apply the same basic source-text hygiene used by the incumbent renderer."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    # Remove bidi/zero-width controls and C0/C1 controls while retaining tabs
    # and newlines as useful article evidence.
    text = "".join(
        char
        for char in text
        if char not in "\u200b\u200c\u200d\u2060\ufeff"
        and not (0x202A <= ord(char) <= 0x202E)
        and not (0x2066 <= ord(char) <= 0x2069)
        and not (0 <= ord(char) < 32 and char not in "\t\n")
        and not (0x7F <= ord(char) <= 0x9F)
    )
    return text


def _clip(value: Any, maximum: int) -> str:
    text = _normalise_text(value)
    return text[:maximum] + "..." if len(text) > maximum else text


def _validation_error(message: str) -> Exception:
    return _BundleValidationError(message)


def _validate_records(records: Any) -> list[dict[str, str]]:
    # The renderer owns normalization and clipping.  The adapter must send the
    # frozen bytes exactly as captured, including the incumbent's explicit
    # trailing snippet ellipsis, rather than silently producing a new evidence
    # set.  Shared validation also rejects extra fields, duplicate IDs, and
    # records outside the reviewed bounds before any paid request.
    validated = _shared_validate_records(records)
    return [dict(record) for record in validated]


def _normalise_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        policy = {}
    if not isinstance(policy, Mapping):
        raise _validation_error("relevance policy must be an object")
    result = dict(DEFAULT_POLICY)
    result.update(dict(policy))
    if "schema_version" in result and result["schema_version"] != "news-shadow-policy/v1":
        raise _validation_error("unsupported relevance policy schema")
    version = result.get("version", result.get("policy_version"))
    if not isinstance(version, str) or not version.strip():
        raise _validation_error("relevance policy requires a non-empty version")
    result["version"] = version
    result["rubric_version"] = str(result.get("rubric_version", "frontier-news-v1"))
    if "model" in result and result["model"] is not None:
        if not isinstance(result["model"], str) or not result["model"].strip():
            raise _validation_error("relevance policy model must be a non-empty ID")
    for name in ("reject_max", "keep_min", "sufficiency_min"):
        value = _finite_probability(result.get(name))
        if value is None:
            raise _validation_error(f"relevance policy {name} must be a finite probability")
        result[name] = value
    if not (result["reject_max"] < result["keep_min"]):
        raise _validation_error("relevance policy requires reject_max < keep_min")
    batch_size = result.get("chunk_size", result.get("batch_size", DEFAULT_POLICY["batch_size"]))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise _validation_error("relevance policy batch_size must be a positive integer")
    result["batch_size"] = batch_size
    if result.get("fallback", "retain") != "retain":
        raise _validation_error("TypeSafe relevance policy must retain abstentions")
    if "questions_per_article" in result and result["questions_per_article"] != 2:
        raise _validation_error("TypeSafe relevance policy requires exactly two questions per article")
    for key in ("concurrency", "max_attempts"):
        if key in result:
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise _validation_error(f"relevance policy {key} must be a positive integer")
    if "timeout_seconds" in result:
        timeout = result["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise _validation_error("relevance policy timeout_seconds must be positive")
    return result


def _estimate_tokens(value: Any) -> int:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        encoded = str(value)
    # This is a reservation estimate only.  The settled provider usage remains
    # authoritative when it is present.
    # UTF-8 bytes provide a conservative reservation for byte-backed model
    # tokenizers. Reconcile to measured usage only when it is reported.
    return max(1, len(encoded.encode("utf-8")) if isinstance(encoded, str) else len(encoded))


def _cost_for(config: TypeSafeConfig, input_tokens: int | None) -> float | None:
    if input_tokens is None or config.input_cost_per_million is None:
        return None
    return input_tokens * config.input_cost_per_million / 1_000_000.0


def _extract_usage(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("usage")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _response_headers(response: Any) -> Mapping[str, Any]:
    headers = getattr(response, "headers", None)
    return headers if isinstance(headers, Mapping) else {}


def _request_id(response: Any, payload: Any = None) -> str | None:
    headers = _response_headers(response)
    for key in ("x-request-id", "request-id", "x-correlation-id"):
        value = headers.get(key) or headers.get(key.title())
        if isinstance(value, str) and value:
            return value[:256]
    if isinstance(payload, Mapping):
        value = payload.get("request_id")
        if isinstance(value, str) and value:
            return value[:256]
    return None


def _response_json(response: Any) -> tuple[Any, str | None]:
    try:
        payload = response.json()
    except Exception as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    return payload, None


def _question_ids(index: int) -> tuple[str, str]:
    # Stable IDs are code-only keys.  Their meaning lives in the full
    # instructions below, as required by the TypeSafe question contract.
    return f"r_{index:04d}", f"s_{index:04d}"


def _questions_for(records: Sequence[Mapping[str, str]], policy: Mapping[str, Any]) -> dict[str, Any]:
    relevance_instruction = policy.get(
        "relevance_instructions",
        "Does this article belong in a frontier AI news newsletter? Evaluate only the article evidence in the state. Include model, company, product, research, safety, policy, infrastructure, controversy, and negative AI news. Exclude unrelated general technology, routine non-AI news, and generic marketing commentary.",
    )
    sufficiency_instruction = policy.get(
        "sufficiency_instructions",
        "Does the bounded title, source, and snippet for this article provide enough evidence to make the frontier-AI relevance judgment without guessing or importing facts not present in the state?",
    )
    relevance_true = policy.get(
        "relevance_true",
        "The article is substantively about frontier artificial intelligence within the stated scope, including important safety, policy, infrastructure, or negative developments.",
    )
    relevance_false = policy.get(
        "relevance_false",
        "The article is unrelated, generic, routine, or lacks a substantive frontier-AI connection in the supplied evidence.",
    )
    sufficiency_true = policy.get(
        "sufficiency_true",
        "The supplied bounded evidence is enough to support a relevance decision without speculation.",
    )
    sufficiency_false = policy.get(
        "sufficiency_false",
        "The supplied evidence is too incomplete, ambiguous, or generic to support a relevance decision.",
    )
    questions: dict[str, Any] = {}
    for index, _record in enumerate(records):
        relevance_id, sufficiency_id = _question_ids(index)
        questions[relevance_id] = {
            "type": "noul",
            "instructions": f"For article array position {index}, {relevance_instruction}",
            "criteria": {"true": relevance_true, "false": relevance_false},
        }
        questions[sufficiency_id] = {
            "type": "noul",
            "instructions": f"For article array position {index}, {sufficiency_instruction}",
            "criteria": {"true": sufficiency_true, "false": sufficiency_false},
        }
    return questions


def _decision(
    record_id: str,
    *,
    semantic: str,
    effective_keep: bool,
    probabilities: Mapping[str, Any] | None,
    fallback_reason: str | None,
    model: str | None,
    request_hash: str | None,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "decision": semantic,
        "effective_keep": bool(effective_keep),
        "probabilities": {
            "relevance": (probabilities or {}).get("relevance"),
            "evidence_sufficient": (probabilities or {}).get("evidence_sufficient"),
        },
        "fallback_reason": fallback_reason,
        "model": model,
        "request_hash": request_hash,
        "policy_version": policy.get("version"),
        "rubric_version": policy.get("rubric_version"),
    }


def _abstain_decisions(
    records: Sequence[Mapping[str, str]],
    *,
    reason: str,
    model: str | None,
    request_hash: str | None,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _decision(
            record["id"],
            semantic="abstain",
            effective_keep=True,
            probabilities=None,
            fallback_reason=reason,
            model=model,
            request_hash=request_hash,
            policy=policy,
        )
        for record in records
    ]


def _normalise_usage_for_settlement(usage: Mapping[str, Any]) -> tuple[int | None, int | None]:
    return (
        _optional_nonnegative_int(usage.get("input_tokens", usage.get("prompt_tokens"))),
        _optional_nonnegative_int(usage.get("output_tokens", usage.get("completion_tokens"))),
    )


def _budget_reserve(
    budget: Any,
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None,
) -> Any:
    if budget is None:
        return None
    return budget.reserve(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


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


class TypeSafeAdapter:
    """Evaluate bounded article records with pinned Jev Noul questions."""

    def __init__(
        self,
        config: TypeSafeConfig | None = None,
        api_key: str | None = None,
        *,
        http_client: Any | None = None,
        transport: Any | None = None,
        sleep: Callable[[float], Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        # A caller-owned client lets the offline contract tests inject a fake
        # transport.  Real runs create one client per adapter invocation and
        # reuse its connection pool for all chunks and retries.
        if http_client is not None and transport is not None:
            raise ValueError("supply only one injected TypeSafe HTTP client")
        self._http_client = http_client if http_client is not None else transport
        self._config = config
        self._api_key = api_key
        self._sleep = sleep or asyncio.sleep
        self._clock = clock or time.monotonic

    async def evaluate(
        self,
        records: list[dict[str, Any]],
        *,
        policy: Mapping[str, Any] | None = None,
        config: TypeSafeConfig | None = None,
        api_key: str | None = None,
        budget: Any | None = None,
    ) -> dict[str, Any]:
        config = config or self._config or TypeSafeConfig()
        api_key = api_key if api_key is not None else self._api_key
        policy = _normalise_policy(policy)
        policy_model = policy.get("model")
        if policy_model is not None and policy_model != config.model:
            raise _validation_error(
                f"relevance policy model {policy_model!r} does not match configured model {config.model!r}"
            )
        validated = _validate_records(records)
        if not validated:
            return {
                "adapter": "typesafe",
                "status": "complete",
                "model": config.model,
                "requested_model": config.model,
                "actual_models": [],
                "input_sha256": _hash_json(validated),
                "policy_version": policy["version"],
                "decisions": [],
                "requests": [],
                "degradations": [],
                "usage": _UsageAccumulator(no_requests=True).finish(),
            }

        if not isinstance(api_key, str) or not api_key.strip():
            reason = "missing_api_key"
            decisions = _abstain_decisions(
                validated,
                reason=reason,
                model=None,
                request_hash=None,
                policy=policy,
            )
            return self._result(
                validated,
                config=config,
                policy=policy,
                decisions=decisions,
                requests=[],
                degradations=[reason],
                usage=_UsageAccumulator().finish(),
                actual_models=[],
                status="unavailable",
            )

        batch_size = min(config.batch_size, int(policy["batch_size"]))
        chunks = [validated[start : start + batch_size] for start in range(0, len(validated), batch_size)]
        max_attempts = min(config.max_attempts, int(policy.get("max_attempts", config.max_attempts)))
        timeout_seconds = min(
            config.timeout_seconds,
            float(policy.get("timeout_seconds", config.timeout_seconds)),
        )
        max_concurrency = min(config.max_concurrency, int(policy.get("concurrency", config.max_concurrency)))
        # Pass the effective policy limits without mutating the caller's frozen
        # config.  They are still included in the request diagnostics below.
        effective_config = TypeSafeConfig(
            endpoint=config.endpoint,
            model=config.model,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            max_http_attempts=config.max_http_attempts,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            backoff_initial_seconds=config.backoff_initial_seconds,
            backoff_max_seconds=config.backoff_max_seconds,
            output_token_estimate=config.output_token_estimate,
            input_cost_per_million=config.input_cost_per_million,
            retry_statuses=config.retry_statuses,
        )
        gate = _AttemptGate(effective_config.max_http_attempts)
        semaphore = asyncio.Semaphore(effective_config.max_concurrency)
        tasks = [
            asyncio.create_task(
                self._evaluate_chunk(
                    chunk,
                    chunk_index=index,
                    policy=policy,
                    config=effective_config,
                    api_key=api_key,
                    budget=budget,
                    gate=gate,
                    semaphore=semaphore,
                )
            )
            for index, chunk in enumerate(chunks)
        ]

        # Gather is intentional here: each chunk is isolated, and a transport
        # failure for one batch must retain only that batch while other batches
        # can finish under the same shared budget.
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda value: value[0])
        decisions: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        degradations: list[str] = []
        usage = _UsageAccumulator()
        actual_models: set[str] = set()
        any_error = False
        any_incomplete = False
        for _index, result in results:
            # ``asyncio.gather`` returns the tuple produced by
            # ``_evaluate_chunk``.  After unpacking, ``result`` is already
            # the chunk payload; indexing it by ``1`` treated the payload as
            # a sequence and raised ``KeyError: 1`` for every successful
            # response.  Keep the unpacking explicit so the per-chunk
            # accounting remains coupled to the same payload.
            decisions.extend(result["decisions"])
            requests.append(result["request"])
            usage.add_request()
            for attempt_usage in result["attempt_usages"]:
                usage.add_attempt(attempt_usage[0], attempt_usage[1])
            for degradation in result["degradations"]:
                if degradation not in degradations:
                    degradations.append(degradation)
            actual_models.update(
                model for model in result["actual_models"] if isinstance(model, str)
            )
            any_error = any_error or result["error"]
            any_incomplete = any_incomplete or result["incomplete"]

        status = "complete"
        if any_incomplete:
            status = "incomplete"
        elif any_error:
            status = "degraded"
        return self._result(
            validated,
            config=config,
            policy=policy,
            decisions=decisions,
            requests=requests,
            degradations=degradations,
            usage=usage.finish(),
            actual_models=sorted(actual_models),
            status=status,
        )

    def _result(
        self,
        records: Sequence[Mapping[str, str]],
        *,
        config: TypeSafeConfig,
        policy: Mapping[str, Any],
        decisions: list[dict[str, Any]],
        requests: list[dict[str, Any]],
        degradations: list[str],
        usage: Mapping[str, Any],
        actual_models: list[str],
        status: str,
    ) -> dict[str, Any]:
        return {
            "adapter": "typesafe",
            "status": status,
            "model": config.model,
            "requested_model": config.model,
            "actual_models": actual_models,
            "input_sha256": _hash_json(records),
            "policy_version": policy["version"],
            "decisions": decisions,
            "requests": requests,
            "degradations": degradations,
            "usage": dict(usage),
        }

    async def _evaluate_chunk(
        self,
        records: list[dict[str, str]],
        *,
        chunk_index: int,
        policy: Mapping[str, Any],
        config: TypeSafeConfig,
        api_key: str,
        budget: Any,
        gate: _AttemptGate,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, dict[str, Any]]:
        bounded_records = [dict(record) for record in records]
        body = {
            "state": {"articles": bounded_records},
            "model": config.model,
            "questions": _questions_for(bounded_records, policy),
        }
        request_hash = _hash_json(body)
        diagnostic: dict[str, Any] = {
            "request_hash": request_hash,
            "chunk_index": chunk_index,
            "article_ids": [record["id"] for record in records],
            "question_ids": sorted(body["questions"]),
            "requested_model": config.model,
            "returned_model": None,
            "request_id": None,
            "endpoint": config.endpoint,
            "attempts": [],
            "status": "pending",
            "terminal": False,
            "latency_ms": None,
            "elapsed_ms": None,
            "usage": {"input_tokens": None, "output_tokens": None, "cost_usd": None},
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        estimate_input = _estimate_tokens(body)
        estimate_output = config.output_token_estimate
        estimate_cost = _cost_for(config, estimate_input)
        attempt_usages: list[tuple[dict[str, Any], float | None]] = []
        degradations: list[str] = []
        started = self._clock()

        async def run_with_client(client: Any) -> dict[str, Any]:
            nonlocal diagnostic, attempt_usages, degradations
            for attempt in range(1, config.max_attempts + 1):
                if not await gate.take():
                    diagnostic["status"] = "budget_exhausted"
                    diagnostic["error"] = "max_http_attempts"
                    diagnostic["terminal"] = False
                    degradations.append("max_http_attempts")
                    return {
                        "decisions": _abstain_decisions(
                            records,
                            reason="abstain_error",
                            model=None,
                            request_hash=request_hash,
                            policy=policy,
                        ),
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [],
                        "error": True,
                        "incomplete": True,
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
                    diagnostic["terminal"] = False
                    degradations.append("shared_budget_exhausted")
                    return {
                        "decisions": _abstain_decisions(
                            records,
                            reason="budget_exhausted",
                            model=None,
                            request_hash=request_hash,
                            policy=policy,
                        ),
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [],
                        "error": True,
                        "incomplete": True,
                    }

                attempt_started = self._clock()
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
                payload: Any = None
                response: Any = None
                usage: dict[str, Any] = {}
                try:
                    request_timeout = config.timeout_seconds
                    if budget is not None:
                        remaining = getattr(budget, "remaining_seconds", None)
                        if isinstance(remaining, (int, float)):
                            request_timeout = min(request_timeout, max(0.001, float(remaining)))
                    async with semaphore:
                        response = await client.post(
                            config.endpoint,
                            headers=headers,
                            json=body,
                            timeout=request_timeout,
                        )
                    status_code = getattr(response, "status_code", None)
                    attempt_record["http_status"] = status_code
                    attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                    attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                    payload, parse_error = _response_json(response)
                    usage = _extract_usage(payload)
                    input_tokens, output_tokens = _normalise_usage_for_settlement(usage)
                    cost_usd = _cost_for(config, input_tokens)
                    attempt_usages.append((usage, cost_usd))
                    attempt_record["usage"] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                    attempt_record["cost_usd"] = cost_usd
                    attempt_record["request_id"] = _request_id(response, payload)
                    _budget_settle(
                        budget,
                        reservation,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
                    )
                    if not isinstance(status_code, int) or not 200 <= status_code < 300:
                        reason = f"http_{status_code}" if status_code is not None else "http_error"
                        attempt_record["status"] = "transport_error"
                        attempt_record["error"] = reason
                        if isinstance(status_code, int) and status_code in config.retry_statuses:
                            diagnostic["attempts"].append(attempt_record)
                            if attempt < config.max_attempts:
                                await self._backoff(attempt, response, config, budget)
                                continue
                        diagnostic["status"] = "transport_error"
                        diagnostic["error"] = reason
                        degradations.append(reason)
                        return {
                            "decisions": _abstain_decisions(
                                records,
                                reason="abstain_error",
                                model=None,
                                request_hash=request_hash,
                                policy=policy,
                            ),
                            "request": diagnostic,
                            "attempt_usages": attempt_usages,
                            "degradations": degradations,
                            "actual_models": [],
                            "error": True,
                            "incomplete": False,
                        }
                    diagnostic["returned_model"] = (
                        payload.get("model") if isinstance(payload, Mapping) else None
                    )
                    if parse_error:
                        attempt_record["status"] = "invalid_response"
                        attempt_record["error"] = parse_error
                        diagnostic["attempts"].append(attempt_record)
                        diagnostic["status"] = "invalid_response"
                        diagnostic["error"] = parse_error
                        degradations.append("invalid_response")
                        return {
                            "decisions": _abstain_decisions(
                                records,
                                reason="abstain_error",
                                model=None,
                                request_hash=request_hash,
                                policy=policy,
                            ),
                            "request": diagnostic,
                            "attempt_usages": attempt_usages,
                            "degradations": degradations,
                            "actual_models": [],
                            "error": True,
                            "incomplete": False,
                        }

                    validation = self._validate_payload(payload, body["questions"], config.model)
                    attempt_record["status"] = validation["status"]
                    attempt_record["error"] = validation.get("error")
                    diagnostic["attempts"].append(attempt_record)
                    if validation["status"] != "ok":
                        diagnostic["status"] = validation["status"]
                        diagnostic["error"] = validation.get("error")
                        diagnostic["terminal"] = bool(validation.get("terminal"))
                        degradations.append(validation.get("degradation", "invalid_response"))
                        return {
                            "decisions": _abstain_decisions(
                                records,
                                reason="abstain_error",
                                model=None,
                                request_hash=request_hash,
                                policy=policy,
                            ),
                            "request": diagnostic,
                            "attempt_usages": attempt_usages,
                            "degradations": degradations,
                            "actual_models": [],
                            "error": True,
                            "incomplete": False,
                        }

                    returned_model = validation["model"]
                    diagnostic["returned_model"] = returned_model
                    diagnostic["request_id"] = attempt_record["request_id"]
                    diagnostic["status"] = "complete"
                    diagnostic["terminal"] = True
                    diagnostic["usage"] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost_usd,
                    }
                    diagnostic["latency_ms"] = round((self._clock() - started) * 1000.0, 3)
                    decisions, anomalies = self._decisions_from_answers(
                        records,
                        payload["answers"],
                        request_hash=request_hash,
                        model=returned_model,
                        policy=policy,
                    )
                    degradations.extend(anomalies)
                    return {
                        "decisions": decisions,
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [returned_model],
                        "error": bool(anomalies),
                        "incomplete": False,
                    }
                except _BudgetExceeded:
                    # A settlement failure is a budget state, not a provider
                    # success.  Keep the request's known usage and retain the
                    # affected inputs.
                    if reservation is not None:
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
                    degradations.append("shared_budget_exhausted")
                    return {
                        "decisions": _abstain_decisions(
                            records,
                            reason="budget_exhausted",
                            model=None,
                            request_hash=request_hash,
                            policy=policy,
                        ),
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [],
                        "error": True,
                        "incomplete": True,
                    }
                except (httpx.TimeoutException, httpx.TransportError, OSError) as exc:
                    # No provider usage is knowable when transport fails before
                    # a response.  The reservation is settled as unknown.
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
                    reason = f"{type(exc).__name__}"
                    attempt_record["status"] = "transport_error"
                    attempt_record["error"] = reason
                    diagnostic["attempts"].append(attempt_record)
                    attempt_record["latency_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                    attempt_record["elapsed_ms"] = attempt_record["latency_ms"]
                    attempt_usages.append(({}, None))
                    if attempt < config.max_attempts:
                        await self._backoff(attempt, response, config, budget)
                        continue
                    diagnostic["status"] = "transport_error"
                    diagnostic["error"] = reason
                    degradations.append(reason)
                    return {
                        "decisions": _abstain_decisions(
                            records,
                            reason="abstain_error",
                            model=None,
                            request_hash=request_hash,
                            policy=policy,
                        ),
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [],
                        "error": True,
                        "incomplete": False,
                    }
                except Exception as exc:
                    # Schema/client failures are terminal for this frozen
                    # request.  Keep only the exception type in diagnostics so
                    # a secret-bearing transport message cannot reach an
                    # artifact.
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
                    reason = type(exc).__name__
                    attempt_usages.append(({}, None))
                    attempt_record["status"] = "error"
                    attempt_record["error"] = reason
                    attempt_record["elapsed_ms"] = round((self._clock() - attempt_started) * 1000.0, 3)
                    attempt_record["latency_ms"] = attempt_record["elapsed_ms"]
                    diagnostic["attempts"].append(attempt_record)
                    diagnostic["status"] = "error"
                    diagnostic["error"] = reason
                    degradations.append(reason)
                    return {
                        "decisions": _abstain_decisions(
                            records,
                            reason="abstain_error",
                            model=None,
                            request_hash=request_hash,
                            policy=policy,
                        ),
                        "request": diagnostic,
                        "attempt_usages": attempt_usages,
                        "degradations": degradations,
                        "actual_models": [],
                        "error": True,
                        "incomplete": False,
                    }

            # The loop always returns.  Keep an explicit guard for future
            # changes to retry policy so a missing branch cannot drop records.
            diagnostic["status"] = "transport_error"
            diagnostic["error"] = "retry_exhausted"
            degradations.append("retry_exhausted")
            return {
                "decisions": _abstain_decisions(
                    records,
                    reason="abstain_error",
                    model=None,
                    request_hash=request_hash,
                    policy=policy,
                ),
                "request": diagnostic,
                "attempt_usages": attempt_usages,
                "degradations": degradations,
                "actual_models": [],
                "error": True,
                "incomplete": False,
            }

        if self._http_client is not None:
            result = await run_with_client(self._http_client)
        else:
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
                result = await run_with_client(client)
        diagnostic["latency_ms"] = round((self._clock() - started) * 1000.0, 3)
        diagnostic["elapsed_ms"] = diagnostic["latency_ms"]
        return chunk_index, result

    async def _backoff(
        self,
        attempt: int,
        response: Any,
        config: TypeSafeConfig,
        budget: Any | None = None,
    ) -> None:
        retry_after: float | None = None
        headers = _response_headers(response)
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                retry_after = max(0.0, float(raw))
            except (TypeError, ValueError):
                retry_after = None
        delay = min(
            config.backoff_initial_seconds * (2 ** max(0, attempt - 1)),
            config.backoff_max_seconds,
        )
        sleep_for = retry_after if retry_after is not None else delay
        if budget is not None:
            remaining = getattr(budget, "remaining_seconds", None)
            if isinstance(remaining, (int, float)):
                sleep_for = min(sleep_for, max(0.0, float(remaining)))
        await self._sleep(sleep_for)

    @staticmethod
    def _validate_payload(
        payload: Any,
        questions: Mapping[str, Any],
        expected_model: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {"status": "invalid_response", "error": "response_not_object", "degradation": "invalid_response"}
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            return {"status": "invalid_response", "error": "missing_returned_model", "degradation": "missing_returned_model"}
        if model != expected_model:
            return {
                "status": "invalid_response",
                "error": "unexpected_returned_model",
                "degradation": "unexpected_returned_model",
            }
        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            return {"status": "invalid_response", "error": "missing_answers", "degradation": "missing_answers"}
        expected_ids = set(questions)
        unknown_ids = set(answers) - expected_ids
        if unknown_ids:
            return {
                "status": "invalid_response",
                "error": "unexpected_question_ids",
                "degradation": "unexpected_question_ids",
            }
        return {"status": "ok", "model": model, "answers": answers}

    @staticmethod
    def _answer_probability(answers: Mapping[str, Any], question_id: str) -> float | None:
        answer = answers.get(question_id)
        if not isinstance(answer, Mapping) or answer.get("type") != "noul":
            return None
        return _finite_probability(answer.get("noul"))

    def _decisions_from_answers(
        self,
        records: Sequence[Mapping[str, str]],
        answers: Mapping[str, Any],
        *,
        request_hash: str,
        model: str,
        policy: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        decisions: list[dict[str, Any]] = []
        anomalies: list[str] = []
        for index, record in enumerate(records):
            relevance_id, sufficiency_id = _question_ids(index)
            relevance = self._answer_probability(answers, relevance_id)
            sufficiency = self._answer_probability(answers, sufficiency_id)
            probabilities = {
                "relevance": relevance,
                "evidence_sufficient": sufficiency,
            }
            if relevance is None or sufficiency is None:
                anomalies.append(f"invalid_answer:{record['id']}")
                decisions.append(
                    _decision(
                        record["id"],
                        semantic="abstain",
                        effective_keep=True,
                        probabilities=probabilities,
                        fallback_reason="abstain_error",
                        model=model,
                        request_hash=request_hash,
                        policy=policy,
                    )
                )
                continue
            if sufficiency < policy["sufficiency_min"]:
                semantic = "abstain"
                effective_keep = True
            elif relevance <= policy["reject_max"]:
                semantic = "reject"
                effective_keep = False
            elif relevance >= policy["keep_min"]:
                semantic = "keep"
                effective_keep = True
            else:
                semantic = "abstain"
                effective_keep = True
            decisions.append(
                _decision(
                    record["id"],
                    semantic=semantic,
                    effective_keep=effective_keep,
                    probabilities=probabilities,
                    fallback_reason=None,
                    model=model,
                    request_hash=request_hash,
                    policy=policy,
                )
            )
        return decisions, anomalies


__all__ = ["TypeSafeAdapter", "TypeSafeConfig", "DEFAULT_TYPESAFE_ENDPOINT", "DEFAULT_TYPESAFE_MODEL"]
