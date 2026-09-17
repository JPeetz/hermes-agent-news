"""Optional explicit model-access probes using synthetic, non-source evidence."""
from __future__ import annotations

import asyncio
import os

from .budget import BudgetLimits, RequestBudget
from .contracts import sha256_json
from .runtime import explicit_route


async def probe_models() -> dict:
    from .incumbent import IncumbentAdapter, OpenAIChatConfig
    from .judge import JudgeClient, JudgeConfig
    from .network import model_egress_only
    from .typesafe import TypeSafeAdapter, TypeSafeConfig

    route = explicit_route(require_credentials=True)
    records = [{"id": "shadow_probe", "title": "Example lab announces a frontier AI model",
                "source": "Synthetic access probe", "snippet": "A fictional lab released a new large language model."}]
    frozen = {"records": records, "ordered_ids": ["shadow_probe"], "input_sha256": sha256_json(records),
              "system_prompt": 'This is an access probe. Return JSON only: {"ai_article_ids":["shadow_probe"]}.',
              "user_message": "Synthetic access probe; return the requested JSON."}
    budgets = {role: RequestBudget(BudgetLimits(max_requests=1, max_input_tokens=250000,
              max_output_tokens=100000, deadline_seconds=600)) for role in ("incumbent", "candidate", "judge")}
    config = OpenAIChatConfig(base_url=route["base_url"], model=route["model"], max_output_tokens=16384,
                             timeout_seconds=240, max_attempts=1, max_http_attempts=1)
    judge_config = JudgeConfig(api_key=os.environ["RDSEC_API_KEY"], max_attempts=1)
    budgets["judge"] = RequestBudget(BudgetLimits(max_requests=1, max_input_tokens=250000,
        max_output_tokens=judge_config.max_output_tokens, deadline_seconds=1800))
    with model_egress_only([TypeSafeConfig().endpoint, config.base_url + "/chat/completions", judge_config.endpoint]):
        incumbent = await IncumbentAdapter(config,
            os.environ.get("SHADOW_INCUMBENT_API_KEY") or os.environ["RDSEC_API_KEY"]).evaluate(frozen, budget=budgets["incumbent"])
        candidate = await TypeSafeAdapter(TypeSafeConfig(max_attempts=1, max_http_attempts=1),
            os.environ["TYPESAFE_API_KEY"]).evaluate(records, budget=budgets["candidate"])
        with JudgeClient(judge_config, budget=budgets["judge"]) as judge:
            adjudicated = judge.adjudicate_inputs(records)
    statuses = {"incumbent": incumbent["status"], "candidate": candidate["status"], "judge": adjudicated["status"]}
    return {"model_access_verified": all(status == "complete" for status in statuses.values()),
            "synthetic_data_only": True, "statuses": statuses,
            "actual_models": {"incumbent": incumbent.get("actual_models"), "candidate": candidate.get("actual_models")},
            # The adapters already restrict diagnostics to model identity,
            # accounting and safe error codes; prompts, responses and headers
            # are excluded. Preserve failures so hosted probes are actionable.
            "diagnostics": {role: {key: result.get(key, []) for key in ("requests", "errors", "degradations")}
                            for role, result in (("incumbent", incumbent), ("candidate", candidate),
                                                 ("judge", adjudicated))},
            "budget": {role: budget.snapshot() for role, budget in budgets.items()}}
