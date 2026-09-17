"""Isolated scratch branches for the qualified, dependency-complete replay."""
from __future__ import annotations

import asyncio
import copy
import os
import subprocess
import sys
import time
from pathlib import Path

from .contracts import read_json, sha256_json, validate_decisions, write_json
from .runtime import ROOT, explicit_route, model_child_environment


def run_pipeline_pair(bundle: Path, destination: Path, control: dict, candidate: dict,
                      *, repeat_control=False, deadline: float | None = None) -> dict:
    branches = [("control", control), ("candidate", candidate)]
    if int(read_json(bundle / "manifest.json")["bundle_sha256"][:8], 16) % 2:
        branches.reverse()
    if repeat_control:
        branches.append(("control-repeat", read_json(destination / "control-repeat.json")))
    results = {}
    deadline = deadline if deadline is not None else time.monotonic() + 7100
    for name, decisions in branches:
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            results[name] = {"status": "unavailable", "reason": "experiment_deadline_exceeded"}
            continue
        branch = destination / name
        branch.mkdir(exist_ok=False)
        write_json(branch / "selection.json", decisions)
        started = time.monotonic()
        try:
            # No repository or source credentials, proxies, dotenv or production
            # settings cross this boundary. stderr/stdout stays in the internal
            # artifact, never committed to the public output tree.
            with (branch / "worker.log").open("x") as log:
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/shadow/pipeline_worker.py"),
                     "--bundle", str(bundle), "--branch", str(branch)], cwd=branch,
                    env=model_child_environment(), stdout=log, stderr=subprocess.STDOUT,
                    timeout=min(3600, remaining), check=False)
            result = read_json(branch / "result.json") if (branch / "result.json").is_file() else {
                "status": "failed", "reason": "worker_exited_without_result"}
            if completed.returncode != 0:
                result["status"] = "failed"
        except subprocess.TimeoutExpired:
            result = {"status": "unavailable", "reason": "branch_deadline_exceeded"}
        result["wall_seconds"] = time.monotonic() - started
        cost_path = branch / "cost-report.json"
        if cost_path.is_file():
            tracked = read_json(cost_path)
            result["usage"] = {"api_calls": tracked.get("api_calls"), "tokens": tracked.get("tokens"),
                               "tracker_cost_estimate": tracked.get("cost"), "cost_usd": None,
                               "cost_basis": "tracker estimate; route billing unverified"}
        else:
            result["usage"] = {"api_calls": None, "tokens": None, "cost_usd": None}
        results[name] = result
        write_json(branch / "result.json", result)
    return {"status": "complete" if all(row["status"] == "complete" for row in results.values()) else "unavailable",
            "branches": results, "unaffected_categories": "original_pre_continuity_reports",
            "branch_order": [name for name, _ in branches],
            "shared_stages": "rerun_for_each_branch", "hero": "original_reused",
            "reused_upstream_cost_excluded": True,
            "model_identity_basis": "configured_route; production client does not prove response.model"}


async def pipeline_worker(bundle: str, branch: str) -> dict:
    from .budget import BudgetLimits, RequestBudget
    from .network import model_egress_only
    from .rendering import keyword_filter, relevance_records
    from .replay_context import ReplayContext, ReplayIntegrityError

    branch = Path(branch).resolve()
    context = ReplayContext(bundle, branch, required_capability="pipeline_replay")
    route = explicit_route(require_credentials=True)
    inputs = context.read_filter_input()
    current_records = relevance_records(keyword_filter(context.read_gathered("news")))
    if sha256_json(current_records) != inputs["input_sha256"]:
        raise ReplayIntegrityError("Current keyword/rendering input differs from captured evidence")
    decision = read_json(branch / "selection.json")
    validate_decisions(inputs["records"], decision["decisions"])
    if decision.get("input_sha256") != inputs["input_sha256"]:
        raise ReplayIntegrityError("Branch selection belongs to another input")
    context.precomputed_exact_kept_ids = [row["id"] for row in decision["decisions"] if row["effective_keep"]]
    context.rerun_categories = {"news"}

    # Materialise only dependencies named in the verified bundle inventory.
    config_dir = branch / "config"
    web_dir = branch / "web"
    data_dir = branch / "data"
    for directory in (config_dir, web_dir, data_dir):
        directory.mkdir(exist_ok=False)
    hero = context.read_bytes("original/hero.webp", required=False)
    if hero is not None:
        hero_dir = web_dir / "data" / context.frozen_report_date
        hero_dir.mkdir(parents=True)
        (hero_dir / "hero.webp").write_bytes(hero)
    (config_dir / "model_releases.yaml").write_bytes(context.read_model_releases())
    (config_dir / "prompts.yaml").write_bytes(context.read_bytes("context/prompts.yaml"))
    for relative in context.history_files():
        if not relative.startswith("history/"):
            raise ReplayIntegrityError("Unexpected frozen history path")
        destination = web_dir / "data" / relative.removeprefix("history/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(context.read_bytes(relative))
    # Historical reports supply prior context to the JSON/feed/search generators.
    # The original same-day outputs are not seeded into either new branch.
    frozen_runtime = context.read_json("context/runtime-settings.json")
    from .capture import RUNTIME_ENV_KEYS
    if not isinstance(frozen_runtime, dict) or set(frozen_runtime) - set(RUNTIME_ENV_KEYS):
        raise ReplayIntegrityError("Invalid frozen runtime environment")
    for key, value in frozen_runtime.items():
        if value is not None:
            if not isinstance(value, str):
                raise ReplayIntegrityError("Frozen runtime settings must be strings or null")
            os.environ[key] = value
    # Retries are additionally bounded across all calls by the egress budget.
    os.environ.update(NEWS_SHADOW_SKIP_DOTENV="1", LLM_TRUST_ENV_PROXY="false",
                      LLM_METRICS_PATH=str(data_dir / "llm_metrics.jsonl"), LLM_REPLAY_CAPTURE="false")
    captured_concurrency = int(os.environ.get("LLM_MAX_CONCURRENT_REQUESTS", "8"))
    os.environ["LLM_MAX_CONCURRENT_REQUESTS"] = str(min(4, captured_concurrency) if captured_concurrency > 0 else 4)
    effective = context.read_effective_config()
    values = effective.get("values", effective)
    llm = copy.deepcopy(values.get("llm") or {})
    if not llm:
        raise ReplayIntegrityError("Missing frozen LLM settings")
    # No remote configuration may choose a destination or credential. The route
    # is explicitly provided by the operator and recorded in experiment identity.
    llm.pop("routes", None)
    llm.update(mode="openai-chat", base_url=route["base_url"].removesuffix("/v1"),
               model=route["model"], api_key=os.environ.get("SHADOW_INCUMBENT_API_KEY") or os.environ["RDSEC_API_KEY"])
    budget = RequestBudget(BudgetLimits(max_requests=64, max_input_tokens=8_000_000,
                                       max_output_tokens=4_000_000, deadline_seconds=3500))
    outcome = {"status": "running"}
    tracker = None
    try:
        with model_egress_only([route["base_url"] + "/chat/completions"], budget=budget):
            # Imports occur only after the process guard is installed. Production
            # entrypoint dotenv loading is disabled before importing the helper.
            from agents.config import ProviderConfig, PromptAccessor, load_prompts
            from agents.orchestrator import MainOrchestrator
            from agents.cost_tracker import get_tracker
            from run_pipeline import generate_pipeline_outputs
            tracker = get_tracker()

            config = ProviderConfig.model_validate({"llm": llm, "pipeline": values.get("pipeline") or {}})
            prompts = PromptAccessor(load_prompts(str(config_dir)))
            orchestrator = MainOrchestrator(config_dir=str(config_dir), data_dir=str(data_dir),
                web_dir=str(web_dir), target_date=context.frozen_report_date,
                provider_config=config, prompt_accessor=prompts, replay_context=context)
            try:
                result = await orchestrator.run()
                generate_pipeline_outputs(result, str(web_dir), config, orchestrator=orchestrator)
                write_json(branch / "orchestrator-result.json", result.to_dict())
                write_json(branch / "cost-report.json", get_tracker().get_json_report())
                failures = [phase.get("name", phase.get("phase", "unknown"))
                            for phase in (result.to_dict().get("phase_status") or [])
                            if isinstance(phase, dict) and phase.get("status") in {"failed", "partial"}]
                outcome = {"status": "complete" if not failures and not result.degradations else "degraded",
                           "failed_phases": failures, "degradations": result.degradations,
                           "date": result.date, "configured_model": route["model"]}
            finally:
                for client in (getattr(orchestrator, "async_client", None), getattr(orchestrator, "llm_client", None)):
                    close = getattr(client, "aclose", None) or getattr(client, "close", None)
                    if close:
                        closed = close()
                        if hasattr(closed, "__await__"):
                            await closed
    except Exception as exc:
        outcome = {"status": "unavailable", "failure_type": type(exc).__name__}
    finally:
        if tracker is not None:
            write_json(branch / "cost-report.json", tracker.get_json_report())
        outcome["budget"] = budget.snapshot()
        write_json(branch / "result.json", outcome)
    return outcome
