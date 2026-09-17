"""Join immutable inputs, bounded model adapters, and independent adjudication."""
from __future__ import annotations

import asyncio
import os
import json
import re
import time
from pathlib import Path

from .contracts import read_json, sha256_json, validate_decisions, write_json
from .runtime import ROOT, budget_for, inspect_run, validate_output_root
from .settings import JUDGE_MODEL, RDSEC_BASE, code_identity, experiment_identity, load_policy


def attempt_metrics(artifact: dict) -> list[dict]:
    """Normalise adapter diagnostics without discarding failed paid attempts."""
    result = []
    requests = artifact.get("requests", artifact.get("attempts", []))
    if not requests and isinstance(artifact.get("usage"), dict):
        requests = [{"usage": artifact["usage"], "status": "success",
                     "lineage": "final_captured_response_only; earlier_attempt_usage_unavailable"}]
    for request in requests:
        for attempt in request.get("attempts", [request]):
            usage = attempt.get("usage") or {}
            input_tokens = attempt.get("input_tokens", usage.get("input_tokens", usage.get("prompt_tokens")))
            output_tokens = attempt.get("output_tokens", usage.get("output_tokens", usage.get("completion_tokens")))
            cost = attempt.get("cost_usd")
            status = attempt.get("status", attempt.get("outcome"))
            result.append({**attempt, "input_tokens": input_tokens, "output_tokens": output_tokens,
                           "status": "success" if status in {"success", "ok", "complete"} else "failed",
                           "total_tokens": input_tokens + output_tokens if type(input_tokens) is int and type(output_tokens) is int else None,
                           "usage_known": type(input_tokens) is int and type(output_tokens) is int,
                           "cost_usd": cost, "cost_known": cost is not None})
    return result


def control_repeat_evidence(control: dict, repeated: dict) -> dict:
    """Require distinct model response IDs, not gateway request IDs, for drift evidence."""
    def response_ids(artifact):
        return {row["response_id"] for row in attempt_metrics(artifact)
                if row.get("status") == "success" and row.get("response_id")}
    first_ids, repeat_ids = response_ids(control), response_ids(repeated)
    shared = first_ids & repeat_ids
    verified = (control.get("status") == repeated.get("status") == "complete" and
                bool(first_ids) and bool(repeat_ids) and not shared)
    return {"distinct_response_ids": verified,
            "basis": "distinct_model_response_ids" if verified else
                     "reused_model_response_id" if shared else "missing_successful_response_identity",
            "shared_response_ids": sorted(shared)}


async def run_experiment(bundle, out, policy_path, *, mode="filter", cohort="engineering",
                         repeat_control=False, retry_version="1") -> dict:
    # Imports are deliberately below preflight in the CLI; acquisition never
    # imports production agent code or resolves a provider configuration.
    from .incumbent import IncumbentAdapter, OpenAIChatConfig
    from .judge import JudgeClient, JudgeConfig
    from .metrics import compare_decisions
    from .network import model_egress_only
    from .report import write_assessment
    from .typesafe import TypeSafeAdapter, TypeSafeConfig

    preflight = inspect_run(bundle, policy_path, mode=mode, cohort=cohort, require_credentials=True)
    manifest = preflight["manifest"]
    policy = load_policy(policy_path)
    bundle = Path(bundle).resolve()
    route = preflight["route"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", retry_version):
        raise ValueError("Invalid retry version")
    runtime = {"cohort": cohort, "retry_version": retry_version,
               "cohort_sha256": preflight["cohort_sha256"], "control": {key: route[key] for key in
               ("base_url", "model", "max_output_tokens", "max_attempts", "timeout_seconds")}}
    code = code_identity(ROOT)
    experiment_id, identity = experiment_identity(manifest, code, policy, mode=mode,
                                                   repeat_control=repeat_control, runtime=runtime)
    destination = validate_output_root(out, bundle) / experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "identity.json", identity)
    write_json(destination / "code.json", code)
    write_json(destination / "policy.json", policy)
    write_json(destination / "source-manifest.json", manifest)
    frozen = read_json(bundle / "relevance/input.json")
    records = frozen["records"]
    original = read_json(bundle / "relevance/incumbent-decision.json")
    write_json(destination / "input.json", frozen)
    write_json(destination / "original-decision.json", original)
    budgets = {role: budget_for(policy, role) for role in ("candidate", "incumbent", "judge")}
    candidate_config = TypeSafeConfig(model=policy["model"], batch_size=policy["chunk_size"],
        max_concurrency=policy["concurrency"], max_attempts=policy["max_attempts"],
        timeout_seconds=policy["timeout_seconds"], max_http_attempts=budgets["candidate"].limits.max_requests)
    control_config = OpenAIChatConfig(base_url=route["base_url"], model=route["model"],
        max_output_tokens=route["max_output_tokens"], max_attempts=route["max_attempts"],
        timeout_seconds=route["timeout_seconds"], max_http_attempts=3)
    judge_config = JudgeConfig(api_key=os.environ["RDSEC_API_KEY"])
    secret = os.environ.get("SHADOW_INCUMBENT_API_KEY") or os.environ["RDSEC_API_KEY"]
    allowed = [candidate_config.endpoint, route["base_url"] + "/chat/completions", judge_config.endpoint]
    started = time.monotonic()
    experiment = {"schema_version": "news-shadow-experiment-result/v1", "experiment_id": experiment_id,
                  "status": "running", "mode": mode, "cohort": cohort, "manifest": manifest,
                  "identity": identity, "policy_frozen": policy["frozen"], "timings": {},
                  "control_model": route["model"],
                  "judge": {"model": JUDGE_MODEL, "max_output_tokens": judge_config.max_output_tokens}, "human_review_required": True,
                  "promotion": "manual_only"}
    write_json(destination / "experiment.json", experiment)
    try:
        with model_egress_only(allowed):
            control_start = time.monotonic()
            control = await IncumbentAdapter(control_config, secret).evaluate(frozen, budget=budgets["incumbent"])
            experiment["timings"]["control_filter_seconds"] = time.monotonic() - control_start
            validate_decisions(records, control["decisions"])
            if control.get("input_sha256") != frozen["input_sha256"]:
                raise ValueError("Control input hash changed")
            write_json(destination / "control-decision.json", control)
            candidate_start = time.monotonic()
            candidate = await TypeSafeAdapter(candidate_config, os.environ["TYPESAFE_API_KEY"]).evaluate(
                records, policy=policy, budget=budgets["candidate"])
            experiment["timings"]["candidate_filter_seconds"] = time.monotonic() - candidate_start
            validate_decisions(records, candidate["decisions"])
            if candidate.get("input_sha256") != frozen["input_sha256"]:
                raise ValueError("Candidate input hash changed")
            write_json(destination / "candidate-decision.json", candidate)
            if repeat_control:
                repeated = await IncumbentAdapter(control_config, secret).evaluate(frozen, budget=budgets["incumbent"])
                validate_decisions(records, repeated["decisions"])
                write_json(destination / "control-repeat.json", repeated)
                experiment["control_repeat"] = compare_decisions(records, original["decisions"],
                    control["decisions"], repeated["decisions"], candidate_requests=attempt_metrics(repeated))
                experiment["control_repeat_evidence"] = control_repeat_evidence(control, repeated)
            if mode == "pipeline":
                from .pipeline_runner import run_pipeline_pair
                # Finish branch execution before starting the adjudication
                # deadline; model waiting time belongs to its own stage.
                try:
                    experiment["pipeline"] = run_pipeline_pair(bundle, destination, control, candidate,
                        repeat_control=repeat_control, deadline=started + 6000)
                except Exception as exc:
                    experiment["pipeline"] = {"status": "unavailable", "failure_type": type(exc).__name__}
            budgets["judge"] = budget_for(policy, "judge")
            with JudgeClient(judge_config, budget=budgets["judge"]) as judge:
                judge_start = time.monotonic()
                adjudication = judge.adjudicate_inputs(records, output_path=destination / "judge-inputs.json")
                if adjudication["input_sha256"] != frozen["input_sha256"]:
                    raise ValueError("Judge did not preserve the frozen input")
                experiment["timings"]["judge_seconds"] = time.monotonic() - judge_start
                comparisons = []
                if mode == "pipeline" and experiment["pipeline"]["status"] == "complete":
                    comparisons, output_metrics = compare_pipeline_outputs(judge, bundle, destination, experiment_id)
                    experiment["pipeline"]["output_metrics"] = output_metrics
                    experiment["pipeline_metrics"] = output_metrics
                experiment["timings"]["judge_seconds"] = time.monotonic() - judge_start
                experiment["judge"].update(status=adjudication["status"],
                    adjudicated_count=len(adjudication["adjudications"]), error_count=len(adjudication["errors"]),
                    same_model_family_bias="deepseek" in route["model"].lower())
                experiment["metrics"] = compare_decisions(records, original["decisions"], control["decisions"],
                    candidate["decisions"], judge_adjudications=adjudication["adjudications"],
                    original_requests=attempt_metrics(original), control_requests=attempt_metrics(control),
                    candidate_requests=attempt_metrics(candidate), judge_requests=adjudication["requests"] +
                        [attempt for pair in comparisons for attempt in pair.get("requests", [])],
                    output_comparisons=comparisons)
                experiment["metrics"]["population"]["input_ids_hash"] = sha256_json([r["id"] for r in records])
                experiment["filter_usage"] = {"control": control.get("usage"), "candidate": candidate.get("usage")}
                if (bundle / "diagnostics/filter-cost.json").is_file():
                    experiment["filter_usage"]["original_recorded_estimate"] = read_json(bundle / "diagnostics/filter-cost.json")
                experiment["filter_status"] = "complete" if all(v == "complete" for v in
                    (control["status"], candidate["status"], adjudication["status"])) else "degraded"
                output_status = all(row.get("status") == "complete" for row in comparisons)
                repeat_verified = not repeat_control or experiment["control_repeat_evidence"]["distinct_response_ids"]
                experiment["status"] = "complete" if output_status and repeat_verified and all(v == "complete" for v in
                    (control["status"], candidate["status"], adjudication["status"],
                     experiment.get("pipeline", {}).get("status", "complete"))) else "degraded"
    except Exception as exc:
        # HTTP error bodies and exception strings may contain credentials or
        # prompts. Detailed safe adapter diagnostics remain in their artifacts.
        experiment.update(status="failed", failure={"type": type(exc).__name__})
    finally:
        experiment["budget"] = {role: budget.snapshot() for role, budget in budgets.items()}
        experiment["timings"]["experiment_seconds"] = time.monotonic() - started
        write_json(destination / "experiment.json", experiment)
        report = write_assessment(destination / "assessment.md", experiment)
        write_json(destination / "assessment.json", report)
    return {"experiment_id": experiment_id, "status": experiment["status"], "path": str(destination)}


def compare_pipeline_outputs(judge, bundle: Path, destination: Path, seed: str) -> tuple[list[dict], dict]:
    """Judge summaries and executive synthesis with blinded branch labels."""
    comparisons = []
    manifest = read_json(bundle / "manifest.json")
    date = manifest["report_date"]
    names = ("summary.json", "news.json", "research.json", "social.json", "reddit.json")
    branches = ["control", "candidate"]
    if (destination / "control-repeat/web/data" / date / "summary.json").is_file():
        branches.append("control-repeat")
    outputs = {branch: {name: read_json(destination / branch / "web/data" / date / name)
                        for name in names} for branch in branches}
    outputs["original"] = {name: read_json(bundle / "original" / name) for name in names}
    # Downstream synthesis can cite any gathered category. Use the same frozen
    # source excerpts for every pair, including items only one branch retained.
    projections = {branch: {name: output_review_projection(rows[name]) for name in ("summary.json", "news.json")}
                   for branch, rows in outputs.items()}
    referenced = set(re.findall(r"\b[0-9a-f]{12}\b", json.dumps(projections, ensure_ascii=False)))
    evidence = []
    for category in ("news", "research", "social", "reddit"):
        for row in read_json(bundle / "gathered" / f"{category}.json"):
            if row["id"] in referenced:
                text = "\n".join(str(row.get(key) or "") for key in ("title", "source", "content"))
                evidence.append({"id": row["id"], "text": text[:4000]})
    if sum(len(row["text"].encode("utf-8")) for row in evidence) > 500000:
        # Mark unavailable instead of silently dropping sources from a judge
        # packet and mislabelling resulting unsupported claims as hallucinations.
        return [{"status": "unavailable", "reason": "output_evidence_exceeds_review_bound"}], {}
    pairs = [("original", "control"), ("control", "candidate")]
    if "control-repeat" in outputs:
        pairs.append(("control", "control-repeat"))
    from .metrics import compare_pipeline_views
    metrics = {name: compare_pipeline_views(outputs["original"][name], outputs["control"][name],
               outputs["candidate"][name], outputs.get("control-repeat", {}).get(name)) for name in names}
    for left, right in pairs:
        for name in ("summary.json", "news.json"):
            result = judge.compare_output_pair(projections[left][name],
                projections[right][name], pair_id=f"{date}:{left}:{right}:{name}",
                seed=seed, evidence=evidence, article_ids=[row["id"] for row in evidence],
                output_path=destination / f"judge-{left}-{right}-{name}")
            result["branches"] = {"control": left, "candidate": right}
            comparisons.append(result)
    return comparisons, metrics


def output_review_projection(value: dict) -> dict:
    """Bound blinded review to synthesis and the highest ranked twenty items."""
    fields = ("executive_summary", "category_summary", "themes", "top_topics", "category")
    result = {key: value[key] for key in fields if key in value}
    items = value.get("items", [])
    if isinstance(items, list):
        result["items"] = [{key: row[key] for key in
                           ("id", "title", "summary", "summary_html", "importance_score", "rank", "url") if key in row}
                          for row in items[:20] if isinstance(row, dict)]
    return result
