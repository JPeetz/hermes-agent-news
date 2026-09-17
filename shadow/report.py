"""Self-contained concise assessment reports for news shadow experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _value(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return default
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _count(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    return value if isinstance(value, int) and value >= 0 else 0


def _model_family(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    name = value.lower().rsplit("/", 1)[-1]
    for marker in ("-v", ":v", "_v"):
        if marker in name:
            name = name.split(marker, 1)[0]
    return name


def _branch_summary(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    branches = metrics.get("branches")
    if isinstance(branches, Mapping) and isinstance(branches.get(name), Mapping):
        return branches[name]
    return {}


def _difference(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    differences = metrics.get("set_differences")
    if isinstance(differences, Mapping) and isinstance(differences.get(name), Mapping):
        return differences[name]
    return {}


def _quality(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    quality = metrics.get("judge_estimated_quality")
    if isinstance(quality, Mapping) and isinstance(quality.get(name), Mapping):
        return quality[name]
    return {}


def _usage(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    usage = metrics.get("usage")
    if isinstance(usage, Mapping) and isinstance(usage.get(name), Mapping):
        return usage[name]
    return {}


def build_assessment(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe report model from one joined experiment object.

    The report includes provenance and metrics, but intentionally excludes raw
    prompts, responses, API keys, and branch-blinding payloads.  Those remain in
    the immutable experiment artifacts.
    """

    manifest = experiment.get("manifest") if isinstance(experiment.get("manifest"), Mapping) else {}
    metrics = experiment.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = experiment.get("comparison") if isinstance(experiment.get("comparison"), Mapping) else {}
    judge = experiment.get("judge") if isinstance(experiment.get("judge"), Mapping) else {}
    budget = experiment.get("budget") if isinstance(experiment.get("budget"), Mapping) else {}
    timings = experiment.get("timings")
    pipeline_result = experiment.get("pipeline") if isinstance(experiment.get("pipeline"), Mapping) else {}
    if not isinstance(timings, Mapping):
        timings = metrics.get("timings") if isinstance(metrics.get("timings"), Mapping) else {}
    status = str(experiment.get("status") or manifest.get("status") or "unknown")
    mode = str(experiment.get("mode") or manifest.get("mode") or "filter")
    experiment_id = str(
        experiment.get("experiment_id")
        or manifest.get("experiment_id")
        or manifest.get("id")
        or "unidentified"
    )
    population = metrics.get("population") if isinstance(metrics.get("population"), Mapping) else {}
    candidate_control = _difference(metrics, "candidate_vs_control")
    candidate_branch = _branch_summary(metrics, "candidate")
    control_branch = _branch_summary(metrics, "control")
    candidate_quality = _quality(metrics, "candidate")
    control_quality = _quality(metrics, "control")
    output_comparisons = metrics.get("output_comparisons") if isinstance(metrics.get("output_comparisons"), Mapping) else {}
    critical_findings = output_comparisons.get("critical_findings") if isinstance(output_comparisons.get("critical_findings"), list) else []
    critical_finding_summary = [
        {
            key: finding.get(key)
            for key in ("kind", "side", "severity", "article_id", "reason")
            if finding.get(key) is not None
        }
        for finding in critical_findings
        if isinstance(finding, Mapping)
    ]
    critical_lost = _count(candidate_quality, "critical_lost_count")
    judge_errors = _count(candidate_branch, "error_count") + _count(control_branch, "error_count")
    judge_model = judge.get("model") or "deepseek-v4.1-flash"
    control_model = judge.get("control_model") or experiment.get("control_model")
    if control_model is None and isinstance(manifest.get("versions"), Mapping):
        control_model = manifest["versions"].get("control_model") or manifest["versions"].get("incumbent_model")
    declared_family_bias = judge.get("same_model_family_bias")
    if isinstance(declared_family_bias, bool):
        # The runner may know the route model even when it is not repeated in
        # the manifest.  Preserve that explicit provenance rather than
        # treating a missing control_model as evidence of no overlap.
        model_family_bias: bool | None = declared_family_bias
    elif isinstance(control_model, str) and isinstance(judge_model, str):
        model_family_bias = _model_family(control_model) == _model_family(judge_model)
    else:
        model_family_bias = None
    judge_usage = _usage(metrics, "judge")
    judge_elapsed = judge.get("elapsed_seconds")
    if judge_elapsed is None and isinstance(judge_usage.get("elapsed_ms_known"), (int, float)):
        judge_elapsed = judge_usage["elapsed_ms_known"] / 1000
    if status not in {"complete", "success"}:
        gate = "not_eligible"
    elif critical_lost or critical_findings or judge_errors:
        gate = "requires_human_review"
    else:
        gate = "review_required"
    pipeline_metrics = experiment.get("pipeline_metrics")
    if not isinstance(pipeline_metrics, Mapping):
        pipeline_metrics = metrics.get("pipeline") if isinstance(metrics.get("pipeline"), Mapping) else None
    report = {
        "schema_version": "news-shadow-assessment/v1",
        "experiment_id": experiment_id,
        "status": status,
        "mode": mode,
        "gate": gate,
        "quality_basis": "judge_estimated_not_ground_truth",
        "human_review_required": True,
        "provenance": {
            key: manifest[key]
            for key in (
                "report_date",
                "coverage",
                "source",
                "publication",
                "capabilities",
                "versions",
                "bundle_sha256",
            )
            if key in manifest
        },
        "judge": {
            "status": judge.get("status", "unknown"),
            "model": judge_model,
            "control_model": control_model,
            "model_family_bias": model_family_bias,
            # Missing is different from all-input coverage.  Keep it null so
            # an absent judge artifact cannot masquerade as a complete run.
            "adjudicated_count": judge.get("adjudicated_count") if isinstance(judge.get("adjudicated_count"), int) else None,
            "error_count": _count(judge, "error_count"),
            "elapsed_seconds": judge_elapsed,
        },
        "population": {
            "input_count": population.get("input_count"),
            "input_ids_hash": population.get("input_ids_hash"),
        },
        "branches": {
            "control": {
                "coverage_count": control_branch.get("coverage_count"),
                "coverage_rate": control_branch.get("coverage_rate"),
                "kept_count": control_branch.get("kept_count"),
                "abstention_rate": control_branch.get("abstention_rate"),
                "error_rate": control_branch.get("error_rate"),
                "fallback_rate": control_branch.get("fallback_rate"),
            },
            "candidate": {
                "coverage_count": candidate_branch.get("coverage_count"),
                "coverage_rate": candidate_branch.get("coverage_rate"),
                "kept_count": candidate_branch.get("kept_count"),
                "abstention_rate": candidate_branch.get("abstention_rate"),
                "error_rate": candidate_branch.get("error_rate"),
                "fallback_rate": candidate_branch.get("fallback_rate"),
            },
        },
        "selection_change": {
            "added_count": candidate_control.get("added_count"),
            "removed_count": candidate_control.get("removed_count"),
            "added_ids": candidate_control.get("added_ids", []),
            "removed_ids": candidate_control.get("removed_ids", []),
            "jaccard": candidate_control.get("jaccard"),
        },
        "judge_estimated_quality": {
            "control": {
                "coverage_rate": control_quality.get("coverage_rate"),
                "precision_estimate": control_quality.get("precision_estimate"),
                "recall_estimate": control_quality.get("recall_estimate"),
                "irrelevant_keep_rate_estimate": control_quality.get("irrelevant_keep_rate_estimate"),
                "critical_lost_count": control_quality.get("critical_lost_count"),
            },
            "candidate": {
                "coverage_rate": candidate_quality.get("coverage_rate"),
                "precision_estimate": candidate_quality.get("precision_estimate"),
                "recall_estimate": candidate_quality.get("recall_estimate"),
                "irrelevant_keep_rate_estimate": candidate_quality.get("irrelevant_keep_rate_estimate"),
                "critical_lost_count": candidate_quality.get("critical_lost_count"),
            },
        },
        "usage": {
            "control": _usage(metrics, "control"),
            "candidate": _usage(metrics, "candidate"),
            "judge": _usage(metrics, "judge"),
            "budget": dict(budget),
        },
        "filter_cost_details": dict(experiment.get("filter_usage") or {}),
        "control_repeat_evidence": experiment.get("control_repeat_evidence"),
        "downstream_usage": {name: {"status": result.get("status"), "usage": result.get("usage"),
                                    "wall_seconds": result.get("wall_seconds")}
                             for name, result in pipeline_result.get("branches", {}).items()},
        "total_experiment_cost_usd": None,
        "total_cost_basis": "Unknown until all model-route prices and failed-attempt usage are verified; repeated controls and judge work are evaluation overhead.",
        "timings": dict(timings),
        "output_comparisons": {
            "comparison_count": output_comparisons.get("comparison_count"),
            "overall": output_comparisons.get("overall", {}),
            "critical_finding_count": len(critical_findings),
            "critical_findings": critical_finding_summary,
            "repeat_selected_count": output_comparisons.get("repeat_selected_count"),
            "repeat_consistency_rate": output_comparisons.get("repeat_consistency_rate"),
        },
        "limitations": [
            "Judge-estimated quality is not ground truth and must not be reported as labelled precision or recall.",
            "Abstentions and unknown usage/cost retain explicit uncertainty; undefined denominators remain null.",
            "Human review of critical losses and judge disagreements is required before any production proposal.",
        ],
    }
    if isinstance(pipeline_metrics, Mapping):
        # Keep branch-output drift separate from model-judge outcomes.  The
        # latter are evidence-backed quality estimates; this section is a
        # deterministic artifact comparison and may be unavailable in
        # filter-only experiments.  The runner may provide one metric object
        # or a map keyed by output artifact (summary.json/news.json/etc.).
        if isinstance(pipeline_metrics.get("comparisons"), Mapping):
            pipeline_report = {
                "schema_version": pipeline_metrics.get("schema_version"),
                "top_k": pipeline_metrics.get("top_k"),
                "comparisons": pipeline_metrics.get("comparisons", {}),
            }
        else:
            pipeline_report = {
                "artifacts": {
                    str(name): {
                        "schema_version": item.get("schema_version"),
                        "top_k": item.get("top_k"),
                        "comparisons": item.get("comparisons", {}),
                    }
                    for name, item in pipeline_metrics.items()
                    if isinstance(item, Mapping) and isinstance(item.get("comparisons"), Mapping)
                }
            }
        if pipeline_report.get("comparisons") or pipeline_report.get("artifacts"):
            report["pipeline_comparisons"] = pipeline_report
    return report


def render_assessment(value: Mapping[str, Any]) -> str:
    """Render a compact, self-contained Markdown assessment."""

    report = value if value.get("schema_version") == "news-shadow-assessment/v1" else build_assessment(value)
    provenance = report.get("provenance") if isinstance(report.get("provenance"), Mapping) else {}
    population = report.get("population") if isinstance(report.get("population"), Mapping) else {}
    branches = report.get("branches") if isinstance(report.get("branches"), Mapping) else {}
    change = report.get("selection_change") if isinstance(report.get("selection_change"), Mapping) else {}
    quality = report.get("judge_estimated_quality") if isinstance(report.get("judge_estimated_quality"), Mapping) else {}
    usage = report.get("usage") if isinstance(report.get("usage"), Mapping) else {}
    output = report.get("output_comparisons") if isinstance(report.get("output_comparisons"), Mapping) else {}
    pipeline = report.get("pipeline_comparisons") if isinstance(report.get("pipeline_comparisons"), Mapping) else {}
    judge = report.get("judge") if isinstance(report.get("judge"), Mapping) else {}
    timings = report.get("timings") if isinstance(report.get("timings"), Mapping) else {}
    lines = [
        f"# News shadow assessment: {_value(report.get('experiment_id'))}",
        "",
        f"Status: **{_value(report.get('status'))}**  ",
        f"Mode: {_value(report.get('mode'))}  ",
        f"Gate: **{_value(report.get('gate'))}**  ",
        f"Judge model: `{_value(judge.get('model'))}`  ",
        f"Quality basis: **judge-estimated, not ground truth**",
        "",
        "## Scope",
        "",
        f"- Report date: {_value(provenance.get('report_date'))}",
        f"- Source run: {_value((provenance.get('source') or {}).get('run_id') if isinstance(provenance.get('source'), Mapping) else None)}",
        f"- Input records: {_value(population.get('input_count'))}",
        f"- Inputs adjudicated: {_value(judge.get('adjudicated_count'))}",
        f"- Control model: `{_value(judge.get('control_model'))}`; judge/control model-family overlap: {_value(judge.get('model_family_bias'))}",
        "",
        "## Filter comparison",
        "",
        "| Branch | Kept | Coverage | Abstention | Fallback | Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch_name in ("control", "candidate"):
        branch = branches.get(branch_name) if isinstance(branches.get(branch_name), Mapping) else {}
        lines.append(
            f"| {branch_name} | {_value(branch.get('kept_count'))} | "
            f"{_value(branch.get('coverage_rate'))} | {_value(branch.get('abstention_rate'))} | "
            f"{_value(branch.get('fallback_rate'))} | {_value(branch.get('error_rate'))} |"
        )
    lines.extend(
        [
            "",
            f"Candidate vs control: **{_value(change.get('added_count'))} added**, **{_value(change.get('removed_count'))} removed**, Jaccard {_value(change.get('jaccard'))}.",
            f"Added IDs: `{', '.join(change.get('added_ids', [])) or 'none'}`  ",
            f"Removed IDs: `{', '.join(change.get('removed_ids', [])) or 'none'}`",
            "",
            "## Judge-estimated quality",
            "",
            "| Branch | Coverage | Precision estimate | Recall estimate | Irrelevant keep estimate | Critical losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for branch_name in ("control", "candidate"):
        branch = quality.get(branch_name) if isinstance(quality.get(branch_name), Mapping) else {}
        lines.append(
            f"| {branch_name} | {_value(branch.get('coverage_rate'))} | "
            f"{_value(branch.get('precision_estimate'))} | {_value(branch.get('recall_estimate'))} | "
            f"{_value(branch.get('irrelevant_keep_rate_estimate'))} | {_value(branch.get('critical_lost_count'))} |"
        )
    lines.extend(
        [
            "",
            "These estimates come from the dedicated judge's bounded evidence labels. They are not ground truth, and null values mean the denominator or evidence was unavailable.",
            "",
            "## Usage and output review",
            "",
            f"- Filter usage/estimates by branch: `{json.dumps(report.get('filter_cost_details', {}), sort_keys=True)}`. Missing prices remain unknown; recorded estimates are not invoices.",
            f"- Downstream usage by branch: `{json.dumps(report.get('downstream_usage', {}), sort_keys=True)}`. Reused categories and hero generation are excluded from incremental spend.",
            f"- Total experiment cost: unknown. {report.get('total_cost_basis', '')}",
            f"- Judge requests: {_value((usage.get('judge') or {}).get('request_count') if isinstance(usage.get('judge'), Mapping) else None)}; cost status: **{_value((usage.get('judge') or {}).get('cost_status') if isinstance(usage.get('judge'), Mapping) else None)}**; known cost: {_value((usage.get('judge') or {}).get('cost_usd_known') if isinstance(usage.get('judge'), Mapping) else None)}.",
            f"- Timings: `{json.dumps(timings, sort_keys=True)}`.",
            f"- Repeated-control identity evidence: `{json.dumps(report.get('control_repeat_evidence'), sort_keys=True)}`. Distinct response IDs show no duplicate completion was observed; they do not prove uncached inference.",
            f"- Output comparisons: {_value(output.get('comparison_count'))}; overall outcomes: `{json.dumps(output.get('overall', {}), sort_keys=True)}`.",
            f"- Critical output findings: {_value(output.get('critical_finding_count'))}.",
            f"- Opposite-order repeat checks: {_value(output.get('repeat_selected_count'))}; consistency: {_value(output.get('repeat_consistency_rate'))}.",
            f"- Human review required: **{_value(report.get('human_review_required'))}**.",
            "",
            "## Limitations",
            "",
        ]
    )
    if pipeline:
        comparisons = pipeline.get("comparisons") if isinstance(pipeline.get("comparisons"), Mapping) else {}
        lines.extend(["", "## Deterministic pipeline-output comparison", ""])
        artifact_comparisons = pipeline.get("artifacts") if isinstance(pipeline.get("artifacts"), Mapping) else {"output": {"comparisons": comparisons}}
        for artifact_name, artifact in artifact_comparisons.items():
            if not isinstance(artifact, Mapping):
                continue
            artifact_pairs = artifact.get("comparisons") if isinstance(artifact.get("comparisons"), Mapping) else {}
            for key in ("original_vs_control", "control_vs_candidate", "control_vs_repeated_control"):
                pair = artifact_pairs.get(key) if isinstance(artifact_pairs.get(key), Mapping) else {}
                survivor = pair.get("survivor_top_k") if isinstance(pair.get("survivor_top_k"), Mapping) else {}
                lines.append(
                    f"- {artifact_name} {key}: available={_value(pair.get('available'))}; "
                    f"added={_value(pair.get('added_count'))}; removed={_value(pair.get('removed_count'))}; "
                    f"rank changes={_value(pair.get('rank_changed_count'))}; "
                    f"top-k overlap={_value(survivor.get('overlap'))}."
                )
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    findings = output.get("critical_findings")
    if isinstance(findings, list) and findings:
        lines.extend(["", "Critical findings requiring review:"])
        for finding in findings:
            if isinstance(finding, Mapping):
                lines.append(
                    f"- {_value(finding.get('severity'))} {_value(finding.get('kind'))} "
                    f"({_value(finding.get('side'))}, article {_value(finding.get('article_id'))}): "
                    f"{_value(finding.get('reason'))}"
                )
    return "\n".join(lines).rstrip() + "\n"


def write_assessment(path: str | Path, experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Write Markdown and return the JSON-safe report model."""

    report = build_assessment(experiment)
    target = Path(path)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlinked assessment path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_assessment(report), encoding="utf-8")
    return report


def write_report_json(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.is_symlink():
        raise ValueError("refusing to replace a symlinked report path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


__all__ = ["build_assessment", "render_assessment", "write_assessment", "write_report_json"]
