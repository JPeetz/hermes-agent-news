#!/usr/bin/env python3
"""Filesystem CLI for the bounded news-shadow judge and comparison report.

The command is offline unless ``--live`` is explicitly supplied.  Endpoint and
model are fixed by ``shadow.judge.JudgeConfig``; the only environment value
read for live adjudication is the approved API key named by ``--api-key-env``.
No production provider routes or model overrides are consulted.

Examples (all offline):

    python3 scripts/shadow/judge.py --experiment-dir /tmp/exp --mode prepare \
      --records /tmp/records.json
    python3 scripts/shadow/judge.py --experiment-dir /tmp/exp --mode compare

The live invocation is deliberately explicit and is for an owner-authorized
run after preflight; this development CLI never invokes it by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from _bootstrap import ROOT  # noqa: F401  (adds the repository root to sys.path)

from shadow.contracts import read_json, validate_input, write_json
from shadow.judge import (
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    JudgeClient,
    JudgeConfig,
    build_input_artifact,
)
from shadow.metrics import compare_decisions
from shadow.report import build_assessment, write_assessment


def _read(path: Path) -> Any:
    return read_json(path)


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise SystemExit("no artifact found; tried: " + ", ".join(str(path) for path in paths))


def _records_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("records"), list):
        return list(value["records"])
    raise SystemExit("records artifact must be a list or an object with records")


def _decisions_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("decisions"), list):
        return list(value["decisions"])
    raise SystemExit("decision artifact must be a list or an object with decisions")


def _adjudications_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("adjudications"), list):
        return list(value["adjudications"])
    return []


def _records_path(args: argparse.Namespace, root: Path) -> Path:
    return Path(args.records) if args.records else _first_existing(
        root / "relevance" / "input.json", root / "input.json"
    )


def _load_records(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    return _records_from(_read(_records_path(args, root)))


def _write_prepare(args: argparse.Namespace, root: Path) -> int:
    records = _load_records(args, root)
    artifact = build_input_artifact(records)
    destination = Path(args.input_output) if args.input_output else root / "judge" / "input.json"
    write_json(destination, artifact)
    print(f"prepared {len(artifact['records'])} bounded records at {destination}")
    return 0


def _write_adjudication(args: argparse.Namespace, root: Path) -> int:
    input_path = Path(args.input) if args.input else root / "judge" / "input.json"
    if input_path.exists():
        input_artifact = _read(input_path)
        # A prepared artifact is the frozen evidence contract.  Do not let a
        # hand-edited records list silently change the judge population or its
        # hash before an explicit live run.
        validate_input(input_artifact)
        records = _records_from(input_artifact)
    else:
        records = _load_records(args, root)
        input_artifact = build_input_artifact(records)
        write_json(root / "judge" / "input.json", input_artifact)
    if not args.live:
        existing = root / "judge" / "result.json"
        if existing.exists():
            result = _read(existing)
            print(f"offline: existing judge result status={result.get('status', 'unknown')}")
        else:
            print("offline: input prepared; no model inference was requested")
        return 0
    key_name = args.api_key_env
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"--live requires the approved API key in ${key_name}")
    from shadow.budget import BudgetLimits, RequestBudget

    config = JudgeConfig(api_key=api_key, max_output_tokens=args.max_output_tokens)
    budget = RequestBudget(
        BudgetLimits(
            max_requests=config.max_requests,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=args.aggregate_output_tokens,
            deadline_seconds=args.deadline_seconds,
        )
    )
    with JudgeClient(config, budget=budget) as client:
        result = client.adjudicate_inputs(records)
    result["budget"] = budget.snapshot()
    result_path = root / "judge" / "result.json"
    write_json(result_path, result)
    # Keep the compact decisions/requests shape separately for downstream
    # contracts and metrics.
    write_json(root / "judge" / "decisions.json", {
        "decisions": result.get("decisions", []),
        "requests": result.get("requests", []),
    })
    print(f"judge status={result.get('status')} records={len(records)} output={result_path}")
    return 0 if result.get("status") == "complete" else 2


def _load_decision(root: Path, branch: str, override: str | None) -> list[dict[str, Any]]:
    path = Path(override) if override else _first_existing(
        root / branch / "decisions.json",
        root / branch / "decision.json",
        root / f"{branch}-decisions.json",
        root / f"{branch}-decision.json",
    )
    return _decisions_from(_read(path))


def _write_compare(args: argparse.Namespace, root: Path) -> int:
    records = _load_records(args, root)
    original = _load_decision(root, "original", args.original)
    control = _load_decision(root, "control", args.control)
    candidate = _load_decision(root, "candidate", args.candidate)
    judge_path = Path(args.judge_results) if args.judge_results else root / "judge" / "result.json"
    judge_result: Mapping[str, Any] = _read(judge_path) if judge_path.exists() else {}
    metrics = compare_decisions(
        records,
        original,
        control,
        candidate,
        judge_adjudications=_adjudications_from(judge_result),
        original_requests=[],
        control_requests=[],
        candidate_requests=[],
        judge_requests=judge_result.get("requests", []) if isinstance(judge_result, Mapping) else [],
    )
    comparison_dir = root / "comparison"
    metrics_path = Path(args.metrics_output) if args.metrics_output else comparison_dir / "metrics.json"
    write_json(metrics_path, metrics)
    manifest_path = root / "manifest.json"
    manifest = _read(manifest_path) if manifest_path.exists() else {}
    budget = judge_result.get("budget", {}) if isinstance(judge_result, Mapping) else {}
    adjudications = _adjudications_from(judge_result)
    adjudicated_count = len(adjudications) if isinstance(judge_result, Mapping) and "adjudications" in judge_result else None
    experiment = {
        "experiment_id": args.experiment_id or root.name,
        "status": args.status,
        "mode": args.mode,
        "manifest": manifest,
        "metrics": metrics,
        "judge": {
            "status": judge_result.get("status", "unknown") if isinstance(judge_result, Mapping) else "unknown",
            "model": "deepseek-v4.1-flash",
            "adjudicated_count": adjudicated_count,
            "error_count": len(judge_result.get("errors", [])) if isinstance(judge_result, Mapping) and isinstance(judge_result.get("errors"), list) else 0,
        },
        "budget": budget,
    }
    report = build_assessment(experiment)
    assessment_path = Path(args.assessment_output) if args.assessment_output else root / "assessment.md"
    write_assessment(assessment_path, experiment)
    write_json(comparison_dir / "assessment.json", report)
    print(f"wrote metrics={metrics_path} assessment={assessment_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, help="experiment artifact directory")
    parser.add_argument("--mode", choices=("prepare", "adjudicate", "compare"), default="compare")
    parser.add_argument("--records", help="records/input JSON path")
    parser.add_argument("--input", help="prepared judge/input.json path")
    parser.add_argument("--input-output", help="where prepare writes judge input")
    parser.add_argument("--original", help="original decisions artifact")
    parser.add_argument("--control", help="control decisions artifact")
    parser.add_argument("--candidate", help="candidate decisions artifact")
    parser.add_argument("--judge-results", help="judge result artifact")
    parser.add_argument("--metrics-output", help="comparison metrics output")
    parser.add_argument("--assessment-output", help="Markdown assessment output")
    parser.add_argument("--experiment-id")
    parser.add_argument("--status", default="complete")
    parser.add_argument("--live", action="store_true", help="explicitly allow the dedicated model request")
    parser.add_argument("--api-key-env", default="RDSec_API_KEY", help="environment variable containing the approved API key")
    parser.add_argument("--max-output-tokens", type=int, default=DEEPSEEK_MAX_OUTPUT_TOKENS)
    parser.add_argument("--aggregate-output-tokens", type=int, default=1920000)
    parser.add_argument("--deadline-seconds", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.experiment_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        return _write_prepare(args, root)
    if args.mode == "adjudicate":
        return _write_adjudication(args, root)
    return _write_compare(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
