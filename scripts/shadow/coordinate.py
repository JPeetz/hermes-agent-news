#!/usr/bin/env python3
"""Offline-safe coordinator entry point for the internal shadow workflow.

The command reads JSON metadata and verified bundle manifests.  It never
executes files from a downloaded source artifact and it never calls a model or
GitHub.  The workflow's acquisition job is responsible for producing the
metadata JSON; this script records the bounded decision in the results index.

Examples::

    python3 scripts/shadow/coordinate.py discover \
      --metadata /tmp/source-runs.json --from-date 2026-09-09 \
      --to-date 2026-09-12 --state-dir /tmp/shadow-state

    python3 scripts/shadow/coordinate.py record \
      --bundle /tmp/bundles/35192960377-1 --mode filter \
      --policy-version news-relevance-v1-dev --state-dir /tmp/shadow-state
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Keep direct execution independent of the production application and its
# optional network dependencies.
from _bootstrap import ROOT  # noqa: F401

from shadow.coordinator import (  # noqa: E402
    Coordinator,
    DiscoveryError,
    SelectorError,
    parse_selector,
    validate_trusted_ref,
)
from shadow.contracts import BundleValidationError, hash_file, read_json, sha256_json  # noqa: E402
from shadow.state import (  # noqa: E402
    StateConflict,
    StateError,
    STATUS_COMPLETED,
    STATUS_DISCOVERED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_RUNNING,
    STATUS_UNAVAILABLE,
)


DEFAULT_STATE_DIR = Path(".shadow-state")
STATUS_CHOICES = (
    STATUS_DISCOVERED,
    STATUS_RUNNING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_UNAVAILABLE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser(
        "discover",
        help="classify source-run metadata and append metadata-only records",
    )
    discover.add_argument("--metadata", required=True, help="JSON metadata inventory; data only")
    discover.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    discover.add_argument("--source-runs", help="comma/space separated run_id[:attempt] selectors")
    discover.add_argument("--from-date", help="inclusive report date, YYYY-MM-DD")
    discover.add_argument("--to-date", help="inclusive report date, YYYY-MM-DD")
    discover.add_argument("--mode", choices=("filter", "pipeline"), default="filter")
    discover.add_argument("--policy-version")
    discover.add_argument("--rubric-version")
    discover.add_argument("--candidate-model")
    discover.add_argument("--judge-version")
    discover.add_argument("--judge-model")
    discover.add_argument("--sampling-plan")
    discover.add_argument("--repeat-plan")
    discover.add_argument("--policy", help="actual policy JSON; its file hash is part of identity")
    discover.add_argument("--evaluator-sha", help="full trusted evaluator commit SHA")
    discover.add_argument("--cohort")
    discover.add_argument(
        "--cohort-file", default="config/shadow/cohorts-2026-09-17.json",
        help="frozen cohort manifest whose canonical hash is part of identity",
    )
    discover.add_argument("--cohort-sha256")
    discover.add_argument("--incumbent-route")
    discover.add_argument("--incumbent-model")
    discover.add_argument("--trusted-ref", default="main")

    record = sub.add_parser(
        "record",
        help="verify a sealed bundle and append/update its compact index record",
    )
    record.add_argument("--bundle", required=True)
    record.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    record.add_argument("--mode", choices=("filter", "pipeline"), default="filter")
    record.add_argument("--status", choices=STATUS_CHOICES, default=STATUS_DISCOVERED)
    record.add_argument("--policy-version")
    record.add_argument("--rubric-version")
    record.add_argument("--candidate-model")
    record.add_argument("--judge-version")
    record.add_argument("--judge-model")
    record.add_argument("--sampling-plan")
    record.add_argument("--repeat-plan")
    record.add_argument("--policy", help="actual policy JSON; its file hash is part of identity")
    record.add_argument("--evaluator-sha", help="full trusted evaluator commit SHA")
    record.add_argument("--cohort")
    record.add_argument(
        "--cohort-file", default="config/shadow/cohorts-2026-09-17.json",
        help="frozen cohort manifest whose canonical hash is part of identity",
    )
    record.add_argument("--cohort-sha256")
    record.add_argument("--incumbent-route")
    record.add_argument("--incumbent-model")
    record.add_argument("--result-id", help="actual evaluator experiment_id, if a run completed")
    record.add_argument("--result-path", help="artifact path containing the evaluator result")
    record.add_argument("--require-healthy", action="store_true")
    record.add_argument("--require-published", action="store_true")
    record.add_argument("--trusted-ref", default="main")

    listing = sub.add_parser("list", help="print compact coordinator records")
    listing.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    listing.add_argument("--status", choices=STATUS_CHOICES, action="append")

    claim = sub.add_parser("claim", help="claim at most one serial work item")
    claim.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    claim.add_argument("--owner", required=True)
    claim.add_argument("--limit", type=int, default=2)
    claim.add_argument("--lease-seconds", type=float, default=1800)
    claim.add_argument("--experiment-id", action="append",
                       help="restrict claims to identities acquired by this job; repeatable")

    return parser


def _read_metadata(path: str) -> Any:
    try:
        return read_json(Path(path))
    except (OSError, BundleValidationError) as exc:
        raise DiscoveryError(f"cannot read metadata inventory: {exc}") from exc


def _common_versions(args: argparse.Namespace) -> dict[str, Any]:
    policy_sha256 = None
    if getattr(args, "policy", None):
        policy_sha256 = hash_file(Path(args.policy))
    cohort_sha256 = getattr(args, "cohort_sha256", None)
    cohort_file = getattr(args, "cohort_file", None)
    if not cohort_sha256 and cohort_file and getattr(args, "cohort", None):
        cohort_sha256 = sha256_json(read_json(Path(cohort_file)))
    return {
        "mode": args.mode,
        "policy_version": args.policy_version,
        "rubric_version": args.rubric_version,
        "candidate_model": args.candidate_model,
        "judge_version": args.judge_version,
        "judge_model": args.judge_model,
        "sampling_plan": args.sampling_plan,
        "repeat_plan": args.repeat_plan,
        "evaluator_sha": getattr(args, "evaluator_sha", None) or os.environ.get("SHADOW_EVALUATOR_SHA"),
        "policy_sha256": policy_sha256,
        "cohort": getattr(args, "cohort", None),
        "cohort_sha256": cohort_sha256,
        "incumbent_route": getattr(args, "incumbent_route", None)
        or os.environ.get("SHADOW_INCUMBENT_BASE_URL"),
        "incumbent_model": getattr(args, "incumbent_model", None)
        or os.environ.get("SHADOW_INCUMBENT_MODEL"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_trusted_ref(getattr(args, "trusted_ref", "main"))
    if args.command == "discover":
        selector = parse_selector(
            source_runs=args.source_runs,
            from_date=args.from_date,
            to_date=args.to_date,
        )
        coordinator = Coordinator(args.state_dir)
        rows = coordinator.discover(
            _read_metadata(args.metadata),
            selector=selector,
            **_common_versions(args),
        )
        return {
            "schema_version": "news-shadow-coordinate-result/v1",
            "operation": "discover",
            "selector": selector.as_dict(),
            "count": len(rows),
            "records": rows,
            "queued": coordinator.select_pending(limit=2),
        }
    if args.command == "record":
        coordinator = Coordinator(args.state_dir)
        result = coordinator.record_bundle(
            args.bundle,
            status=args.status,
            require_healthy=args.require_healthy,
            require_published=args.require_published,
            result_experiment_id=args.result_id,
            result_path=args.result_path,
            **_common_versions(args),
        )
        return {
            "schema_version": "news-shadow-coordinate-result/v1",
            "operation": "record",
            **result,
        }
    if args.command == "list":
        coordinator = Coordinator(args.state_dir)
        return {
            "schema_version": "news-shadow-coordinate-result/v1",
            "operation": "list",
            "revision": coordinator.state.revision,
            "records": coordinator.state.list_records(statuses=args.status),
        }
    if args.command == "claim":
        coordinator = Coordinator(args.state_dir)
        if args.experiment_id and any(not re.fullmatch(r"[0-9a-f]{64}", value)
                                      for value in args.experiment_id):
            raise SelectorError("Claim experiment IDs must be SHA-256 identities")
        claimed = coordinator.claim_next(
            owner=args.owner,
            limit=args.limit,
            lease_seconds=args.lease_seconds,
            experiment_ids=set(args.experiment_id) if args.experiment_id else None,
        )
        return {
            "schema_version": "news-shadow-coordinate-result/v1",
            "operation": "claim",
            "count": len(claimed),
            "reason": (
                "prospective_date_cap_reached"
                if not claimed and coordinator.prospective_cap_reached()
                else ("no_eligible_work" if not claimed else None)
            ),
            "records": claimed,
        }
    raise AssertionError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        result = run(parser.parse_args(argv))
    except (DiscoveryError, SelectorError, StateConflict, StateError, BundleValidationError, ValueError) as exc:
        error = {
            "schema_version": "news-shadow-coordinate-error/v1",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        print(json.dumps(error, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Actions
    raise SystemExit(main())
