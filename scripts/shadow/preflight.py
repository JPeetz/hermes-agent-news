#!/usr/bin/env python3
"""Inspect shadow integrity/configuration; model calls require --probe-models."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import _bootstrap  # noqa: F401
from shadow.contracts import BundleValidationError, write_json
from shadow.runtime import COHORTS, DEFAULT_POLICY, inspect_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--mode", choices=("filter", "pipeline"), default="filter")
    parser.add_argument("--cohort", choices=COHORTS, default="engineering")
    parser.add_argument("--out")
    parser.add_argument("--require-credentials", action="store_true")
    parser.add_argument("--probe-models", action="store_true", help="Spend on three bounded synthetic model-access probes")
    args = parser.parse_args()
    try:
        result = inspect_run(args.bundle, args.policy, mode=args.mode, cohort=args.cohort,
                             require_credentials=args.require_credentials or args.probe_models)
        if args.probe_models:
            from shadow.preflight import probe_models
            probes = asyncio.run(probe_models())
            result.update(probes)
            result["ready"] = result["ready"] and probes["model_access_verified"]
    except Exception as exc:
        result = {"schema_version": "news-shadow-preflight/v1", "ready": False,
                  "model_access_verified": False, "source_access_verified": False,
                  "failure_type": type(exc).__name__}
        if isinstance(exc, BundleValidationError):
            result["reason"] = str(exc)
        print(f"Preflight unavailable: {type(exc).__name__}", file=sys.stderr)
    if args.out:
        write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
