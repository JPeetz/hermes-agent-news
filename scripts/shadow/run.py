#!/usr/bin/env python3
"""Explicitly execute a paid shadow experiment in a scrubbed child process."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import signal
import re

import _bootstrap  # noqa: F401
from shadow.runtime import COHORTS, DEFAULT_POLICY, ROOT, inspect_run, model_child_environment, validate_output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--mode", choices=("filter", "pipeline"), default="filter")
    parser.add_argument("--cohort", choices=COHORTS, default="engineering")
    parser.add_argument("--repeat-control", action="store_true")
    parser.add_argument("--retry-version", default="1",
                        help="explicit version for a fresh, separately identified retry")
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", args.retry_version):
        raise ValueError("Invalid retry version")
    # This preflight makes no model calls and is intentionally before launching
    # a child or importing any HTTP/client modules.
    inspect_run(args.bundle, args.policy, mode=args.mode, cohort=args.cohort, require_credentials=True)
    validate_output_root(args.out, args.bundle)
    if not args.internal_worker:
        process = subprocess.Popen([sys.executable, str(__file__), *sys.argv[1:], "--internal-worker"],
                                   cwd=ROOT, env=model_child_environment(), start_new_session=True)
        try:
            return process.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            print("Shadow experiment exceeded its process deadline", file=sys.stderr)
            return 2
        finally:
            if process.poll() is None:
                # This newly created process group belongs only to this run,
                # including any active downstream replay worker.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    from shadow.experiment import run_experiment
    result = asyncio.run(run_experiment(args.bundle, args.out, args.policy, mode=args.mode,
                                       cohort=args.cohort, repeat_control=args.repeat_control,
                                       retry_version=args.retry_version))
    print(json.dumps(result))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        # Do not include provider error text or environment values in CI output.
        print(f"Shadow execution unavailable: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2)
