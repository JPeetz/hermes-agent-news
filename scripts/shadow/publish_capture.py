#!/usr/bin/env python3
"""Verify captured output against the pushed Git commit, then seal the bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from finalize_bundle import FinalizationError, finalize_bundle
from shadow.contracts import read_json, write_json
from shadow.history import source_health


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=30).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    bundle = args.bundle
    inputs = read_json(bundle / "input-manifest.json")
    report_date = inputs["report_date"]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", report_date):
        raise FinalizationError("Invalid report date")
    execution_sha = os.environ["NEWS_SHADOW_EXECUTION_SHA"]
    output_commit = git("rev-parse", "HEAD").decode().strip()
    # The workflow already fetched origin/main after pushing. A same-date
    # pre-existing report is not enough: every captured report byte must match.
    git("merge-base", "--is-ancestor", output_commit, "refs/remotes/origin/main")
    hashes = {name: hashlib.sha256(git("show", f"{output_commit}:web/data/{report_date}/{name}")).hexdigest()
              for name in ("summary.json", "news.json", "research.json", "social.json", "reddit.json")}
    metadata = read_json(bundle / "gathered/metadata.json")
    gathering = {"collection_status": metadata.get("collection_status", {}),
                 "categories": {category: read_json(bundle / "gathered" / f"{category}.json")
                                for category in ("news", "research", "social", "reddit")}}
    health = source_health(gathering, read_json(bundle / "original/summary.json"))
    write_json(bundle / "diagnostics/health.json", health)
    common = dict(run_id=os.environ["GITHUB_RUN_ID"], run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
                  execution_sha=execution_sha, output_commit=output_commit, report_date=report_date,
                  event_sha=os.environ["GITHUB_SHA"], repository=os.environ["GITHUB_REPOSITORY"],
                  expected_output_hashes=hashes, healthy=health["healthy"])
    # Missing full-replay dependencies should preserve a useful filter bundle.
    # This check does not claim counterfactual freshness URL coverage.
    from finalize_bundle import _require_pipeline_dependencies
    try:
        _require_pipeline_dependencies(bundle)
        capable = True
        missing = []
    except FinalizationError:
        capable = False
        missing = [{"key": "pipeline_dependencies", "reason": "capture_incomplete"}]
    manifest = finalize_bundle(bundle, pipeline_replay=capable, missing_dependencies=missing, **common)
    print(json.dumps({"bundle_sha256": manifest["bundle_sha256"], "capabilities": manifest["capabilities"],
                      "healthy": health["healthy"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
