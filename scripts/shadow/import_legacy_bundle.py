#!/usr/bin/env python3
"""Recover a legacy diagnostics archive without inference or production imports.

Offline metadata must include run_id, run_attempt, execution_sha, event_sha,
conclusion, and optionally report_date and artifact.digest. execution_sha is an
explicit provenance assertion; it must describe the actual checkout, not head_sha.
For publication proof use trusted workflow logs plus GitHub, or --git-repo with
explicit output_commit metadata. Omit both to preserve an unverified publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from shadow.contracts import read_json
from shadow.github import GitHubSource, SOURCE_REPOSITORY, SHA_RE, lineage_from_logs, parse_date
from shadow.history import import_legacy_bundle, read_diagnostics_archive, HistoricalEvidenceError, MAX_ARCHIVE_BYTES


def _local_publication(repository: Path, commit: str, report_date: str):
    parse_date(report_date)
    if not SHA_RE.fullmatch(commit):
        raise HistoricalEvidenceError("Offline output_commit must be a full commit SHA")
    def git(*args):
        result = subprocess.run(["git", "-C", str(repository), *args], capture_output=True, check=False, timeout=60)
        if result.returncode:
            raise HistoricalEvidenceError("Could not verify publication against local Git objects")
        return result.stdout
    if git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise HistoricalEvidenceError("Output commit does not resolve exactly")
    current = git("rev-parse", "--verify", "refs/remotes/origin/main^{commit}").decode().strip()
    files, current_files = {}, {}
    for name in ("summary", "news", "research", "social", "reddit"):
        path = f"web/data/{report_date}/{name}.json"
        files[f"{name}.json"] = git("show", f"{commit}:{path}")
        current_files[f"{name}.json"] = git("show", f"{current}:{path}")
    status = "published" if files == current_files else "superseded"
    revert = None
    if status != "published":
        history = git("log", "--format=%H%x00%B%x00", f"{commit}..{current}", "--", f"web/data/{report_date}").decode()
        chunks = history.split("\x00")
        for index in range(0, len(chunks) - 1, 2):
            if f"This reverts commit {commit}" in chunks[index + 1]:
                status, revert = "reverted", chunks[index].strip()
    return {"status": status, "output_commit": commit, "current_commit": current,
            "current_reference": "local refs/remotes/origin/main (last fetched snapshot)",
            "revert_commit": revert, "verification": "local_git_blob_hash",
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}, files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=SOURCE_REPOSITORY, choices=[SOURCE_REPOSITORY])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--diagnostics", type=Path, help="Local pipeline-diagnostics ZIP (offline)")
    source.add_argument("--run-id", type=int, help="Download a source GitHub run through existing gh auth")
    parser.add_argument("--attempt", type=int, help="Required for GitHub imports; never silently selects latest")
    parser.add_argument("--run-metadata", type=Path, help="Offline run/attempt and execution/publication provenance JSON")
    parser.add_argument("--logs", type=Path, help="Saved gh run view --log output; no logs are copied into the bundle")
    parser.add_argument("--git-repo", type=Path, help="Verify offline output_commit against local Git objects")
    parser.add_argument("--out", required=True, type=Path, help="New immutable bundle directory")
    args = parser.parse_args(argv)
    try:
        publication, outputs = None, None
        github = None
        if args.diagnostics:
            if not args.run_metadata:
                parser.error("--diagnostics requires --run-metadata")
            if args.attempt is not None:
                parser.error("Offline run_attempt comes from --run-metadata")
            if args.diagnostics.stat().st_size > MAX_ARCHIVE_BYTES:
                raise HistoricalEvidenceError("Archive exceeds size limit")
            payload = args.diagnostics.read_bytes()
            metadata = read_json(args.run_metadata)
        else:
            if args.attempt is None:
                parser.error("--run-id requires explicit --attempt")
            if args.run_metadata or args.logs or args.git_repo:
                parser.error("Offline metadata/logs/Git options cannot be mixed with GitHub mode")
            github = GitHubSource(args.source_repo)
            run = github.run(args.run_id, args.attempt)
            if run.get("status") != "completed":
                raise HistoricalEvidenceError("Source attempt is still running")
            payload, artifact = github.diagnostics(run)
            metadata = {"repository": args.source_repo, "run_id": run["id"], "run_attempt": run["run_attempt"],
                        "event_sha": run["head_sha"], "conclusion": run.get("conclusion"), "artifact": artifact}
            lineage = lineage_from_logs(github.logs(args.run_id, args.attempt))
            metadata.update(lineage)
            metadata["execution_sha_evidence"] = "trusted_workflow_step_logs"
        if args.logs:
            lineage = lineage_from_logs(args.logs.read_text())
            for key in ("execution_sha", "output_commit"):
                if metadata.get(key) and lineage.get(key) and not metadata[key].startswith(lineage[key]):
                    raise HistoricalEvidenceError("Explicit metadata disagrees with trusted step logs")
                if lineage.get(key):
                    metadata[key] = lineage[key]
            metadata["execution_sha_evidence"] = "trusted_workflow_step_logs"
        members = read_diagnostics_archive(payload)
        dates = {name.split("/")[1] for name in members if name.startswith("checkpoints/")}
        report_date = metadata.get("report_date") or (next(iter(dates)) if len(dates) == 1 else None)
        if metadata.get("output_commit") and report_date:
            if github:
                publication, outputs = github.publication(metadata["output_commit"], report_date)
            elif args.git_repo:
                publication, outputs = _local_publication(args.git_repo, metadata["output_commit"], report_date)
        manifest = import_legacy_bundle(payload, metadata, args.out, publication, outputs)
        print(json.dumps({"bundle": str(args.out.resolve()), "bundle_sha256": manifest["bundle_sha256"],
                          "report_date": manifest["report_date"], "source": manifest["source"],
                          "capabilities": manifest["capabilities"], "eligibility": manifest["eligibility"],
                          "publication": manifest["publication"], "missing_dependencies": manifest["missing_dependencies"]}, indent=2))
        return 0 if manifest["capabilities"]["filter_replay"] else 2
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Import unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
