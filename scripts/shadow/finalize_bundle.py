#!/usr/bin/env python3
"""Bind a captured bundle to its source run and verified published bytes.

This command is intentionally post-push: it accepts the final execution and
output commit SHAs as inputs, verifies the five report files in the bundle,
writes a publication receipt, and seals the directory once.  It never shells
out to GitHub, copies logs/environment state, or marks a bundle published from
a same-date path alone.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow.capture import RUNTIME_ENV_KEYS, SCHEMA_VERSION  # noqa: E402
from shadow.contracts import (  # noqa: E402
    BundleValidationError,
    bundle_path,
    hash_file,
    read_json,
    seal_bundle,
    sha256_json,
    write_json,
)


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OUTPUT_NAMES = ("summary.json", "news.json", "research.json", "social.json", "reddit.json")
CATEGORIES = ("news", "research", "social", "reddit")


class FinalizationError(BundleValidationError):
    """Bundle cannot be safely sealed."""


def _full_sha(value: str | None, field: str, *, required: bool = True) -> str | None:
    if value is None or value == "":
        if required:
            raise FinalizationError(f"{field} is required")
        return None
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise FinalizationError(f"{field} must be a full 40-character commit SHA")
    return value


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
        raise FinalizationError(f"{field} must be a positive integer")
    return int(value)


def _read_input_manifest(bundle: Path) -> tuple[dict[str, Any], str]:
    path = bundle / "input-manifest.json"
    if not path.is_file() or path.is_symlink():
        raise FinalizationError("input-manifest.json is required before sealing")
    value = read_json(path)
    if not isinstance(value, dict):
        raise FinalizationError("input-manifest.json must be an object")
    # The receipt binds the bytes that were captured, including the canonical
    # newline written by the shared JSON helper, rather than a re-serialized
    # object whose representation could differ from the uploaded artifact.
    return value, hash_file(path)


def _output_bytes(bundle: Path, output_dir: str | Path | None, report_date: str) -> tuple[dict[str, bytes], dict[str, str]]:
    root = Path(output_dir) if output_dir is not None else bundle / "original"
    if not root.is_absolute():
        root = bundle / root
    root = root.resolve()
    if bundle.resolve() not in root.parents and root != bundle.resolve():
        raise FinalizationError("Output directory must be inside the bundle")
    files: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    for name in OUTPUT_NAMES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FinalizationError(f"Missing output file: {name}")
        body = path.read_bytes()
        try:
            value = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise FinalizationError(f"Output {name} is not valid JSON") from exc
        if not isinstance(value, dict) or value.get("date") != report_date:
            raise FinalizationError(f"Output {name} is not bound to report date {report_date}")
        if name != "summary.json" and value.get("category") != name[:-5]:
            raise FinalizationError(f"Output {name} has a mismatched category")
        files[name] = body
        hashes[name] = hashlib.sha256(body).hexdigest()
    return files, hashes


def _check_relevance(bundle: Path) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate the shared relevance contracts when present."""
    input_path = bundle / "relevance" / "input.json"
    decision_path = bundle / "relevance" / "incumbent-decision.json"
    if not input_path.is_file() or not decision_path.is_file():
        return False, "relevance_input_or_decision_missing", {}
    try:
        from shadow.contracts import validate_decisions, validate_input

        inputs = validate_input(read_json(input_path))
        decisions = read_json(decision_path)
        if not isinstance(decisions, dict):
            raise FinalizationError("incumbent decision is not an object")
        if decisions.get("input_sha256") != inputs.get("input_sha256"):
            raise FinalizationError("incumbent decision input hash mismatch")
        validate_decisions(inputs["records"], decisions.get("decisions"))
        return True, None, inputs
    except Exception as exc:
        raise FinalizationError(f"Frozen relevance contracts are invalid: {exc}") from exc


def _health(bundle: Path, explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    path = bundle / "diagnostics" / "health.json"
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = read_json(path)
        return isinstance(value, dict) and value.get("healthy") is True
    except Exception:
        return False


def _require_pipeline_dependencies(bundle: Path) -> None:
    summary = read_json(bundle / "original/summary.json")
    if summary.get("hero_image_url"):
        hero = bundle / "original/hero.webp"
        if not hero.is_file() or hero.is_symlink() or not hero.stat().st_size:
            raise FinalizationError("pipeline_replay requires the recorded hero asset")
    required = (
        "context/grounding.txt",
        "context/prompts.yaml",
        "context/effective-config.json",
        "context/runtime-settings.json",
        "context/model-releases-before.yaml",
        "context/history-manifest.json",
        "evidence/index.json",
        "checkpoints/analysis_pre_continuity.json",
        "gathered/metadata.json",
        *(f"gathered/{category}.json" for category in CATEGORIES),
    )
    missing = [relative for relative in required if not (bundle / relative).is_file()]
    if missing:
        raise FinalizationError(
            "pipeline_replay requires frozen dependencies: " + ", ".join(missing)
        )

    prompts = bundle / "context/prompts.yaml"
    try:
        prompt_bytes = prompts.read_bytes()
        if not prompt_bytes or b"\x00" in prompt_bytes:
            raise ValueError("prompt snapshot is empty or contains NUL bytes")
        prompt_bytes.decode("utf-8")
    except Exception as exc:
        raise FinalizationError(f"Frozen prompts are invalid: {exc}") from exc

    grounding = bundle / "context/grounding.txt"
    try:
        grounding_bytes = grounding.read_bytes()
        if not grounding_bytes or b"\x00" in grounding_bytes:
            raise ValueError("grounding snapshot is empty or contains NUL bytes")
        grounding_bytes.decode("utf-8")
    except Exception as exc:
        raise FinalizationError(f"Frozen grounding is invalid: {exc}") from exc

    releases = bundle / "context/model-releases-before.yaml"
    try:
        release_bytes = releases.read_bytes()
        if not release_bytes or b"\x00" in release_bytes:
            raise ValueError("model release snapshot is empty")
        release_bytes.decode("utf-8")
    except Exception as exc:
        raise FinalizationError(f"Frozen model releases are invalid: {exc}") from exc

    try:
        effective = read_json(bundle / "context/effective-config.json")
    except Exception as exc:
        raise FinalizationError(f"Frozen effective config is invalid: {exc}") from exc
    if (
        not isinstance(effective, dict)
        or effective.get("schema_version") != SCHEMA_VERSION
        or effective.get("valid") is not True
        or effective.get("issues") not in ([], None)
    ):
        raise FinalizationError("Frozen effective config has an unsupported schema")
    values = effective.get("values")
    if not isinstance(values, dict) or not values:
        raise FinalizationError("Frozen effective config has no values")
    if not isinstance(values.get("llm"), dict) or not values["llm"]:
        raise FinalizationError("Frozen effective config has no effective LLM settings")
    pipeline = values.get("pipeline")
    if not isinstance(pipeline, dict):
        raise FinalizationError("Frozen effective config has no pipeline settings")
    pipeline_url = pipeline.get("base_url")
    parsed_pipeline_url = urlparse(pipeline_url) if isinstance(pipeline_url, str) else None
    if (
        parsed_pipeline_url is None
        or parsed_pipeline_url.scheme.lower() != "https"
        or not parsed_pipeline_url.hostname
        or parsed_pipeline_url.username
        or parsed_pipeline_url.password
        or parsed_pipeline_url.query
        or parsed_pipeline_url.fragment
    ):
        raise FinalizationError("Frozen pipeline base_url must be an HTTPS origin without credentials/query")
    lookback_hours = pipeline.get("lookback_hours")
    if isinstance(lookback_hours, bool) or not isinstance(lookback_hours, int) or not 1 <= lookback_hours <= 168:
        raise FinalizationError("Frozen pipeline lookback_hours is outside 1-168")

    try:
        runtime = read_json(bundle / "context/runtime-settings.json")
    except Exception as exc:
        raise FinalizationError(f"Frozen runtime settings are invalid: {exc}") from exc
    if not isinstance(runtime, dict) or set(runtime) != set(RUNTIME_ENV_KEYS):
        raise FinalizationError("Frozen runtime settings do not match the allowlist")
    for key, value in runtime.items():
        if value is not None and (
            not isinstance(value, str) or "\x00" in value or "\r" in value or "\n" in value
        ):
            raise FinalizationError(f"Frozen runtime setting {key} is not a string or null")

    try:
        history = read_json(bundle / "context/history-manifest.json")
    except Exception as exc:
        raise FinalizationError(f"Frozen history manifest is invalid: {exc}") from exc
    if not isinstance(history, dict) or history.get("schema_version") != SCHEMA_VERSION:
        raise FinalizationError("Frozen history manifest has an unsupported schema")
    history_date = history.get("report_date")
    if history_date is None or not DATE_RE.fullmatch(str(history_date)):
        raise FinalizationError("Frozen history manifest has no report date")
    if history.get("lookback_days") != 45:
        raise FinalizationError("Frozen history manifest must cover exactly 45 days")
    try:
        from datetime import date, timedelta

        target = date.fromisoformat(str(history_date))
        expected_dates = {(target - timedelta(days=offset)).isoformat() for offset in range(1, 46)}
    except ValueError as exc:
        raise FinalizationError("Frozen history manifest report date is invalid") from exc
    if set(history.get("dates") or []) != expected_dates:
        raise FinalizationError("Frozen history manifest does not enumerate all 45 prior dates")
    used = history.get("used")
    missing_rows = history.get("missing")
    not_used = history.get("not_used") or []
    if not isinstance(used, list) or not isinstance(missing_rows, list) or not isinstance(not_used, list):
        raise FinalizationError("Frozen history manifest lacks explicit used/missing paths")
    used_hashes = history.get("used_hashes")
    if not isinstance(used_hashes, dict):
        raise FinalizationError("Frozen history manifest lacks used path hashes")
    used_norm: set[str] = set()
    for relative in used:
        if not isinstance(relative, str) or not relative.startswith("history/"):
            raise FinalizationError("Frozen history used path is invalid")
        try:
            path = bundle_path(bundle, relative)
        except Exception as exc:
            raise FinalizationError(f"Frozen history used path is invalid: {relative}") from exc
        if not path.is_file() or path.is_symlink():
            raise FinalizationError(f"Frozen history used path is missing: {relative}")
        expected_hash = used_hashes.get(relative)
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise FinalizationError(f"Frozen history used path lacks a valid hash: {relative}")
        if hash_file(path) != expected_hash:
            raise FinalizationError(f"Frozen history used path hash mismatch: {relative}")
        used_norm.add(relative.removeprefix("history/"))
    declared_missing = {str(value) for value in missing_rows}
    declared_not_used = {str(value) for value in not_used}
    expected_paths = {
        f"{day}/{category}.json"
        for day in expected_dates
        for category in CATEGORIES
    }
    if not expected_paths.issubset(used_norm | declared_missing | declared_not_used):
        raise FinalizationError("Frozen history manifest silently omits a prior-day path")
    if history.get("search_documents_used"):
        if "search-documents.json" not in used_norm and "search-documents.json" not in declared_missing:
            raise FinalizationError("Frozen history manifest claims search documents without an explicit path")
    elif "search-documents.json" not in declared_not_used and "search-documents.json" not in declared_missing:
        raise FinalizationError("Frozen history manifest does not record search-document semantics")

    try:
        evidence = read_json(bundle / "evidence/index.json")
    except Exception as exc:
        raise FinalizationError(f"Frozen evidence index is invalid: {exc}") from exc
    if not isinstance(evidence, dict) or evidence.get("schema_version") != SCHEMA_VERSION:
        raise FinalizationError("Frozen evidence index has an unsupported schema")
    entries = evidence.get("entries")
    if not isinstance(entries, dict):
        raise FinalizationError("Frozen evidence index has no entries map")
    for url, row in entries.items():
        if not isinstance(url, str) or not isinstance(row, dict):
            raise FinalizationError("Frozen evidence entry is malformed")
        kind = row.get("kind")
        if kind == "failure":
            if not isinstance(row.get("error"), str):
                raise FinalizationError("Frozen evidence failure has no redacted reason")
            continue
        if kind != "response" or not isinstance(row.get("object"), str):
            raise FinalizationError("Frozen evidence response is malformed")
        object_path = bundle_path(bundle, f"evidence/objects/{row['object']}.json")
        if not object_path.is_file() or object_path.is_symlink():
            raise FinalizationError(f"Frozen evidence object is missing: {row['object']}")
        try:
            object_value = read_json(object_path)
            body = base64.b64decode(object_value.get("body_b64", ""), validate=True)
            if hashlib.sha256(body).hexdigest() != row["object"]:
                raise ValueError("body hash mismatch")
        except Exception as exc:
            raise FinalizationError(f"Frozen evidence object is invalid: {row['object']}") from exc

    try:
        precontinuity = read_json(bundle / "checkpoints/analysis_pre_continuity.json")
    except Exception as exc:
        raise FinalizationError(f"Frozen pre-continuity checkpoint is invalid: {exc}") from exc
    if (
        not isinstance(precontinuity, dict)
        or precontinuity.get("schema_version") != SCHEMA_VERSION
        or not isinstance(precontinuity.get("reports"), dict)
        or not precontinuity["reports"]
    ):
        raise FinalizationError("Frozen pre-continuity checkpoint has no reports")


def finalize_bundle(
    bundle_root: str | Path,
    *,
    run_id: int | str,
    run_attempt: int | str,
    execution_sha: str,
    output_commit: str,
    report_date: str,
    event_sha: str | None = None,
    repository: str = "flyryan/ai-news-aggregator",
    workflow_path: str = ".github/workflows/daily-pipeline.yml",
    output_dir: str | Path | None = None,
    healthy: bool | None = None,
    pipeline_replay: bool = False,
    missing_dependencies: list[Any] | None = None,
    versions: Mapping[str, Any] | None = None,
    expected_output_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Finalize and seal one immutable bundle.

    All source and publication lineage values are supplied by the trusted
    caller after its workflow steps.  The function never guesses an execution
    SHA from the event SHA and never treats current ``main`` as publication
    proof.
    """
    bundle = Path(bundle_root)
    if bundle.is_symlink():
        raise FinalizationError("Bundle root may not be a symlink")
    bundle = bundle.resolve()
    if (bundle / "manifest.json").exists():
        raise FinalizationError("Bundle is already sealed; use a new output directory")
    if not DATE_RE.fullmatch(str(report_date)):
        raise FinalizationError("report_date must use YYYY-MM-DD")
    try:
        from datetime import date

        if date.fromisoformat(report_date).isoformat() != report_date:
            raise ValueError
    except ValueError as exc:
        raise FinalizationError("report_date is invalid") from exc
    run_id = _positive(run_id, "run_id")
    run_attempt = _positive(run_attempt, "run_attempt")
    execution_sha = _full_sha(execution_sha, "execution_sha")  # type: ignore[assignment]
    output_commit = _full_sha(output_commit, "output_commit")  # type: ignore[assignment]
    event_sha = _full_sha(event_sha, "event_sha", required=False)
    if not bundle.is_dir():
        raise FinalizationError("Bundle directory does not exist")

    input_manifest, input_manifest_sha = _read_input_manifest(bundle)
    source = input_manifest.get("source") or {}
    # Existing metadata can corroborate the caller's values, but cannot replace
    # them.  A mismatch means the capture and publication lineage are different.
    if source.get("run_id") is not None and int(source["run_id"]) != run_id:
        raise FinalizationError("source run_id disagrees with input-manifest")
    if source.get("run_attempt") is not None and int(source["run_attempt"]) != run_attempt:
        raise FinalizationError("source run_attempt disagrees with input-manifest")
    if source.get("execution_sha") and source["execution_sha"] != execution_sha:
        raise FinalizationError("execution_sha disagrees with input-manifest")
    if source.get("event_sha") and event_sha and source["event_sha"] != event_sha:
        raise FinalizationError("event_sha disagrees with input-manifest")
    if input_manifest.get("report_date") and input_manifest["report_date"] != report_date:
        raise FinalizationError("report_date disagrees with input-manifest")

    relevance_ok, relevance_missing, inputs = _check_relevance(bundle)
    files, output_hashes = _output_bytes(bundle, output_dir, report_date)
    if expected_output_hashes is None:
        raise FinalizationError(
            "All five post-push output hashes are required to prove publication"
        )
    if set(expected_output_hashes) != set(OUTPUT_NAMES):
        raise FinalizationError("Exactly five post-push output hashes are required")
    for name in OUTPUT_NAMES:
        expected = expected_output_hashes.get(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise FinalizationError(f"Output hash is invalid for {name}")
        if output_hashes.get(name) != expected:
            raise FinalizationError(f"Output hash mismatch for {name}")
    if pipeline_replay:
        _require_pipeline_dependencies(bundle)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "published",
        "verification": "postpush_commit_and_file_sha256",
        "repository": repository,
        "workflow_path": workflow_path,
        "report_date": report_date,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "event_sha": event_sha,
        "execution_sha": execution_sha,
        "output_commit": output_commit,
        "original_publication_verified": True,
        "input_manifest_sha256": input_manifest_sha,
        "files": output_hashes,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    publication_path = bundle / "publication.json"
    if publication_path.exists():
        raise FinalizationError("publication.json already exists; bundle is not mutable")
    write_json(publication_path, receipt)

    missing = list(missing_dependencies or [])
    if relevance_missing:
        missing.append({"reason": relevance_missing, "key": "relevance/input.json"})
    healthy_value = _health(bundle, healthy)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": repository,
            "workflow_path": workflow_path,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "event_sha": event_sha,
            "execution_sha": execution_sha,
            "output_commit": output_commit,
        },
        "report_date": report_date,
        "coverage": input_manifest.get("coverage") or {},
        "capabilities": {
            "filter_replay": bool(relevance_ok),
            "pipeline_replay": bool(pipeline_replay),
        },
        "capability": "pipeline_replay" if pipeline_replay else ("filter_replay" if relevance_ok else "output_review"),
        "missing_dependencies": missing,
        "versions": dict(versions or {}),
        "source_health": {"healthy": healthy_value},
        "eligibility": {
            "healthy": healthy_value,
            "successful_run": True,
            "published": True,
            "original_publication_verified": True,
        },
        "publication": {
            "status": "published",
            "receipt_sha256": hash_file(publication_path),
            "output_commit": output_commit,
            "files": output_hashes,
        },
    }
    try:
        return seal_bundle(bundle, manifest)
    except Exception as exc:
        # Leave the receipt visible for diagnosis, but do not manufacture a
        # manifest or claim publication if the final inventory could not be
        # sealed.
        raise FinalizationError(f"Bundle integrity sealing failed: {exc}") from exc


def _parse_hashes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise FinalizationError("--output-hash must use NAME=SHA256")
        name, digest = value.split("=", 1)
        if name not in OUTPUT_NAMES or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FinalizationError("--output-hash contains an invalid file or SHA256")
        result[name] = digest
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--output-commit", required=True)
    parser.add_argument("--event-sha")
    parser.add_argument("--repository", default="flyryan/ai-news-aggregator")
    parser.add_argument("--workflow-path", default=".github/workflows/daily-pipeline.yml")
    parser.add_argument("--output-dir")
    parser.add_argument("--healthy", action="store_true", default=None)
    parser.add_argument("--unhealthy", action="store_false", dest="healthy")
    parser.add_argument("--pipeline-replay", action="store_true")
    parser.add_argument("--output-hash", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = finalize_bundle(
            args.bundle_root,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            report_date=args.report_date,
            execution_sha=args.execution_sha,
            output_commit=args.output_commit,
            event_sha=args.event_sha,
            repository=args.repository,
            workflow_path=args.workflow_path,
            output_dir=args.output_dir,
            healthy=args.healthy,
            pipeline_replay=args.pipeline_replay,
            expected_output_hashes=_parse_hashes(args.output_hash),
        )
    except Exception as exc:
        print(f"finalize_bundle: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"sealed": True, "bundle_sha256": manifest.get("bundle_sha256"), "path": str(Path(args.bundle_root).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
