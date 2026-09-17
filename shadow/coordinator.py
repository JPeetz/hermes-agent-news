"""Bounded discovery and scheduling for the internal news shadow experiment.

This module intentionally stops at metadata and state management.  It never
executes a downloaded production artifact and it never invokes a provider.
The evaluator (``scripts/shadow/run.py`` and its model adapters) owns those
operations after this coordinator has selected and recorded a verified bundle.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import BundleValidationError, read_json, verify_bundle
from .state import (
    CoordinatorState,
    ExperimentIdentity,
    StateError,
    StateStore,
    STATUS_COMPLETED,
    STATUS_DISCOVERED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    STATUS_RUNNING,
    STATUS_UNAVAILABLE,
    experiment_id,
    identity_from_mapping,
)


INTERNAL_REPOSITORY = "trend-ai-acceleration-task-force/ai-news-aggregator"
SOURCE_REPOSITORY = "flyryan/ai-news-aggregator"
SOURCE_WORKFLOW_PATH = ".github/workflows/daily-pipeline.yml"
TRUSTED_EVALUATOR_REF = "main"
MAX_DATE_RANGE = 14
MAX_COORDINATOR_BATCH = 2
MAX_PROSPECTIVE_DATES = 7
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RUN_RE = re.compile(r"^(?P<run>[1-9]\d*)(?::(?P<attempt>[1-9]\d*))?$" )


class SelectorError(ValueError):
    """A manual selector is ambiguous, too broad, or malformed."""


class TrustedRefError(ValueError):
    """A workflow attempted to evaluate code from an untrusted ref."""


class DiscoveryError(ValueError):
    """Source metadata is insufficient for a safe discovery record."""


_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROUTES = frozenset(
    {
        "https://api.rdsec.trendmicro.com/prod/aiendpoint/v1",
        "https://openrouter.ai/api/v1",
    }
)


def _validated_hash(value: Any, field: str, pattern: re.Pattern[str]) -> str | None:
    """Validate an identity hash before it can affect scheduling/deduplication."""

    if value in (None, ""):
        return None
    if not isinstance(value, str) or pattern.fullmatch(value.lower()) is None:
        raise DiscoveryError(f"{field} must be a lowercase hexadecimal hash")
    return value.lower()


def _evaluator_sha(value: Any = None, manifest: Mapping[str, Any] | None = None) -> str | None:
    """Resolve only the trusted evaluator SHA for the replay identity.

    ``source.execution_sha`` identifies the production checkout and is
    provenance only.  It must never silently stand in for the evaluator code
    that actually runs the comparison.
    """

    candidate = value
    if candidate in (None, "") and isinstance(manifest, Mapping):
        candidate = manifest.get("evaluator_sha")
    if candidate in (None, ""):
        candidate = os.environ.get("SHADOW_EVALUATOR_SHA")
    return _validated_hash(candidate, "evaluator_sha", _SHA40_RE)


def _validated_route(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or value.rstrip("/") not in _ALLOWED_ROUTES:
        raise DiscoveryError("incumbent_route is outside the approved model endpoints")
    return value.rstrip("/")


@dataclass(frozen=True, order=True)
class RunAttempt:
    run_id: int
    attempt: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, int) or self.run_id <= 0:
            raise SelectorError("run_id must be a positive integer")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt <= 0:
            raise SelectorError("attempt must be a positive integer")

    @property
    def key(self) -> str:
        return f"{self.run_id}:{self.attempt}"

    def as_dict(self) -> dict[str, int]:
        return {"run_id": self.run_id, "attempt": self.attempt}


@dataclass(frozen=True)
class Selector:
    """Exactly one source-run or inclusive report-date selector."""

    source_runs: tuple[RunAttempt, ...] = ()
    from_date: date | None = None
    to_date: date | None = None

    def __post_init__(self) -> None:
        if self.source_runs and (self.from_date is not None or self.to_date is not None):
            raise SelectorError("source_runs cannot be combined with a date range")
        if (self.from_date is None) != (self.to_date is None):
            raise SelectorError("from_date and to_date must be supplied together")
        if self.from_date is not None and self.to_date is not None:
            if self.to_date < self.from_date:
                raise SelectorError("to_date must not precede from_date")
            if (self.to_date - self.from_date).days + 1 > MAX_DATE_RANGE:
                raise SelectorError(f"date range may contain at most {MAX_DATE_RANGE} dates")
        if len(self.source_runs) > MAX_DATE_RANGE:
            raise SelectorError(f"source_runs may contain at most {MAX_DATE_RANGE} runs")
        if len(set(self.source_runs)) != len(self.source_runs):
            raise SelectorError("duplicate source run/attempt selector")

    @property
    def kind(self) -> str:
        if self.source_runs:
            return "source_runs"
        if self.from_date is not None:
            return "date_range"
        return "all"

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        run = run_attempt_from_metadata(metadata, require=False)
        if self.source_runs:
            return run in set(self.source_runs)
        if self.from_date is not None:
            report_date = explicit_report_date(metadata)
            return report_date is not None and self.from_date <= report_date <= self.to_date
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_runs": [run.as_dict() for run in self.source_runs],
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
        }


@dataclass(frozen=True)
class SourceClassification:
    run: RunAttempt
    workflow_path: str | None
    actual_pipeline: bool
    completed_successfully: bool
    source_healthy: bool
    publication_verified: bool
    metadata_eligible: bool
    publication_eligible: bool
    eligible: bool
    classification: str
    reasons: tuple[str, ...]
    report_date: str | None = None
    publication_status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run.run_id,
            "attempt": self.run.attempt,
            "workflow_path": self.workflow_path,
            "actual_pipeline": self.actual_pipeline,
            "completed_successfully": self.completed_successfully,
            "source_healthy": self.source_healthy,
            "publication_verified": self.publication_verified,
            "metadata_eligible": self.metadata_eligible,
            "publication_eligible": self.publication_eligible,
            "eligible": self.eligible,
            "classification": self.classification,
            "reasons": list(self.reasons),
            "report_date": self.report_date,
            "publication_status": self.publication_status,
        }


def _first(value: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _nested(value: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = value
        ok = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "healthy", "success", "published", "verified"}:
            return True
        if lowered in {"false", "no", "failed", "partial", "unhealthy", "reverted"}:
            return False
    return None


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise SelectorError(f"{field} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SelectorError(f"{field} is not a calendar date") from exc
    if parsed.isoformat() != value:
        raise SelectorError(f"{field} must be canonical YYYY-MM-DD")
    return parsed


def parse_run_attempts(value: str | Sequence[str] | Sequence[Mapping[str, Any]] | None) -> tuple[RunAttempt, ...]:
    """Parse ``run[:attempt]`` values while preserving attempt identity."""

    if value is None or value == "" or value == []:
        return ()
    raw_values: list[Any]
    if isinstance(value, str):
        raw_values = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    else:
        raw_values = list(value)
    result: list[RunAttempt] = []
    for raw in raw_values:
        if isinstance(raw, Mapping):
            try:
                run_id = int(raw["run_id"])
                attempt = int(raw.get("attempt", 1))
            except (KeyError, TypeError, ValueError) as exc:
                raise SelectorError("source run objects require run_id and attempt") from exc
            result.append(RunAttempt(run_id, attempt))
            continue
        if not isinstance(raw, str):
            raise SelectorError("source_runs must contain run:attempt strings")
        match = RUN_RE.fullmatch(raw.strip())
        if not match:
            raise SelectorError(f"Invalid source run selector: {raw!r}")
        result.append(RunAttempt(int(match.group("run")), int(match.group("attempt") or 1)))
    parsed = tuple(result)
    Selector(source_runs=parsed)  # validate duplicates and the 14-item bound
    return parsed


def parse_selector(
    *,
    source_runs: str | Sequence[str] | Sequence[Mapping[str, Any]] | None = None,
    from_date: str | date | None = None,
    to_date: str | date | None = None,
) -> Selector:
    runs = parse_run_attempts(source_runs)
    if isinstance(from_date, date) and not isinstance(from_date, str):
        start = from_date
    elif from_date is None or from_date == "":
        start = None
    else:
        start = _parse_date(from_date, "from_date")
    if isinstance(to_date, date) and not isinstance(to_date, str):
        end = to_date
    elif to_date is None or to_date == "":
        end = None
    else:
        end = _parse_date(to_date, "to_date")
    return Selector(source_runs=runs, from_date=start, to_date=end)


def validate_trusted_ref(ref: str | None) -> str:
    if ref in (None, ""):
        return TRUSTED_EVALUATOR_REF
    if ref not in {TRUSTED_EVALUATOR_REF, "refs/heads/main"}:
        raise TrustedRefError("shadow evaluation is allowed only from trusted main")
    return TRUSTED_EVALUATOR_REF


def run_attempt_from_metadata(metadata: Mapping[str, Any], *, require: bool = True) -> RunAttempt | None:
    run_id = _first(metadata, "run_id", "id")
    attempt = _first(metadata, "run_attempt", "attempt")
    try:
        if run_id is None or isinstance(run_id, bool) or isinstance(attempt, bool):
            raise ValueError()
        run_value = int(run_id)
        attempt_value = int(attempt if attempt is not None else 1)
        return RunAttempt(run_value, attempt_value)
    except (TypeError, ValueError, SelectorError) as exc:
        if require:
            raise DiscoveryError("source metadata lacks a valid run_id/run_attempt") from exc
        return None


def explicit_report_date(metadata: Mapping[str, Any]) -> date | None:
    # Never fall back to created_at/run_started_at.  A workflow run's calendar
    # day and the report date can differ, and the importer is responsible for
    # proving the latter from bundle evidence.
    value = _first(metadata, "report_date", "target_date")
    if value is None:
        value = _nested(metadata, "inputs.target_date", "inputs.report_date", "source.report_date")
    if value is None:
        return None
    try:
        return _parse_date(value, "report_date")
    except SelectorError:
        return None


def classify_source_run(
    metadata: Mapping[str, Any], *, expected_workflow: str = SOURCE_WORKFLOW_PATH
) -> SourceClassification:
    """Classify metadata conservatively without claiming importer eligibility."""

    if not isinstance(metadata, Mapping):
        raise DiscoveryError("source run metadata must be an object")
    run = run_attempt_from_metadata(metadata)
    workflow_path = _first(metadata, "workflow_path", "path")
    if workflow_path is None:
        workflow = metadata.get("workflow")
        workflow_path = workflow.get("path") if isinstance(workflow, Mapping) else None
    if isinstance(workflow_path, str):
        workflow_path = workflow_path.split("@", 1)[0]
    actual_pipeline = workflow_path == expected_workflow

    status = str(_first(metadata, "status", "run_status") or "").lower()
    conclusion = str(_first(metadata, "conclusion", "result") or "").lower()
    completed_successfully = status == "completed" and conclusion == "success"

    health_value = _first(metadata, "source_healthy", "healthy")
    if health_value is None:
        health_value = _nested(metadata, "source_health.overall", "collection_status.overall", "health.overall")
    source_healthy = _as_bool(health_value) is True

    publication_value = _first(metadata, "publication_verified", "published")
    publication_status_value = _nested(metadata, "publication.status", "publication.receipt_status")
    if publication_status_value is not None:
        publication_status = str(publication_status_value or "").lower()
    else:
        publication_status = str(publication_value or "").lower()
    if publication_value is None:
        publication_value = publication_status_value
    original_publication_verified = _first(metadata, "original_publication_verified")
    if original_publication_verified is None:
        original_publication_verified = _nested(metadata, "publication.original_publication_verified")
    original_publication_verified = _as_bool(original_publication_verified) is True
    publication_verified = (
        _as_bool(publication_value) is True
        or publication_status in {"published", "verified", "success"}
        or publication_status == "superseded" and original_publication_verified
    )

    reasons: list[str] = []
    if not actual_pipeline:
        reasons.append("workflow_path_mismatch")
    if status != "completed":
        reasons.append("run_not_completed")
    if conclusion != "success":
        reasons.append("run_conclusion_not_success")
    if not source_healthy:
        reasons.append("source_health_unverified_or_degraded")
    if not publication_verified:
        reasons.append("publication_unverified")
    metadata_eligible = actual_pipeline and completed_successfully and source_healthy
    publication_eligible = metadata_eligible and publication_verified
    eligible = metadata_eligible and publication_verified

    if eligible:
        classification = "eligible"
    elif not actual_pipeline or not completed_successfully:
        classification = "failed_or_non_pipeline"
    elif not source_healthy:
        classification = "degraded_source"
    elif not publication_verified:
        classification = "unpublished_or_unverified"
    else:  # defensive fallback for future metadata fields
        classification = "metadata_only"

    report = explicit_report_date(metadata)
    return SourceClassification(
        run=run,
        workflow_path=workflow_path,
        actual_pipeline=actual_pipeline,
        completed_successfully=completed_successfully,
        source_healthy=source_healthy,
        publication_verified=publication_verified,
        metadata_eligible=metadata_eligible,
        publication_eligible=publication_eligible,
        eligible=eligible,
        classification=classification,
        reasons=tuple(reasons),
        report_date=report.isoformat() if report else None,
        publication_status=publication_status or None,
    )


def _flatten_metadata(metadata: Any) -> list[Mapping[str, Any]]:
    if isinstance(metadata, Mapping):
        for key in ("runs", "workflow_runs", "items", "results"):
            value = metadata.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, Mapping)]
        return [metadata]
    if isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes, bytearray)):
        flattened: list[Mapping[str, Any]] = []
        for page in metadata:
            flattened.extend(_flatten_metadata(page))
        return flattened
    raise DiscoveryError("metadata inventory must be an object or list")


def discover_metadata(
    metadata: Any,
    selector: Selector | None = None,
    *,
    expected_workflow: str = SOURCE_WORKFLOW_PATH,
) -> list[dict[str, Any]]:
    """Return structured, metadata-only discovery candidates.

    This function does not infer health/publication from a run's conclusion
    and does not turn a creation timestamp into a report date.  The importer
    must later verify artifact contents and publication bytes.
    """

    selector = selector or Selector()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in _flatten_metadata(metadata):
        classification = classify_source_run(raw, expected_workflow=expected_workflow)
        key = (classification.run.run_id, classification.run.attempt)
        if key in seen or not selector.matches(raw):
            continue
        seen.add(key)
        candidates.append(
            {
                "source": copy.deepcopy(dict(raw)),
                "classification": classification.as_dict(),
                "source_key": f"{SOURCE_REPOSITORY}#{classification.run.run_id}:{classification.run.attempt}",
                "report_date": classification.report_date,
                "run_id": classification.run.run_id,
                "attempt": classification.run.attempt,
            }
        )
    candidates.sort(key=lambda row: (row.get("report_date") or "9999-99-99", row["run_id"], row["attempt"]))
    return candidates


def identity_for_discovery(
    candidate: Mapping[str, Any],
    *,
    mode: str = "filter",
    policy_version: str | None = None,
    rubric_version: str | None = None,
    candidate_model: str | None = None,
    judge_version: str | None = None,
    judge_model: str | None = None,
    sampling_plan: str | None = None,
    repeat_plan: str | None = None,
    evaluator_sha: str | None = None,
    policy_sha256: str | None = None,
    cohort: str | None = None,
    cohort_sha256: str | None = None,
    incumbent_route: str | None = None,
    incumbent_model: str | None = None,
) -> ExperimentIdentity:
    classification = candidate.get("classification") if isinstance(candidate, Mapping) else None
    source = candidate.get("source") if isinstance(candidate, Mapping) else None
    source = source if isinstance(source, Mapping) else candidate
    run = candidate.get("run_id", source.get("run_id", source.get("id")))
    attempt = candidate.get("attempt", source.get("run_attempt", source.get("attempt", 1)))
    return identity_from_mapping(
        {
            "repository": SOURCE_REPOSITORY,
            "source_run_id": run,
            "source_attempt": attempt,
            "bundle_sha256": candidate.get("bundle_sha256", source.get("bundle_sha256")),
            # Metadata-only discovery may not know the evaluator yet.  If it
            # does, bind the trusted evaluator SHA; never use the production
            # execution SHA as a replay identity.
            "replay_sha": _evaluator_sha(
                evaluator_sha or candidate.get("evaluator_sha"), candidate
            ),
            "dependency_lock": candidate.get("dependency_lock", source.get("dependency_lock")),
            "container_digest": candidate.get("container_digest", source.get("container_digest")),
            "mode": mode,
            "candidate_policy": policy_version,
            "rubric_version": rubric_version,
            "candidate_model": candidate_model,
            "judge_version": judge_version,
            "judge_model": judge_model,
            "sampling_plan": sampling_plan,
            "repeat_plan": repeat_plan,
            "policy_sha256": _validated_hash(policy_sha256, "policy_sha256", _SHA64_RE),
            "cohort": cohort,
            "cohort_sha256": _validated_hash(cohort_sha256, "cohort_sha256", _SHA64_RE),
            "incumbent_route": _validated_route(incumbent_route),
            "incumbent_model": incumbent_model,
        }
    )


def identity_for_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str | None = None,
    policy_version: str | None = None,
    rubric_version: str | None = None,
    candidate_model: str | None = None,
    judge_version: str | None = None,
    judge_model: str | None = None,
    sampling_plan: str | None = None,
    repeat_plan: str | None = None,
    evaluator_sha: str | None = None,
    policy_sha256: str | None = None,
    cohort: str | None = None,
    cohort_sha256: str | None = None,
    incumbent_route: str | None = None,
    incumbent_model: str | None = None,
) -> ExperimentIdentity:
    if not isinstance(manifest, Mapping):
        raise DiscoveryError("bundle manifest must be an object")
    source = manifest.get("source")
    source = source if isinstance(source, Mapping) else manifest
    repository = source.get("repository", SOURCE_REPOSITORY)
    if repository != SOURCE_REPOSITORY:
        raise DiscoveryError("bundle source repository is outside the approved source")
    resolved_evaluator_sha = _evaluator_sha(evaluator_sha, manifest)
    return identity_from_mapping(
        {
            "repository": repository,
            "source_run_id": source.get("run_id", source.get("source_run_id")),
            "source_attempt": source.get("run_attempt", source.get("source_attempt", 1)),
            "bundle_sha256": manifest.get("bundle_sha256"),
            # The source execution SHA is provenance and permission context,
            # never the code version used by this evaluator.
            "replay_sha": resolved_evaluator_sha,
            "dependency_lock": manifest.get("dependency_lock", manifest.get("dependency_lock_hash")),
            "container_digest": manifest.get("container_digest"),
            "mode": mode or manifest.get("mode", "filter"),
            "candidate_policy": policy_version or manifest.get("candidate_policy", manifest.get("policy_version")),
            "rubric_version": rubric_version or manifest.get("rubric_version"),
            "candidate_model": candidate_model or manifest.get("candidate_model"),
            "judge_version": judge_version or manifest.get("judge_version"),
            "judge_model": judge_model or manifest.get("judge_model"),
            "sampling_plan": sampling_plan or manifest.get("sampling_plan"),
            "repeat_plan": repeat_plan or manifest.get("repeat_plan"),
            "policy_sha256": _validated_hash(
                policy_sha256 or manifest.get("policy_sha256"), "policy_sha256", _SHA64_RE
            ),
            "cohort": cohort or manifest.get("cohort"),
            "cohort_sha256": _validated_hash(
                cohort_sha256 or manifest.get("cohort_sha256"), "cohort_sha256", _SHA64_RE
            ),
            "incumbent_route": _validated_route(
                incumbent_route or manifest.get("incumbent_route")
            ),
            "incumbent_model": incumbent_model or manifest.get("incumbent_model"),
        }
    )


class Coordinator:
    """Small facade shared by manual and scheduled workflow entry points."""

    def __init__(self, state: StateStore | CoordinatorState | str | Path):
        self.state = state if isinstance(state, StateStore) else StateStore(state)

    def discover(
        self,
        metadata: Any,
        *,
        selector: Selector | None = None,
        mode: str = "filter",
        policy_version: str | None = None,
        rubric_version: str | None = None,
        candidate_model: str | None = None,
        judge_version: str | None = None,
        judge_model: str | None = None,
        sampling_plan: str | None = None,
        repeat_plan: str | None = None,
        evaluator_sha: str | None = None,
        policy_sha256: str | None = None,
        cohort: str | None = None,
        cohort_sha256: str | None = None,
        incumbent_route: str | None = None,
        incumbent_model: str | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"filter", "pipeline"}:
            raise SelectorError("mode must be filter or pipeline")
        discovered = []
        for candidate in discover_metadata(metadata, selector):
            identity = identity_for_discovery(
                candidate,
                mode=mode,
                policy_version=policy_version,
                rubric_version=rubric_version,
                candidate_model=candidate_model,
                judge_version=judge_version,
                judge_model=judge_model,
                sampling_plan=sampling_plan,
                repeat_plan=repeat_plan,
                evaluator_sha=evaluator_sha,
                policy_sha256=policy_sha256,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
                incumbent_route=incumbent_route,
                incumbent_model=incumbent_model,
            )
            classification = candidate["classification"]
            row = self.state.ensure_discovered(
                identity,
                source={
                    "repository": SOURCE_REPOSITORY,
                    "run_id": candidate["run_id"],
                    "attempt": candidate["attempt"],
                    "report_date": candidate.get("report_date"),
                    "metadata_only": True,
                },
                eligibility={
                    "metadata_eligible": classification["metadata_eligible"],
                    "healthy": classification["source_healthy"],
                    "publication_verified": classification["publication_verified"],
                    "classification": classification["classification"],
                    "reasons": classification["reasons"],
                    "importer_verified": False,
                    "auto_eligible": False,
                },
                capabilities={
                    "filter_replay": False,
                    "pipeline_replay": False,
                },
            )
            discovered.append(row)
        return discovered

    def record_bundle(
        self,
        bundle: str | Path,
        *,
        mode: str = "filter",
        status: str = STATUS_DISCOVERED,
        policy_version: str | None = None,
        rubric_version: str | None = None,
        candidate_model: str | None = None,
        judge_version: str | None = None,
        judge_model: str | None = None,
        sampling_plan: str | None = None,
        repeat_plan: str | None = None,
        evaluator_sha: str | None = None,
        policy_sha256: str | None = None,
        cohort: str | None = None,
        cohort_sha256: str | None = None,
        incumbent_route: str | None = None,
        incumbent_model: str | None = None,
        result_experiment_id: str | None = None,
        result_path: str | None = None,
        require_healthy: bool = False,
        require_published: bool = False,
    ) -> dict[str, Any]:
        if mode not in {"filter", "pipeline"}:
            raise SelectorError("mode must be filter or pipeline")
        path = Path(bundle)
        try:
            manifest = verify_bundle(
                path,
                capability="pipeline_replay" if mode == "pipeline" else "filter_replay",
                require_healthy=require_healthy,
                require_published=require_published,
            )
        except (OSError, BundleValidationError, KeyError, TypeError, ValueError) as exc:
            raise DiscoveryError(f"bundle verification failed: {exc}") from exc
        identity = identity_for_manifest(
            manifest,
            mode=mode,
            policy_version=policy_version,
            rubric_version=rubric_version,
            candidate_model=candidate_model,
            judge_version=judge_version,
            judge_model=judge_model,
            sampling_plan=sampling_plan,
            repeat_plan=repeat_plan,
            evaluator_sha=evaluator_sha,
            policy_sha256=policy_sha256,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            incumbent_route=incumbent_route,
            incumbent_model=incumbent_model,
        )
        source = manifest.get("source") if isinstance(manifest.get("source"), Mapping) else {}
        eligibility = manifest.get("eligibility") if isinstance(manifest.get("eligibility"), Mapping) else {}
        capabilities = manifest.get("capabilities") if isinstance(manifest.get("capabilities"), Mapping) else {}
        publication = manifest.get("publication") if isinstance(manifest.get("publication"), Mapping) else {}
        # Normalize importer vocabulary to the scheduler's explicit gates.  A
        # manifest may contain only ``successful_run`` and publication.status;
        # the queue still requires both health and publication proof.
        normalized_eligibility = copy.deepcopy(dict(eligibility))
        normalized_eligibility.setdefault(
            "metadata_eligible",
            normalized_eligibility.get("successful_run") is True
            and normalized_eligibility.get("healthy") is True,
        )
        recovery = cohort == "recovery"
        if recovery:
            # Recovery is an explicit operational cohort. It can preserve a
            # failed or degraded source run with verified replay inputs;
            # ordinary cohorts remain behind the healthy-source gate.
            normalized_eligibility["metadata_eligible"] = True
        publication_status = str(publication.get("status") or "").lower()
        original_publication_verified = (
            publication.get("original_publication_verified") is True
            or normalized_eligibility.get("original_publication_verified") is True
        )
        # A superseded date remains valid when its original publication was
        # verified against the immutable receipt. Reverted/recovery records
        # are kept in the index but are auto-queued only by an explicit
        # recovery cohort.
        publication_verified = (
            publication_status in {"published", "verified", "success"}
            or publication_status == "superseded" and original_publication_verified
        )
        normalized_eligibility["publication_status"] = publication_status or None
        normalized_eligibility["original_publication_verified"] = original_publication_verified
        normalized_eligibility["publication_verified"] = publication_verified
        normalized_eligibility["importer_verified"] = True
        capability_ready = any(
            capabilities.get(key) is True for key in ("filter_replay", "pipeline_replay")
        )
        normalized_eligibility["auto_eligible"] = bool(
            capability_ready
            and normalized_eligibility.get("metadata_eligible") is True
            and (recovery or publication_verified)
            and (recovery or publication_status in {"published", "superseded", "verified", "success"})
        )
        row = self.state.ensure_discovered(
            identity,
            source={
                "repository": source.get("repository", SOURCE_REPOSITORY),
                "run_id": source.get("run_id", source.get("source_run_id")),
                "attempt": source.get("run_attempt", source.get("source_attempt", 1)),
                "report_date": manifest.get("report_date"),
                "bundle_sha256": manifest.get("bundle_sha256"),
                "manifest_path": str(path),
                "metadata_only": False,
            },
            eligibility=normalized_eligibility,
            capabilities=copy.deepcopy(dict(capabilities)),
        )
        # A metadata-only discovery may have created this identity before the
        # sealed bundle arrived. Refresh the same identity's evidence gates
        # without changing its versioned key.
        row = self.state.update_metadata(
            row["experiment_id"],
            source={
                "metadata_only": False,
                "manifest_path": str(path),
                "bundle_sha256": manifest.get("bundle_sha256"),
                "report_date": manifest.get("report_date"),
            },
            eligibility=normalized_eligibility,
            capabilities=copy.deepcopy(dict(capabilities)),
        )
        if status != STATUS_DISCOVERED:
            # A result writer may record an evaluator outcome in one command.
            # The transition rules still prevent a completed experiment from
            # being overwritten by a later attempt.
            row = self.state.record_status(
                row["experiment_id"],
                status,
                details={
                    "manifest_path": str(path),
                    "bundle_sha256": manifest.get("bundle_sha256"),
                    "result_experiment_id": result_experiment_id,
                    "result_path": result_path,
                    "evaluator_sha": identity.replay_sha,
                },
            )
        return {
            "experiment_id": row["experiment_id"],
            "manifest": manifest,
            "record": row,
            "evaluator_sha": identity.replay_sha,
            "result_experiment_id": result_experiment_id,
        }

    def select_pending(self, *, limit: int = MAX_COORDINATOR_BATCH,
                       experiment_ids: set[str] | None = None) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_COORDINATOR_BATCH:
            raise SelectorError("pending selection is bounded to at most two records")
        rows = [
            row
            for row in self.state.pending()
            if (
                (experiment_ids is None or row["experiment_id"] in experiment_ids)
                and
                row.get("status") == STATUS_DISCOVERED
                and
                (row.get("eligibility") or {}).get("auto_eligible") is True
                and (row.get("eligibility") or {}).get("importer_verified") is True
                and (
                    (row.get("capabilities") or {}).get("filter_replay") is True
                    or (row.get("capabilities") or {}).get("pipeline_replay") is True
                )
            )
        ]
        rows.sort(key=lambda row: (row.get("created_at") or "", row["experiment_id"]))
        # The scheduled prospective pilot is deliberately finite. A changed
        # evaluator SHA/policy/retry plan creates a new identity, while an
        # unchanged completed source date cannot consume another paid slot.
        completed_by_version: dict[tuple[Any, ...], set[str]] = {}
        for row in self.state.list_records(statuses={STATUS_COMPLETED}):
            identity = row.get("identity") or {}
            report_date = (row.get("source") or {}).get("report_date")
            if identity.get("cohort") != "prospective" or not report_date:
                continue
            version = (
                identity.get("policy_sha256"),
                identity.get("cohort_sha256"),
                identity.get("repeat_plan"),
            )
            completed_by_version.setdefault(version, set()).add(report_date)
        filtered: list[dict[str, Any]] = []
        for row in rows:
            identity = row.get("identity") or {}
            if identity.get("cohort") == "prospective":
                version = (
                    identity.get("policy_sha256"),
                    identity.get("cohort_sha256"),
                    identity.get("repeat_plan"),
                )
                if len(completed_by_version.get(version, set())) >= MAX_PROSPECTIVE_DATES:
                    continue
            filtered.append(row)
        rows = filtered
        return rows[:limit]

    def prospective_cap_reached(self) -> bool:
        """Return whether a versioned prospective pilot has its seven-date cap."""

        completed_by_version: dict[tuple[Any, ...], set[str]] = {}
        for row in self.state.list_records(statuses={STATUS_COMPLETED}):
            identity = row.get("identity") or {}
            report_date = (row.get("source") or {}).get("report_date")
            if identity.get("cohort") != "prospective" or not report_date:
                continue
            version = (
                identity.get("policy_sha256"),
                identity.get("cohort_sha256"),
                identity.get("repeat_plan"),
            )
            completed_by_version.setdefault(version, set()).add(report_date)
        return any(len(dates) >= MAX_PROSPECTIVE_DATES for dates in completed_by_version.values())

    def claim_next(
        self,
        *,
        owner: str,
        lease_seconds: float = 1800,
        limit: int = MAX_COORDINATOR_BATCH,
        experiment_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        # The initial policy is serial: an active lease makes the scheduler a
        # no-op until that date completes.  ``limit`` remains capped at two so
        # a future explicit parallel policy cannot accidentally fan out.
        self.state.recover_expired_leases()
        eligible_ids = {row["experiment_id"] for row in self.select_pending(
            limit=limit, experiment_ids=experiment_ids)}
        if not eligible_ids:
            return []
        return self.state.claim_next(
            owner=owner,
            limit=limit,
            lease_seconds=lease_seconds,
            serial=True,
            eligible_only=True,
            eligible_experiments=eligible_ids,
        )


# Compatibility aliases used by scripts and offline callers.
CoordinatorConfig = Selector
parse_source_runs = parse_run_attempts
validate_selector = parse_selector
classify_run = classify_source_run
make_experiment_id = experiment_id


__all__ = [
    "Coordinator",
    "CoordinatorConfig",
    "DiscoveryError",
    "INTERNAL_REPOSITORY",
    "MAX_COORDINATOR_BATCH",
    "MAX_DATE_RANGE",
    "MAX_PROSPECTIVE_DATES",
    "RunAttempt",
    "SOURCE_REPOSITORY",
    "SOURCE_WORKFLOW_PATH",
    "Selector",
    "SelectorError",
    "SourceClassification",
    "TrustedRefError",
    "TRUSTED_EVALUATOR_REF",
    "classify_run",
    "classify_source_run",
    "discover_metadata",
    "explicit_report_date",
    "experiment_id",
    "identity_for_discovery",
    "identity_for_manifest",
    "make_experiment_id",
    "parse_run_attempts",
    "parse_selector",
    "parse_source_runs",
    "run_attempt_from_metadata",
    "validate_selector",
    "validate_trusted_ref",
]
