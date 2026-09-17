"""Persistent, append-only state for the internal news shadow coordinator.

The coordinator is deliberately boring state machinery.  It does not know how
to download a GitHub artifact, run a replay, or call a model.  It records the
small amount of metadata needed to deduplicate work and to resume a bounded
experiment after an Actions runner disappears.

``index.json`` is a compact materialized view.  ``events.jsonl`` is the
append-only audit trail.  Both are written while holding a small process/file
lock and every mutation can be guarded by the caller's expected revision.  The
event log contains metadata only; source articles, prompts, model responses,
and other large evidence remain in retained workflow artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .contracts import BundleValidationError, canonical_bytes, read_json

try:  # pragma: no cover - exercised on the POSIX CI runners
    import fcntl
except ImportError:  # pragma: no cover - makes the module importable on Windows
    fcntl = None


INDEX_SCHEMA = "news-shadow-index/v1"
EVENT_SCHEMA = "news-shadow-event/v1"
STATUS_DISCOVERED = "discovered"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_INCOMPLETE = "incomplete"
STATUS_UNAVAILABLE = "unavailable"
STATUSES = frozenset(
    {
        STATUS_DISCOVERED,
        STATUS_RUNNING,
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_INCOMPLETE,
        STATUS_UNAVAILABLE,
    }
)
TERMINAL_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_FAILED, STATUS_INCOMPLETE, STATUS_UNAVAILABLE}
)

# A run can be retried after a transient runner failure.  A completed
# experiment is immutable; a repeat/version change must create a new identity.
ALLOWED_TRANSITIONS = {
    STATUS_DISCOVERED: {
        STATUS_DISCOVERED,
        STATUS_RUNNING,
        # A trusted evaluator may finish between acquisition and its first
        # lease write.  The result writer records that terminal outcome
        # idempotently; a version/repeat still has a different identity.
        STATUS_COMPLETED,
        STATUS_UNAVAILABLE,
        STATUS_FAILED,
    },
    STATUS_RUNNING: {
        STATUS_RUNNING,
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_INCOMPLETE,
        STATUS_UNAVAILABLE,
        STATUS_DISCOVERED,  # expired lease recovery
    },
    STATUS_COMPLETED: {STATUS_COMPLETED},
    STATUS_FAILED: {STATUS_FAILED, STATUS_RUNNING},
    STATUS_INCOMPLETE: {STATUS_INCOMPLETE, STATUS_RUNNING},
    STATUS_UNAVAILABLE: {STATUS_UNAVAILABLE, STATUS_RUNNING},
}


class StateError(RuntimeError):
    """The persistent coordinator state is malformed or cannot be updated."""


class StateConflict(StateError):
    """An optimistic revision or lease check failed."""


class InvalidTransition(StateError):
    """A status transition would mutate a terminal experiment."""


class LeaseError(StateError):
    """A worker attempted to renew or release a lease it does not own."""


def _now(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateError("Coordinator clock returned a nonnumeric value")
    return float(value)


def _timestamp(epoch: float) -> str:
    # Keep the state readable and deterministic enough for tests.  The
    # identity never includes timestamps, so wall-clock changes cannot create a
    # second experiment.
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0 or str(result) != str(value).strip() and not isinstance(value, int):
        # The string check rejects values such as "1.0" and "01" while still
        # accepting GitHub's normal decimal input.  Integer arguments are
        # already unambiguous.
        raise ValueError(f"{field} must be a positive integer")
    return result


@dataclass(frozen=True)
class ExperimentIdentity:
    """All versioned inputs that define one reproducible comparison."""

    repository: str
    source_run_id: int
    source_attempt: int
    bundle_sha256: str | None
    replay_sha: str | None
    dependency_lock: str | None
    container_digest: str | None
    mode: str
    candidate_policy: str | None
    rubric_version: str | None
    candidate_model: str | None
    judge_version: str | None
    judge_model: str | None
    sampling_plan: str | None
    repeat_plan: str | None
    # These fields make the version actually executed visible in the
    # coordinator key.  They are optional for metadata-only discovery, but a
    # sealed/evaluated bundle should provide them.
    policy_sha256: str | None = None
    cohort: str | None = None
    cohort_sha256: str | None = None
    incumbent_route: str | None = None
    incumbent_model: str | None = None

    def __post_init__(self) -> None:
        if not self.repository or not isinstance(self.repository, str):
            raise ValueError("repository is required")
        if self.source_run_id <= 0 or self.source_attempt <= 0:
            raise ValueError("source run and attempt must be positive")
        if self.mode not in {"filter", "pipeline"}:
            raise ValueError("mode must be filter or pipeline")
        for field in (
            "bundle_sha256",
            "replay_sha",
            "dependency_lock",
            "container_digest",
            "candidate_policy",
            "rubric_version",
            "candidate_model",
            "judge_version",
            "judge_model",
            "sampling_plan",
            "repeat_plan",
            "policy_sha256",
            "cohort",
            "cohort_sha256",
            "incumbent_route",
            "incumbent_model",
        ):
            value = getattr(self, field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field} must be a string or null")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentIdentity":
        if not isinstance(value, Mapping):
            raise ValueError("identity must be an object")
        run_id = value.get("source_run_id", value.get("run_id"))
        attempt = value.get("source_attempt", value.get("attempt", 1))
        return cls(
            repository=str(value.get("repository", "")),
            source_run_id=_positive_int(run_id, "source_run_id"),
            source_attempt=_positive_int(attempt, "source_attempt"),
            bundle_sha256=value.get("bundle_sha256"),
            replay_sha=value.get("replay_sha"),
            dependency_lock=value.get("dependency_lock"),
            container_digest=value.get("container_digest"),
            mode=str(value.get("mode", "filter")),
            candidate_policy=value.get("candidate_policy", value.get("policy_version")),
            rubric_version=value.get("rubric_version"),
            candidate_model=value.get("candidate_model"),
            judge_version=value.get("judge_version"),
            judge_model=value.get("judge_model"),
            sampling_plan=value.get("sampling_plan"),
            repeat_plan=value.get("repeat_plan"),
            policy_sha256=value.get("policy_sha256", value.get("policy_hash")),
            cohort=value.get("cohort"),
            cohort_sha256=value.get("cohort_sha256", value.get("cohort_hash")),
            incumbent_route=value.get("incumbent_route", value.get("route_id")),
            incumbent_model=value.get("incumbent_model"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "source_run_id": self.source_run_id,
            "source_attempt": self.source_attempt,
            "bundle_sha256": self.bundle_sha256,
            "replay_sha": self.replay_sha,
            "dependency_lock": self.dependency_lock,
            "container_digest": self.container_digest,
            "mode": self.mode,
            "candidate_policy": self.candidate_policy,
            "rubric_version": self.rubric_version,
            "candidate_model": self.candidate_model,
            "judge_version": self.judge_version,
            "judge_model": self.judge_model,
            "sampling_plan": self.sampling_plan,
            "repeat_plan": self.repeat_plan,
            "policy_sha256": self.policy_sha256,
            "cohort": self.cohort,
            "cohort_sha256": self.cohort_sha256,
            "incumbent_route": self.incumbent_route,
            "incumbent_model": self.incumbent_model,
        }

    @property
    def experiment_id(self) -> str:
        return _canonical_hash(self.as_dict())

    @property
    def source_key(self) -> str:
        return f"{self.repository}#{self.source_run_id}:{self.source_attempt}"


def identity_from_mapping(value: Mapping[str, Any]) -> ExperimentIdentity:
    """Public adapter used by the coordinator and importer.

    Keeping this function permissive about old field aliases lets the importer
    evolve its manifest without weakening the identity hash itself.
    """

    return ExperimentIdentity.from_mapping(value)


def experiment_id(value: Mapping[str, Any] | ExperimentIdentity) -> str:
    identity = value if isinstance(value, ExperimentIdentity) else identity_from_mapping(value)
    return identity.experiment_id


def _default_index() -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA,
        "revision": 0,
        "updated_at": None,
        "experiments": {},
    }


def _ensure_index(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != INDEX_SCHEMA:
        raise StateError("Unsupported shadow coordinator index")
    revision = value.get("revision")
    if type(revision) is not int or revision < 0:
        raise StateError("Invalid shadow coordinator revision")
    experiments = value.get("experiments")
    if not isinstance(experiments, dict):
        raise StateError("Invalid shadow coordinator experiment map")
    for key, row in experiments.items():
        if not isinstance(key, str) or not isinstance(row, dict):
            raise StateError("Invalid shadow coordinator experiment record")
        status = row.get("status")
        if status not in STATUSES:
            raise StateError("Invalid shadow coordinator status")
        if row.get("experiment_id") != key:
            raise StateError("Experiment record key does not match experiment_id")
    return value


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Lock a state directory without introducing a third-party dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(value) + b"\n"
    fd, name = tempfile.mkstemp(prefix=".shadow-index-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise StateError("Refusing to append to a symlinked event log")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_bytes(value).decode("utf-8"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class StateStore:
    """Materialized index and append-only events for one results branch."""

    def __init__(self, root: str | Path, *, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        if self.root.exists() and self.root.is_symlink():
            raise StateError("State root may not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.index_path = self.root / "index.json"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / ".lock"
        if self.index_path.exists() and self.index_path.is_symlink():
            raise StateError("State index may not be a symlink")
        if self.events_path.exists() and self.events_path.is_symlink():
            raise StateError("State event log may not be a symlink")
        if not self.index_path.exists():
            _atomic_write(self.index_path, _default_index())

    def _read_unlocked(self) -> dict[str, Any]:
        try:
            value = read_json(self.index_path)
        except (OSError, BundleValidationError) as exc:
            raise StateError("Unable to read coordinator index") from exc
        return _ensure_index(value)

    def snapshot(self) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            return copy.deepcopy(self._read_unlocked())

    @property
    def revision(self) -> int:
        return int(self.snapshot()["revision"])

    def get(self, experiment: str) -> dict[str, Any] | None:
        with _file_lock(self.lock_path):
            index = self._read_unlocked()
            row = index["experiments"].get(experiment)
            return copy.deepcopy(row) if row is not None else None

    def list_records(self, *, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        allowed = set(statuses) if statuses is not None else None
        if allowed is not None and not allowed <= STATUSES:
            raise ValueError("Unknown coordinator status")
        with _file_lock(self.lock_path):
            index = self._read_unlocked()
            rows = [
                copy.deepcopy(row)
                for row in index["experiments"].values()
                if allowed is None or row["status"] in allowed
            ]
        rows.sort(key=lambda row: (row.get("created_at") or "", row["experiment_id"]))
        return rows

    def _mutate(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        mutator: Callable[[dict[str, Any], float], Any],
        *,
        expected_revision: int | None = None,
    ) -> Any:
        if not event_type or not isinstance(event_type, str):
            raise ValueError("event_type is required")
        with _file_lock(self.lock_path):
            index = self._read_unlocked()
            current_revision = index["revision"]
            if expected_revision is not None and current_revision != expected_revision:
                raise StateConflict(
                    f"State revision changed: expected {expected_revision}, got {current_revision}"
                )
            now = _now(self.clock)
            result = mutator(index, now)
            new_revision = current_revision + 1
            index["revision"] = new_revision
            index["updated_at"] = _timestamp(now)
            event = {
                "schema_version": EVENT_SCHEMA,
                "revision": new_revision,
                "event_type": event_type,
                "recorded_at": _timestamp(now),
                "payload": copy.deepcopy(dict(payload)),
            }
            event["event_id"] = _canonical_hash(event)
            # The event is durable before the view advances.  A subsequent
            # invocation can inspect the last event and retry the materialized
            # view without issuing model work; no provider call is hidden here.
            _append_jsonl(self.events_path, event)
            _atomic_write(self.index_path, index)
            return copy.deepcopy(result)

    def ensure_discovered(
        self,
        identity: ExperimentIdentity | Mapping[str, Any],
        *,
        source: Mapping[str, Any] | None = None,
        eligibility: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        identity = (
            identity
            if isinstance(identity, ExperimentIdentity)
            else identity_from_mapping(identity)
        )
        experiment = identity.experiment_id

        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            existing = index["experiments"].get(experiment)
            if existing is not None:
                if existing.get("identity") != identity.as_dict():
                    raise StateError("Experiment identity hash collision or mutation")
                # Discovery is idempotent.  Refreshing metadata is explicit via
                # update_metadata; this avoids rewriting a completed record
                # during every 15-minute poll.
                return existing
            row = {
                "experiment_id": experiment,
                "identity": identity.as_dict(),
                "source": copy.deepcopy(dict(source or {})),
                "eligibility": copy.deepcopy(dict(eligibility or {})),
                "capabilities": copy.deepcopy(dict(capabilities or {})),
                "status": STATUS_DISCOVERED,
                "created_at": _timestamp(now),
                "updated_at": _timestamp(now),
                "steps": {},
                "lease": None,
            }
            index["experiments"][experiment] = row
            return row

        # Avoid advancing the revision for an idempotent discovery.  This is
        # important when a schedule overlaps a slow metadata pagination pass.
        current = self.get(experiment)
        if current is not None:
            if current.get("identity") != identity.as_dict():
                raise StateError("Experiment identity hash collision or mutation")
            return current
        return self._mutate(
            "discovered",
            {
                "experiment_id": experiment,
                "source_key": identity.source_key,
                "source": dict(source or {}),
            },
            mutate,
            expected_revision=expected_revision,
        )

    def update_metadata(
        self,
        experiment: str,
        *,
        source: Mapping[str, Any] | None = None,
        eligibility: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            row = index["experiments"].get(experiment)
            if row is None:
                raise StateError(f"Unknown experiment: {experiment}")
            if source is not None:
                row["source"].update(copy.deepcopy(dict(source)))
            if eligibility is not None:
                row["eligibility"].update(copy.deepcopy(dict(eligibility)))
            if capabilities is not None:
                row["capabilities"].update(copy.deepcopy(dict(capabilities)))
            row["updated_at"] = _timestamp(now)
            return row

        return self._mutate(
            "metadata",
            {"experiment_id": experiment},
            mutate,
            expected_revision=expected_revision,
        )

    def record_status(
        self,
        experiment: str,
        status: str,
        *,
        details: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if status not in STATUSES:
            raise ValueError(f"Unknown coordinator status: {status}")

        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            row = index["experiments"].get(experiment)
            if row is None:
                raise StateError(f"Unknown experiment: {experiment}")
            old = row["status"]
            if status not in ALLOWED_TRANSITIONS[old]:
                raise InvalidTransition(f"Cannot transition {old} -> {status}")
            row["status"] = status
            row["updated_at"] = _timestamp(now)
            if details:
                row.setdefault("status_details", {}).update(copy.deepcopy(dict(details)))
            if status != STATUS_RUNNING:
                row["lease"] = None
            return row

        return self._mutate(
            "status",
            {"experiment_id": experiment, "status": status, "details": dict(details or {})},
            mutate,
            expected_revision=expected_revision,
        )

    def record_result(
        self,
        experiment: str,
        *,
        status: str,
        details: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Record an evaluator result without accepting a new identity."""

        return self.record_status(
            experiment,
            status,
            details=details,
            expected_revision=expected_revision,
        )

    def record_step(
        self,
        experiment: str,
        step: str,
        *,
        status: str,
        request_hash: str | None = None,
        details: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if not step or not isinstance(step, str):
            raise ValueError("step is required")
        if not status or not isinstance(status, str):
            raise ValueError("step status is required")

        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            row = index["experiments"].get(experiment)
            if row is None:
                raise StateError(f"Unknown experiment: {experiment}")
            step_row = {
                "status": status,
                "request_hash": request_hash,
                "details": copy.deepcopy(dict(details or {})),
                "updated_at": _timestamp(now),
            }
            row.setdefault("steps", {})[step] = step_row
            row["updated_at"] = _timestamp(now)
            return row

        return self._mutate(
            "step",
            {
                "experiment_id": experiment,
                "step": step,
                "status": status,
                "request_hash": request_hash,
            },
            mutate,
            expected_revision=expected_revision,
        )

    def claim_next(
        self,
        *,
        owner: str,
        limit: int = 2,
        lease_seconds: float = 1800,
        expected_revision: int | None = None,
        serial: bool = True,
        eligible_only: bool = False,
        eligible_experiments: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Claim at most ``limit`` records, bounded to two by policy.

        ``serial=True`` enforces one active lease.  The coordinator uses a
        batch of two only for queue selection; a second date remains pending
        until the first lease is released or expires.  Callers that explicitly
        want two independent leases may set ``serial=False`` while retaining
        the hard two-item ceiling.
        """

        if not owner or not isinstance(owner, str):
            raise ValueError("owner is required")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 2:
            raise ValueError("coordinator claim limit must be between one and two")
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)):
            raise ValueError("lease_seconds must be numeric")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        allowed_experiments = set(eligible_experiments) if eligible_experiments is not None else None
        if allowed_experiments is not None and not all(
            isinstance(value, str) and value for value in allowed_experiments
        ):
            raise ValueError("eligible_experiments must contain nonempty IDs")

        def mutate(index: dict[str, Any], now: float) -> list[dict[str, Any]]:
            active = [
                row
                for row in index["experiments"].values()
                if row.get("status") == STATUS_RUNNING
                and row.get("lease")
                and float(row["lease"].get("expires_at_epoch", 0)) > now
            ]
            if serial and active:
                return []
            candidates = []
            for row in index["experiments"].values():
                if allowed_experiments is not None and row.get("experiment_id") not in allowed_experiments:
                    continue
                if row.get("status") not in {
                    STATUS_DISCOVERED,
                    STATUS_FAILED,
                    STATUS_INCOMPLETE,
                    STATUS_UNAVAILABLE,
                }:
                    continue
                if eligible_only:
                    eligibility = row.get("eligibility") or {}
                    capabilities = row.get("capabilities") or {}
                    importer_ready = (
                        eligibility.get("auto_eligible") is True
                        and
                        eligibility.get("importer_verified") is True
                        and (capabilities.get("filter_replay") is True
                             or capabilities.get("pipeline_replay") is True)
                    )
                    if not importer_ready:
                        continue
                lease = row.get("lease")
                if lease and float(lease.get("expires_at_epoch", 0)) > now:
                    continue
                candidates.append(row)
            candidates.sort(key=lambda row: (row.get("created_at") or "", row["experiment_id"]))
            chosen = candidates[:1 if serial else limit]
            for row in chosen:
                row["status"] = STATUS_RUNNING
                row["lease"] = {
                    "owner": owner,
                    "issued_at": _timestamp(now),
                    "issued_at_epoch": now,
                    "expires_at": _timestamp(now + float(lease_seconds)),
                    "expires_at_epoch": now + float(lease_seconds),
                }
                row["updated_at"] = _timestamp(now)
            return chosen

        return self._mutate(
            "lease_acquired",
            {"owner": owner, "limit": limit, "serial": serial, "eligible_only": eligible_only},
            mutate,
            expected_revision=expected_revision,
        )

    def renew_lease(
        self,
        experiment: str,
        *,
        owner: str,
        lease_seconds: float = 1800,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if not owner:
            raise ValueError("owner is required")

        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            row = index["experiments"].get(experiment)
            if row is None:
                raise StateError(f"Unknown experiment: {experiment}")
            lease = row.get("lease") or {}
            if row.get("status") != STATUS_RUNNING or lease.get("owner") != owner:
                raise LeaseError("Lease is not owned by this coordinator")
            if float(lease.get("expires_at_epoch", 0)) <= now:
                raise LeaseError("Lease has expired")
            lease["expires_at"] = _timestamp(now + float(lease_seconds))
            lease["expires_at_epoch"] = now + float(lease_seconds)
            row["updated_at"] = _timestamp(now)
            return row

        return self._mutate(
            "lease_renewed",
            {"experiment_id": experiment, "owner": owner},
            mutate,
            expected_revision=expected_revision,
        )

    def release_lease(
        self,
        experiment: str,
        *,
        owner: str,
        status: str | None = None,
        details: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in STATUSES:
            raise ValueError("Unknown coordinator status")

        def mutate(index: dict[str, Any], now: float) -> dict[str, Any]:
            row = index["experiments"].get(experiment)
            if row is None:
                raise StateError(f"Unknown experiment: {experiment}")
            lease = row.get("lease") or {}
            if lease.get("owner") != owner:
                raise LeaseError("Lease is not owned by this coordinator")
            if status is not None:
                old = row["status"]
                if status not in ALLOWED_TRANSITIONS[old]:
                    raise InvalidTransition(f"Cannot transition {old} -> {status}")
                row["status"] = status
            row["lease"] = None
            row["updated_at"] = _timestamp(now)
            if details:
                row.setdefault("status_details", {}).update(copy.deepcopy(dict(details)))
            return row

        return self._mutate(
            "lease_released",
            {"experiment_id": experiment, "owner": owner, "status": status},
            mutate,
            expected_revision=expected_revision,
        )

    def recover_expired_leases(self, *, expected_revision: int | None = None) -> list[str]:
        def mutate(index: dict[str, Any], now: float) -> list[str]:
            recovered = []
            for row in index["experiments"].values():
                lease = row.get("lease") or {}
                if row.get("status") == STATUS_RUNNING and lease and float(
                    lease.get("expires_at_epoch", 0)
                ) <= now:
                    row["status"] = STATUS_DISCOVERED
                    row["lease"] = None
                    row.setdefault("status_details", {})["lease_recovered_at"] = _timestamp(now)
                    row["updated_at"] = _timestamp(now)
                    recovered.append(row["experiment_id"])
            return recovered

        # A no-op recovery should not churn the branch every 15 minutes.
        with _file_lock(self.lock_path):
            index = self._read_unlocked()
            now = _now(self.clock)
            pending = [
                row["experiment_id"]
                for row in index["experiments"].values()
                if row.get("status") == STATUS_RUNNING
                and row.get("lease")
                and float(row["lease"].get("expires_at_epoch", 0)) <= now
            ]
        if not pending:
            return []
        return self._mutate(
            "lease_recovered",
            {"experiments": pending},
            mutate,
            expected_revision=expected_revision,
        )

    def pending(self) -> list[dict[str, Any]]:
        return self.list_records(
            statuses={STATUS_DISCOVERED, STATUS_RUNNING, STATUS_FAILED, STATUS_INCOMPLETE, STATUS_UNAVAILABLE}
        )


# Friendly aliases for callers that use the term from the implementation plan.
CoordinatorState = StateStore
StateConflictError = StateConflict


__all__ = [
    "ALLOWED_TRANSITIONS",
    "CoordinatorState",
    "ExperimentIdentity",
    "InvalidTransition",
    "LeaseError",
    "StateConflict",
    "StateConflictError",
    "StateError",
    "StateStore",
    "STATUS_COMPLETED",
    "STATUS_DISCOVERED",
    "STATUS_FAILED",
    "STATUS_INCOMPLETE",
    "STATUS_RUNNING",
    "STATUS_UNAVAILABLE",
    "STATUSES",
    "TERMINAL_STATUSES",
    "experiment_id",
    "identity_from_mapping",
]
