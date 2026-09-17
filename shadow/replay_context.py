"""Fail-closed access to a sealed shadow bundle.

The production pipeline may use an optional :class:`CaptureSession`; replay
uses this class to make every historical dependency explicit.  Missing files,
hash mismatches, dates on/after the report date and absent evidence all raise
``ReplayIntegrityError``.  There is intentionally no HTTP fallback.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .capture import MAX_HISTORY_DAYS, RUNTIME_ENV_KEYS, SCHEMA_VERSION, _safe_relative_path
from .contracts import BundleValidationError, bundle_path, hash_file, read_json, verify_bundle
from .evidence import EvidenceStore, FrozenResponse, ReplayIntegrityError

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CATEGORIES = ("news", "research", "social", "reddit")
_HISTORY_NAMES = frozenset({*(f"{cat}.json" for cat in _CATEGORIES), "search-documents.json"})


def _date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value
    parsed = date.fromisoformat(str(value))
    if parsed.isoformat() != str(value):
        raise ReplayIntegrityError("Dates must use YYYY-MM-DD")
    return parsed


class ReplayContext:
    """Read-only frozen dependencies plus a separate optional branch root."""

    def __init__(
        self,
        bundle_root: str | Path,
        branch_root: str | Path | None = None,
        *,
        required_capability: str = "filter_replay",
        verify: bool = True,
        execution_policy: dict[str, Any] | None = None,
        evidence_store: EvidenceStore | None = None,
    ):
        self.bundle_root = Path(bundle_root)
        if self.bundle_root.is_symlink():
            raise ReplayIntegrityError("Replay bundle root may not be a symlink")
        self.bundle_root = self.bundle_root.resolve()
        self.branch_root = Path(branch_root).resolve() if branch_root is not None else None
        if self.branch_root is not None and (
            self.branch_root == self.bundle_root
            or self.branch_root in self.bundle_root.parents
            or self.bundle_root in self.branch_root.parents
        ):
            raise ReplayIntegrityError("Replay branch root must be separate from frozen bundle")
        self.required_capability = required_capability
        self.execution_policy = copy.deepcopy(execution_policy or {
            "source_http": False,
            "openrouter_catalog": False,
            "git_writes": False,
            "deploy_writes": False,
            "alert_http": False,
            "only_configured_model_endpoints": True,
        })
        try:
            if verify:
                self.manifest = verify_bundle(
                    self.bundle_root,
                    capability=required_capability,
                )
            else:
                self.manifest = read_json(bundle_path(self.bundle_root, "manifest.json"))
        except Exception as exc:
            if isinstance(exc, ReplayIntegrityError):
                raise
            raise ReplayIntegrityError(f"Replay bundle verification failed: {exc}") from exc
        if not isinstance(self.manifest, dict):
            raise ReplayIntegrityError("Replay bundle manifest is not an object")
        try:
            self.report_date = _date(self.manifest["report_date"])
        except Exception as exc:
            if isinstance(exc, ReplayIntegrityError):
                raise
            raise ReplayIntegrityError("Replay bundle has no valid report date") from exc
        coverage = self.manifest.get("coverage") or {}
        self.coverage_start = coverage.get("start")
        self.coverage_end = coverage.get("end")
        self.evidence = evidence_store or EvidenceStore(self.bundle_root, mode="replay", strict=True)
        # Existing freshness integration uses the dependency name rather than
        # the context's shorter ``evidence`` property.
        self.evidence_store = self.evidence
        self._verified = bool(verify)

    @property
    def frozen_report_date(self) -> str:
        return self.report_date.isoformat()

    @property
    def frozen_coverage(self) -> dict[str, Any]:
        return {
            "timezone": (self.manifest.get("coverage") or {}).get("timezone"),
            "start": self.coverage_start,
            "end": self.coverage_end,
        }

    @property
    def capabilities(self) -> dict[str, Any]:
        return dict(self.manifest.get("capabilities") or {})

    def require_capability(self, capability: str) -> None:
        if (self.manifest.get("capabilities") or {}).get(capability) is not True:
            missing = self.manifest.get("missing_dependencies") or []
            raise ReplayIntegrityError(
                f"Replay bundle lacks {capability}; missing dependencies: {missing}"
            )

    def _read_bytes(self, relative: str) -> bytes:
        try:
            path = bundle_path(self.bundle_root, relative)
            if path.is_symlink() or not path.is_file():
                raise ReplayIntegrityError(f"Frozen dependency is missing: {relative}")
            return path.read_bytes()
        except ReplayIntegrityError:
            raise
        except Exception as exc:
            raise ReplayIntegrityError(f"Frozen dependency cannot be read: {relative}: {exc}") from exc

    def read_bytes(self, relative: str, *, required: bool = True) -> bytes | None:
        try:
            return self._read_bytes(relative)
        except ReplayIntegrityError:
            if required:
                raise
            return None

    def read_text(self, relative: str, *, required: bool = True) -> str | None:
        body = self.read_bytes(relative, required=required)
        if body is None:
            return None
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReplayIntegrityError(f"Frozen text dependency is not UTF-8: {relative}") from exc

    def read_json(self, relative: str, *, required: bool = True) -> Any:
        body = self.read_bytes(relative, required=required)
        if body is None:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise ReplayIntegrityError(f"Frozen JSON dependency is invalid: {relative}") from exc

    def _history_date(self, value: str | date | datetime) -> date:
        day = _date(value)
        if day >= self.report_date:
            raise ReplayIntegrityError("Replay history may not contain the report date or a future date")
        if day < self.report_date - timedelta(days=MAX_HISTORY_DAYS):
            raise ReplayIntegrityError("Replay history exceeds the frozen 45-day lookback")
        return day

    def history_path(self, day: str | date | datetime, name: str) -> str:
        historical_day = self._history_date(day)
        if Path(name).name != name or name not in _HISTORY_NAMES:
            raise ReplayIntegrityError("History path is outside the allowlist")
        return f"history/{historical_day.isoformat()}/{name}"

    def read_history(
        self,
        day: str | date | datetime,
        *,
        category: str | None = None,
        name: str | None = None,
        required: bool = True,
    ) -> Any:
        if category is not None:
            if category not in _CATEGORIES:
                raise ReplayIntegrityError(f"Unknown history category: {category}")
            name = f"{category}.json"
        if name is None:
            raise ReplayIntegrityError("History read requires category or file name")
        relative = self.history_path(day, name)
        return self.read_json(relative, required=required)

    def history_files(self, *, include_search_documents: bool = True) -> list[str]:
        """List only frozen history files whose dates precede the report date."""
        history_root = self.bundle_root / "history"
        if not history_root.exists():
            return []
        manifest = self.read_history_manifest(required=False)
        declared_used: set[str] | None = None
        if isinstance(manifest, dict) and isinstance(manifest.get("used"), list):
            declared_used = {str(value) for value in manifest["used"]}
        found: list[str] = []
        for path in sorted(history_root.rglob("*.json")):
            relative = path.relative_to(self.bundle_root).as_posix()
            parts = relative.split("/")
            if len(parts) != 3 or parts[0] != "history" or parts[2] not in _HISTORY_NAMES:
                continue
            if declared_used is not None and relative not in declared_used:
                continue
            day = self._history_date(parts[1])
            if parts[2] == "search-documents.json" and not include_search_documents:
                continue
            found.append(relative)
        return found

    def read_history_manifest(self, *, required: bool = True) -> dict[str, Any] | None:
        return self.read_json("context/history-manifest.json", required=required)

    def load_history(
        self,
        target_date: str | date | datetime | None = None,
        lookback_days: int = MAX_HISTORY_DAYS,
    ) -> list[dict[str, Any]]:
        """Return the exact frozen old-anchor records used by freshness checks."""
        target = _date(target_date or self.report_date)
        if target != self.report_date:
            raise ReplayIntegrityError("Replay history target date disagrees with bundle")
        if lookback_days < 1 or lookback_days > MAX_HISTORY_DAYS:
            raise ReplayIntegrityError("Replay history lookback is outside the frozen bound")
        manifest = self.read_history_manifest(required=True) or {}
        if not isinstance(manifest, dict):
            raise ReplayIntegrityError("Frozen history manifest is not an object")
        if manifest.get("report_date") != self.frozen_report_date:
            raise ReplayIntegrityError("Frozen history manifest date disagrees with bundle")
        used_paths = manifest.get("used")
        missing_paths = manifest.get("missing")
        not_used_paths = manifest.get("not_used") or []
        if not isinstance(used_paths, list) or not isinstance(missing_paths, list) or not isinstance(not_used_paths, list):
            raise ReplayIntegrityError("Frozen history manifest lacks explicit path status")
        used = {str(value) for value in used_paths}
        missing = {str(value) for value in missing_paths}
        not_used = {str(value) for value in not_used_paths}
        used_hashes = manifest.get("used_hashes") or {}
        if not isinstance(used_hashes, dict):
            raise ReplayIntegrityError("Frozen history manifest has invalid used hashes")
        # A listed file is a dependency, not an optional convenience.  Verify
        # its content hash before parsing so a mutable workspace cannot alter
        # old-anchor evidence after bundle verification.
        for relative in used:
            if not relative.startswith("history/"):
                raise ReplayIntegrityError(f"Frozen history used path is invalid: {relative}")
            try:
                path = bundle_path(self.bundle_root, relative)
            except Exception as exc:
                raise ReplayIntegrityError(f"Frozen history used path is invalid: {relative}") from exc
            expected_hash = used_hashes.get(relative)
            try:
                actual_hash = hash_file(path)
            except Exception as exc:
                raise ReplayIntegrityError(f"Frozen history used path is missing: {relative}") from exc
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                raise ReplayIntegrityError(f"Frozen history used path hash mismatch: {relative}")
        expected_categories = {
            f"{(self.report_date - timedelta(days=offset)).isoformat()}/{category}.json"
            for offset in range(1, lookback_days + 1)
            for category in _CATEGORIES
        }
        declared_short = {path.removeprefix("history/") for path in used}
        if not expected_categories.issubset(declared_short | missing | not_used):
            raise ReplayIntegrityError("Frozen history manifest silently omits a dependency")
        result: list[dict[str, Any]] = []

        # A root search corpus is authoritative when capture recorded that it
        # was used.  It contains multiple dates, so apply the same strict
        # report-date/lookback filter as the production checker.
        search_used = "history/search-documents.json" in used
        search_missing = "search-documents.json" in missing
        search_not_used = "search-documents.json" in not_used
        if manifest.get("search_documents_used") and not (search_used or search_missing):
            raise ReplayIntegrityError("Frozen history manifest claims search documents without a path")
        if search_used:
            raw = self.read_json("history/search-documents.json")
            records = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else []
            oldest = self.report_date - timedelta(days=lookback_days)
            for record in records:
                if not isinstance(record, dict):
                    continue
                value = record.get("date")
                try:
                    day = _date(value)
                except Exception:
                    continue
                if day >= self.report_date or day < oldest:
                    continue
                result.append(dict(record))
            # Production falls back to category files when the captured search
            # corpus has no eligible records.  Preserve that exact semantics;
            # do not turn an empty frozen search corpus into a synthetic empty
            # history.
            if result:
                return result

        for offset in range(1, lookback_days + 1):
            day = self.report_date - timedelta(days=offset)
            for category in _CATEGORIES:
                short_relative = f"{day.isoformat()}/{category}.json"
                relative = f"history/{short_relative}"
                if short_relative not in declared_short:
                    # Only an explicit missing/not-used declaration may yield
                    # no records.  An omitted path is an integrity failure.
                    if short_relative not in missing and short_relative not in not_used:
                        raise ReplayIntegrityError(f"Frozen history path has no declaration: {short_relative}")
                    continue
                raw = self.read_json(relative, required=True)
                records = raw.get("items", []) if isinstance(raw, dict) else raw
                if not isinstance(records, list):
                    continue
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    item = dict(record)
                    item.setdefault("date", day.isoformat())
                    item.setdefault("category", category)
                    result.append(item)
        return result

    load_historical_items = load_history
    get_history = load_history

    @property
    def grounding_context(self) -> str:
        return self.read_grounding()

    load_grounding = grounding_context.fget

    @property
    def gathered(self) -> dict[str, Any]:
        metadata = self.read_json("gathered/metadata.json", required=False) or {}
        return {
            "categories": {category: self.read_gathered(category) for category in _CATEGORIES},
            "collection_status": metadata.get("collection_status", {}) if isinstance(metadata, dict) else {},
            "coverage": metadata.get("coverage", {}) if isinstance(metadata, dict) else {},
        }

    gathering = gathered
    load_gathering = gathered.fget
    load_gathered_items = gathered.fget

    @property
    def pre_continuity(self) -> dict[str, Any]:
        return self.read_precontinuity()

    precontinuity = pre_continuity
    load_pre_continuity = pre_continuity.fget

    @property
    def effective_config(self) -> dict[str, Any]:
        return self.read_effective_config()

    def read_gathered(self, category: str) -> Any:
        if category not in _CATEGORIES:
            raise ReplayIntegrityError(f"Unknown gathered category: {category}")
        value = self.read_json(f"gathered/{category}.json")
        # CaptureSession writes the exact list; accept the legacy wrapper for
        # bundles produced by an earlier compatible capture implementation.
        if isinstance(value, dict) and "items" in value:
            return value["items"]
        return value

    def read_filter_input(self) -> dict[str, Any]:
        value = self.read_json("relevance/input.json")
        if not isinstance(value, dict):
            raise ReplayIntegrityError("Frozen filter input is not an object")
        return value

    def read_filter_decision(self) -> dict[str, Any]:
        value = self.read_json("relevance/incumbent-decision.json")
        if not isinstance(value, dict):
            raise ReplayIntegrityError("Frozen filter decision is not an object")
        return value

    def read_precontinuity(self) -> dict[str, Any]:
        value = self.read_json("checkpoints/analysis_pre_continuity.json")
        if not isinstance(value, dict):
            raise ReplayIntegrityError("Frozen pre-continuity checkpoint is not an object")
        return value

    def read_original_output(self, name: str) -> bytes:
        if Path(name).name != name or name not in {"summary.json", *(f"{cat}.json" for cat in _CATEGORIES)}:
            raise ReplayIntegrityError("Original output is outside the allowlist")
        return self._read_bytes(f"original/{name}")

    def original_outputs(self, *, required: bool = True) -> dict[str, bytes]:
        values: dict[str, bytes] = {}
        for name in ("summary.json", *(f"{cat}.json" for cat in _CATEGORIES)):
            try:
                values[name] = self.read_original_output(name)
            except ReplayIntegrityError:
                if required:
                    raise
        return values

    def read_grounding(self) -> str:
        value = self.read_text("context/grounding.txt")
        if value is None:
            raise ReplayIntegrityError("Frozen grounding is missing")
        return value

    def read_prompts(self) -> str:
        value = self.read_text("context/prompts.yaml")
        if value is None:
            raise ReplayIntegrityError("Frozen prompts are missing")
        return value

    prompts = read_prompts

    def read_model_releases(self) -> bytes:
        return self._read_bytes("context/model-releases-before.yaml")

    def read_effective_config(self) -> dict[str, Any]:
        value = self.read_json("context/effective-config.json")
        if not isinstance(value, dict):
            raise ReplayIntegrityError("Frozen effective config is not an object")
        return value

    def read_runtime_settings(self) -> dict[str, str | None]:
        """Read the exact non-secret environment allowlist for a replay."""
        value = self.read_json("context/runtime-settings.json")
        if not isinstance(value, dict) or set(value) != set(RUNTIME_ENV_KEYS):
            raise ReplayIntegrityError("Frozen runtime settings do not match the allowlist")
        for key, setting in value.items():
            if setting is not None and (
                not isinstance(setting, str)
                or "\x00" in setting
                or "\r" in setting
                or "\n" in setting
            ):
                raise ReplayIntegrityError(f"Frozen runtime setting {key} is invalid")
        return value

    runtime_settings = read_runtime_settings

    def resolve_evidence(self, url: str) -> FrozenResponse:
        value = self.evidence.lookup(url, strict=True)
        if value is None:
            raise ReplayIntegrityError(f"No frozen evidence for {url}")
        return value

    def safe_get(self, url: str, timeout: int = 12, max_redirects: int = 5) -> FrozenResponse:
        # Keep the exact incumbent signature and ignore timeout values for
        # replay: they describe the original request, not a permission to do
        # current I/O.
        return self.evidence.safe_get(url, timeout=timeout, max_redirects=max_redirects)

    make_safe_get = safe_get

    def assert_no_source_http(self) -> None:
        if self.execution_policy.get("source_http") is not False:
            raise ReplayIntegrityError("Replay execution policy permits source HTTP")

    def writable_path(self, relative: str | Path) -> Path:
        if self.branch_root is None:
            raise ReplayIntegrityError("Replay branch root was not supplied")
        raw = Path(relative)
        if raw.is_absolute() or any(part in ("", ".", "..") for part in raw.parts):
            raise ReplayIntegrityError("Replay writable path is invalid")
        candidate = _safe_relative_path(self.branch_root, raw)
        if self.bundle_root in candidate.resolve().parents or candidate.resolve() == self.bundle_root:
            raise ReplayIntegrityError("Replay writable path overlaps frozen bundle")
        return candidate

    def branch_directory(self) -> Path:
        if self.branch_root is None:
            raise ReplayIntegrityError("Replay branch root was not supplied")
        if self.branch_root.exists() and self.branch_root.is_symlink():
            raise ReplayIntegrityError("Replay branch root may not be a symlink")
        self.branch_root.mkdir(parents=True, exist_ok=True)
        return self.branch_root


FrozenReplayContext = ReplayContext

__all__ = ["FrozenReplayContext", "ReplayContext", "ReplayIntegrityError"]
