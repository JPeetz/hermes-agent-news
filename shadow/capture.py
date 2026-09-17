"""Bounded, opt-in capture for the TypeSafe news shadow experiment.

The production pipeline must remain useful when capture is unavailable.  This
module therefore behaves like an observer: every public capture method catches
its own failures, records an ``unavailable`` reason when it can, and returns a
small success value.  Replay code uses :mod:`shadow.replay_context` instead;
this module never reads current network state or executes captured content.

The writer intentionally has a narrow vocabulary of artifact paths.  In
particular, callers cannot use it as a general log/environment copier.  The
bundle is immutable once an artifact path has been written: a second write is
allowed only when its bytes are identical.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "news-shadow-bundle/v1"
DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_HISTORY_DAYS = 45

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_CATEGORY_RE = re.compile(r"^(?:news|research|social|reddit)$")
_OUTPUT_RE = re.compile(r"^(?:summary|news|research|social|reddit)\.json$")

# Runtime values that can affect a replay without being credentials or a
# destination.  Capture writes a value for every key (``null`` means that the
# production default was in effect), which makes the boundary auditable while
# keeping provider URLs, proxy settings and tokens out of the bundle.
RUNTIME_ENV_KEYS = frozenset(
    {
        "TARGET_DATE",
        "LOOKBACK_HOURS",
        "TZ",
        "LLM_TRUST_ENV_PROXY",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_CONCURRENT_REQUESTS",
        "LLM_MAX_REQUESTS_PER_MINUTE",
        "LLM_ADAPTIVE_MAX_TOKENS",
        "LLM_MAX_RETRIES",
        "LLM_RETRY_MAX_ATTEMPTS",
        "LLM_RETRY_LIVENESS_WINDOW",
        "LLM_RETRY_MAX_ELAPSED_SECONDS",
        "LLM_RETRY_CONTENDED_DELAY",
        "LLM_RETRY_BASE_DELAY",
        "LLM_RETRY_MAX_DELAY",
        "LLM_LOG_REQUESTS",
        "LLM_HEARTBEAT_SECONDS",
        "LLM_STREAM_STALL_SECONDS",
        "LLM_REPLAY_CAPTURE",
        "LLM_REPLAY_COALESCE_MS",
        "LLM_REPLAY_MAX_DELTAS",
        "LLM_REPLAY_MAX_TOTAL_DELTAS",
        "LLM_REPLAY_MAX_BYTES",
        "ANALYZER_BATCH_SIZE",
        "ANALYZER_MAX_CONCURRENT_BATCHES",
        "ANALYZER_RESULT_MAX_ATTEMPTS",
        "OLD_ANCHOR_MAX_LLM_CHECKS",
        "OLD_ANCHOR_LLM_ENABLED",
    }
)

_SOURCE_ENV_KEYS = {
    # The aliases are explicit and bounded; no arbitrary environment key is
    # ever copied into a bundle.
    "run_id": ("GITHUB_RUN_ID", "RUN_ID"),
    "run_attempt": ("GITHUB_RUN_ATTEMPT", "RUN_ATTEMPT"),
    "event_sha": ("GITHUB_SHA", "GITHUB_EVENT_SHA", "EVENT_SHA", "SHA"),
    "execution_sha": ("NEWS_SHADOW_EXECUTION_SHA", "EXECUTION_SHA"),
    "pre_execution_sha": (
        "NEWS_SHADOW_PRE_EXECUTION_SHA",
        "PRE_EXECUTION_SHA",
        "PREEXEC_SHA",
    ),
}
_SOURCE_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SAFE_RUNTIME_VALUE_RE = re.compile(r"^[^\x00\r\n]{0,4096}$")

# These are names rather than values.  A provider config is deliberately not
# on the allowlist: the normal provider representation contains api_key and
# often embeds endpoint/query credentials.
_CONFIG_KEYS = frozenset(
    {
        "provider_id",
        "provider",
        "provider_name",
        "route_id",
        "route",
        "model",
        "model_id",
        "model_name",
        "model_revision",
        "mode",
        "effort",
        "adaptive_effort",
        "thinking_type",
        "output_max_tokens",
        "max_output_tokens",
        "max_tokens",
        "response_max_tokens",
        "timeout_seconds",
        "timeout",
        "max_retries",
        "retry_policy",
        "retry_after",
        "max_concurrent_requests",
        "concurrency",
        "analyzer_batch_size",
        "analyzer_max_concurrent_batches",
        "batch_size",
        "prompt_hash",
        "prompt_hashes",
        "renderer_hash",
        "rubric_hash",
        "keyword_hash",
        "keyword_rule_hash",
        "freshness_enabled",
        "old_anchor_llm_enabled",
        "freshness_window_days",
        "primary_followup_window_days",
        "old_anchor_lookback_days",
        "old_anchor_max_llm_checks",
        "llm_timeout_seconds",
        "llm_heartbeat_seconds",
        "llm_stream_stall_seconds",
        "llm_replay_capture",
        "llm_replay_coalesce_ms",
        "llm_replay_max_deltas",
        "llm_replay_max_total_deltas",
        "llm_replay_max_bytes",
        "lookback_hours",
        "timezone",
        "tz",
        "target_date",
        "reddit_credit_budget",
        "reddit_fetch_workers",
        "reddit_max_pages",
        "reddit_body_top_n",
        "reddit_min_comments_for_digest",
        "network_policy",
        "llm_trust_env_proxy",
        "llm_adaptive_max_tokens",
        "llm_max_concurrent_requests",
        "llm_max_retries",
        "analysis_profile",
        "temperature",
        "top_p",
        "seed",
        "version",
        "revision",
        "endpoint_id",
        "endpoint_name",
        "base_url",
    }
)
_SECRET_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "credential",
    "authorization",
    "cookie",
    "private",
    "bearer",
    "auth",
)


def _json_default(value: Any) -> Any:
    """Convert the small set of pipeline objects commonly passed to capture."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _jsonable(value: Any) -> Any:
    """Make a detached JSON-compatible copy without invoking arbitrary code."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    # Source/pipeline objects generally expose fields through __dict__.  A
    # caller cannot use this fallback to write arbitrary files; it only shapes
    # the value being written under a pre-approved artifact path.
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _safe_error(exc: BaseException) -> str:
    """Keep failure diagnostics useful without copying URLs, secrets or logs."""
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"https?://[^\s]+", "<url>", text, flags=re.IGNORECASE)
    # Keep the label for diagnosis, but make it a capture group.  The former
    # non-capturing expression used ``\1`` and raised ``invalid group
    # reference`` precisely while handling an observer failure.
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[=:]\s*[^\s,;]+",
        r"\1=<redacted>",
        text,
    )
    return text[:1000]


def _normalise_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = date.fromisoformat(str(value))
    if parsed.isoformat() != str(value):
        raise ValueError("date must use YYYY-MM-DD")
    return parsed


def _safe_relative_path(root: Path, relative: str | Path) -> Path:
    """Resolve a relative artifact path and reject traversal/symlink escapes."""
    raw = Path(relative)
    if raw.is_absolute() or any(part in ("", ".", "..") for part in raw.parts):
        raise ValueError(f"artifact path must be relative and traversal-free: {relative!r}")
    candidate = (root / raw)
    # Existing symlinks are not an acceptable source of an artifact escape.
    cur = root
    for part in raw.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {relative!r}")
    resolved_root = root.resolve()
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != resolved_root and resolved_root not in resolved_parent.parents:
        raise ValueError(f"artifact path escapes bundle root: {relative!r}")
    return candidate


def _sanitize_config(value: Any, *, key: str = "") -> Any:
    """Return only the explicit effective-config allowlist.

    Unknown mapping keys are dropped.  Lists are retained because route and
    retry settings are often represented as lists, but each mapping element is
    filtered independently.  A key containing a secret fragment is always
    dropped even if a future config happens to reuse an allowlisted name.
    """
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            name = str(raw_key)
            lower = name.lower()
            # ``max_tokens`` and the other explicitly allowlisted sizing
            # controls contain the word ``token`` but are not credentials.
            # Apply the exact allowlist before the fragment denylist so these
            # effective settings survive capture without weakening secret
            # filtering for names such as ``token`` or ``api_token``.
            safe_token_setting = lower in {
                "max_tokens",
                "output_max_tokens",
                "max_output_tokens",
                "response_max_tokens",
                "llm_adaptive_max_tokens",
            }
            is_secret_name = (
                lower in {"key", "api_key", "access_key", "secret_key", "private_key"}
                or (
                    any(fragment in lower for fragment in _SECRET_FRAGMENTS)
                    and not safe_token_setting
                )
            )
            if is_secret_name:
                continue
            dynamic_container = key.lower() in {"routes", "prompt_hashes", "retry_policy"} and isinstance(raw_value, Mapping)
            if name not in _CONFIG_KEYS and lower not in _CONFIG_KEYS and not dynamic_container:
                # Permit one level of named sections (``llm``, ``freshness``,
                # ``routes`` and the public pipeline rendering settings) but
                # still filter every leaf below them.
                if lower not in {
                    "llm", "routes", "freshness", "analysis", "sources", "replay", "judge", "pipeline"
                }:
                    continue
            if lower == "base_url":
                # The provider URL is deliberately excluded.  The only
                # public URL that may be frozen is the pipeline's rendering
                # base URL, and it must not carry credentials, query values or
                # a non-HTTPS scheme.
                if key.lower() != "pipeline" or not isinstance(raw_value, str):
                    continue
                from urllib.parse import urlparse

                parsed = urlparse(raw_value)
                if (
                    parsed.scheme.lower() != "https"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    continue
                raw_value = raw_value.rstrip("/")
            if lower in {
                "max_tokens", "output_max_tokens", "max_output_tokens", "response_max_tokens",
                "llm_adaptive_max_tokens",
            }:
                if isinstance(raw_value, bool) or not isinstance(raw_value, int) or not 1024 <= raw_value <= 131072:
                    continue
            if lower == "lookback_hours":
                if isinstance(raw_value, bool) or not isinstance(raw_value, int) or not 1 <= raw_value <= 168:
                    continue
            dynamic_leaf = key.lower() in {"prompt_hashes", "retry_policy"} or (
                key.lower() == "routes" and isinstance(raw_value, Mapping)
            )
            if dynamic_leaf and not isinstance(raw_value, (Mapping, Sequence)):
                nested = _jsonable(raw_value)
            else:
                nested = _sanitize_config(raw_value, key=name)
            if nested is not _DROP:
                result[name] = nested
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            nested = _sanitize_config(item, key=key)
            if nested is not _DROP:
                result.append(nested)
        return result
    if key and (key in _CONFIG_KEYS or key.lower() in _CONFIG_KEYS):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        # Effective config is JSON-shaped by the loader.  Refuse arbitrary
        # object dictionaries here so a caller cannot smuggle a provider or
        # environment object through an allowlisted leaf name.
        return _DROP
    # Scalars under a permitted section still need an explicitly permitted
    # leaf name; dropping here avoids accidental provider/config serialization.
    return _DROP


def _config_issues(value: Any, *, key: str = "", path: str = "") -> list[str]:
    """Find invalid bounded settings without treating secret fields as errors."""
    issues: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            name = str(raw_key)
            lower = name.lower()
            child_path = f"{path}.{name}" if path else name
            if lower in {
                "max_tokens", "output_max_tokens", "max_output_tokens", "response_max_tokens",
                "llm_adaptive_max_tokens",
            }:
                if raw_value is not None and (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, int)
                    or not 1024 <= raw_value <= 131072
                ):
                    issues.append(child_path)
            elif lower == "lookback_hours":
                if isinstance(raw_value, bool) or not isinstance(raw_value, int) or not 1 <= raw_value <= 168:
                    issues.append(child_path)
            elif lower == "base_url" and key.lower() == "pipeline":
                from urllib.parse import urlparse

                parsed = urlparse(raw_value) if isinstance(raw_value, str) else None
                if (
                    parsed is None
                    or parsed.scheme.lower() != "https"
                    or not parsed.hostname
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                ):
                    issues.append(child_path)
            issues.extend(_config_issues(raw_value, key=name, path=child_path))
        return issues
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            issues.extend(_config_issues(item, key=key, path=f"{path}[{index}]"))
    return issues


class _Drop:
    pass


_DROP = _Drop()


class CaptureSession:
    """An opt-in, bounded capture observer.

    ``enabled`` defaults to ``False``.  Production callers must opt in by
    passing it explicitly or setting ``NEWS_SHADOW_CAPTURE=1``.  ``root`` is
    always explicit; there is no implicit current-directory or system-temp
    artifact destination.
    """

    CATEGORIES = ("news", "research", "social", "reddit")
    OUTPUTS = ("summary.json", "news.json", "research.json", "social.json", "reddit.json")

    def __init__(
        self,
        root: str | Path | None,
        report_date: str | date | datetime | None = None,
        *,
        target_date: str | date | datetime | None = None,
        enabled: bool | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        clock: Callable[[], datetime] | None = None,
    ):
        self.root = Path(root) if root is not None else None
        if report_date is None:
            report_date = target_date
        self.report_date = _normalise_date(report_date) if report_date is not None else None
        self.enabled = (
            (
                os.getenv("NEWS_SHADOW_CAPTURE", "").strip().lower() in {"1", "true", "yes", "on"}
                if os.getenv("NEWS_SHADOW_CAPTURE") is not None
                else root is not None
            )
            if enabled is None
            else bool(enabled)
        )
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.max_total_bytes = max(1, int(max_total_bytes))
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.failures: list[dict[str, Any]] = []
        self.written: dict[str, dict[str, Any]] = {}
        self._bytes_written = 0
        self._initialised = False
        self._filter_input_hash: str | None = None
        self.evidence_store = None
        if self.enabled and self.root is not None:
            try:
                from .evidence import EvidenceStore

                self.evidence_store = EvidenceStore(self.root, mode="capture", strict=False)
                # A complete replay bundle records that no freshness HTTP was
                # needed just as explicitly as it records a response/failure.
                self.evidence_store._persist_index()
            except Exception as exc:
                self._record_unavailable(_safe_error(exc), artifact="evidence/index.json", fatal=False)
        if self.enabled and self.root is None:
            self._record_unavailable("capture root was not supplied")

    @property
    def capture_enabled(self) -> bool:
        return self.enabled and self.root is not None and not self._fatal_unavailable

    @property
    def available(self) -> bool:
        return not bool(self.failures)

    @property
    def _fatal_unavailable(self) -> bool:
        return any(row.get("fatal") for row in self.failures)

    def _record_unavailable(self, reason: str, *, artifact: str | None = None, fatal: bool = True) -> None:
        row = {
            "at": self.clock().isoformat(),
            "status": "unavailable",
            "reason": str(reason)[:1000],
            "fatal": bool(fatal),
        }
        if artifact:
            row["artifact"] = artifact
        self.failures.append(row)
        logger.warning("Shadow capture unavailable%s: %s", f" for {artifact}" if artifact else "", reason)
        # Failure reporting is itself best-effort.  Do not recurse through the
        # normal write path, which could mask the production observer failure.
        if self.enabled and self.root is not None and self._initialised:
            try:
                diag = _safe_relative_path(self.root, "diagnostics/capture-failures.json")
                diag.parent.mkdir(parents=True, exist_ok=True)
                payload = _canonical_json(self.failures)
                if len(payload) <= self.max_file_bytes:
                    diag.write_bytes(payload)
            except Exception:
                pass

    record_unavailable = _record_unavailable
    mark_unavailable = _record_unavailable

    def _ensure_root(self) -> bool:
        if not self.enabled:
            return False
        if self.root is None:
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.is_symlink():
                raise ValueError("capture root must not be a symlink")
            self._initialised = True
            return True
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), fatal=True)
            return False

    def _write_bytes(self, relative: str, payload: bytes, *, kind: str = "artifact") -> bool:
        if not self._ensure_root():
            return False
        try:
            path = _safe_relative_path(self.root, relative)
            if len(payload) > self.max_file_bytes:
                raise ValueError(f"{kind} exceeds per-file capture cap")
            existing = path.read_bytes() if path.exists() else None
            digest = hashlib.sha256(payload).hexdigest()
            if existing is not None:
                if existing != payload:
                    raise ValueError("capture artifact is immutable and already contains different bytes")
                self.written[relative] = {"path": relative, "sha256": digest, "bytes": len(payload)}
                return True
            projected = self._bytes_written + len(payload)
            if projected > self.max_total_bytes:
                raise ValueError("capture bundle exceeds total size cap")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("capture artifact path is a symlink")
            # Atomic create.  A concurrent writer may win; identical bytes are
            # accepted on the next read, conflicting bytes remain unavailable.
            with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=str(path.parent), delete=False) as handle:
                temp_name = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temp_name, path)
            finally:
                if temp_name.exists():
                    temp_name.unlink()
            self._bytes_written += len(payload)
            self.written[relative] = {"path": relative, "sha256": digest, "bytes": len(payload)}
            return True
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact=relative, fatal=False)
            return False

    def _write_json(self, relative: str, value: Any) -> bool:
        try:
            return self._write_bytes(relative, _canonical_json(value) + b"\n")
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact=relative, fatal=False)
            return False

    def _replace_json(self, relative: str, value: Any) -> bool:
        """Update one capture-owned metadata file before the bundle is sealed.

        Most artifacts are immutable as soon as observed.  The input manifest
        is the one exception: ``begin_run`` knows source lineage, while the
        Phase 1 observer learns the exact coverage window later.  Keeping this
        narrowly scoped avoids turning the capture writer into a general file
        copier.
        """
        if not self._ensure_root():
            return False
        try:
            payload = _canonical_json(value) + b"\n"
            path = _safe_relative_path(self.root, relative)
            if len(payload) > self.max_file_bytes:
                raise ValueError(f"metadata exceeds per-file capture cap")
            previous = path.read_bytes() if path.exists() else b""
            if previous == payload:
                return True
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ValueError("capture metadata path is a symlink")
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.", dir=str(path.parent), delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            old_size = self.written.get(relative, {}).get("bytes", len(previous))
            self._bytes_written = max(0, self._bytes_written - int(old_size)) + len(payload)
            self.written[relative] = {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            if self._bytes_written > self.max_total_bytes:
                raise ValueError("capture bundle exceeds total size cap")
            return True
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact=relative, fatal=False)
            return False

    @staticmethod
    def _first_environment_value(names: Iterable[str], environ: Mapping[str, str]) -> str | None:
        for name in names:
            value = environ.get(name)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    def _source_lineage(self, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Return bounded run provenance from the explicitly named variables."""
        env = os.environ if environ is None else environ
        source: dict[str, Any] = {
            # The capture is for this one publishing repository.  Do not copy
            # a general environment value into provenance.
            "repository": "flyryan/ai-news-aggregator",
            "workflow_path": ".github/workflows/daily-pipeline.yml",
        }
        run_id = self._first_environment_value(_SOURCE_ENV_KEYS["run_id"], env)
        attempt = self._first_environment_value(_SOURCE_ENV_KEYS["run_attempt"], env)
        if run_id is not None and run_id.isdigit() and int(run_id) > 0:
            source["run_id"] = int(run_id)
        if attempt is not None and attempt.isdigit() and int(attempt) > 0:
            source["run_attempt"] = int(attempt)
        for field in ("event_sha", "execution_sha", "pre_execution_sha"):
            value = self._first_environment_value(_SOURCE_ENV_KEYS[field], env)
            if value is None:
                continue
            # A malformed value is omitted rather than copied into a future
            # receipt as if it were a verified commit identity.
            if _SOURCE_SHA_RE.fullmatch(value):
                source[field] = value.lower()
                if field == "execution_sha":
                    source["execution_sha_evidence"] = "NEWS_SHADOW_EXECUTION_SHA"
                elif field == "pre_execution_sha":
                    source["pre_execution_sha_evidence"] = "NEWS_SHADOW_PRE_EXECUTION_SHA"
        return source

    def capture_runtime_settings(
        self,
        settings: Mapping[str, Any] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Freeze non-secret environment inputs as an exact key/value map.

        ``None`` means the production default was not overridden.  Values are
        strings in the artifact because the replay worker installs them into
        ``os.environ``; it then applies its own mandatory safety overrides.
        """
        try:
            source = os.environ if environ is None else environ
            values: dict[str, str | None] = {}
            for key in sorted(RUNTIME_ENV_KEYS):
                raw = settings.get(key) if settings is not None and key in settings else source.get(key)
                if raw is None:
                    values[key] = None
                    continue
                if isinstance(raw, bool):
                    rendered = "true" if raw else "false"
                elif isinstance(raw, (str, int, float)):
                    rendered = str(raw)
                else:
                    raise TypeError(f"runtime setting {key} must be scalar")
                if not _SAFE_RUNTIME_VALUE_RE.fullmatch(rendered):
                    raise ValueError(f"runtime setting {key} contains unsafe characters")
                values[key] = rendered
            return self._write_json("context/runtime-settings.json", values)
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact="context/runtime-settings.json", fatal=False)
            return False

    capture_runtime = capture_runtime_settings

    def _capture_input_manifest(self, *, report_date: str | None = None) -> bool:
        value = {
            "schema_version": "news-shadow-input-manifest/v1",
            "source": self._source_lineage(),
            "report_date": report_date or (self.report_date.isoformat() if self.report_date else None),
            "coverage": {},
            "input_sha256": None,
        }
        return self._write_json("input-manifest.json", value)

    def _update_input_manifest(self, **updates: Any) -> bool:
        try:
            path = _safe_relative_path(self.root, "input-manifest.json")
            if not path.is_file() or path.is_symlink():
                return True
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                raise ValueError("input manifest is not an object")
            current.update(_jsonable(updates))
            return self._replace_json("input-manifest.json", current)
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact="input-manifest.json", fatal=False)
            return False

    def capture_config(self, config: Mapping[str, Any] | None, *, effective: bool = True) -> bool:
        """Capture only the effective-config allowlist; never provider secrets."""
        try:
            sanitized = _sanitize_config(config or {})
            if sanitized is _DROP:
                sanitized = {}
            issues = _config_issues(config or {})
            payload = {
                "schema_version": SCHEMA_VERSION,
                "effective": bool(effective),
                "valid": not issues,
                "issues": issues,
                "values": sanitized,
            }
            written = self._write_json("context/effective-config.json", payload)
            if issues:
                self._record_unavailable(
                    "effective config contains unsupported bounded settings: " + ", ".join(issues[:20]),
                    artifact="context/effective-config.json",
                    fatal=False,
                )
            return written and not issues
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact="context/effective-config.json", fatal=False)
            return False

    capture_effective_config = capture_config

    def capture_grounding(
        self,
        grounding_text: str | None = None,
        *,
        model_releases: str | bytes | Mapping[str, Any] | None = None,
        resolved_ecosystem: Any = None,
        grounding_context: str | None = None,
        target_date: str | date | datetime | None = None,
        **kwargs: Any,
    ) -> bool:
        """Capture resolved grounding and the pre-run release state."""
        if grounding_text is None:
            grounding_text = grounding_context
        ok = True
        if grounding_text is not None:
            raw = str(grounding_text).encode("utf-8")
            ok = self._write_bytes("context/grounding.txt", raw, kind="grounding") and ok
        if model_releases is not None:
            if isinstance(model_releases, bytes):
                payload = model_releases
            elif isinstance(model_releases, str):
                payload = model_releases.encode("utf-8")
            else:
                payload = _canonical_json(model_releases) + b"\n"
            ok = self._write_bytes("context/model-releases-before.yaml", payload, kind="model release state") and ok
        if resolved_ecosystem is not None:
            ok = self._write_json("context/resolved-ecosystem.json", resolved_ecosystem) and ok
        return ok

    capture_context = capture_grounding

    # Names used by the orchestrator's optional observer adapter.  They keep
    # capture concerns out of the production code's control flow.
    record_grounding = capture_grounding
    record_context = capture_grounding

    def capture_prompts(
        self,
        prompts: str | bytes | Path | None = None,
        *,
        prompts_path: str | Path | None = None,
        **kwargs: Any,
    ) -> bool:
        """Freeze the tracked prompt templates used by the effective run."""
        source = prompts_path if prompts_path is not None else prompts
        if source is None:
            self._record_unavailable("prompt source was not supplied", artifact="context/prompts.yaml", fatal=False)
            return False
        try:
            # A raw YAML string may contain newlines or be longer than a valid
            # filesystem path.  Probe paths only for Path objects and short,
            # newline-free strings so an observer failure cannot obscure the
            # actual prompt bytes.
            path_source = isinstance(source, Path) or (
                isinstance(source, str) and "\n" not in source and "\r" not in source and len(source) < 4096
            )
            if path_source and Path(source).is_file():
                if Path(source).is_symlink():
                    raise ValueError("prompt source may not be a symlink")
                raw = Path(source).read_bytes()
            elif isinstance(source, bytes):
                raw = source
            else:
                raw = str(source).encode("utf-8")
            return self._write_bytes("context/prompts.yaml", raw, kind="prompt snapshot")
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact="context/prompts.yaml", fatal=False)
            return False

    capture_prompt_snapshot = capture_prompts

    def begin_run(self, *, report_date: str | None = None, **kwargs: Any) -> bool:
        if report_date and self.report_date is None:
            try:
                self.report_date = _normalise_date(report_date)
            except Exception as exc:
                self._record_unavailable(_safe_error(exc), artifact="capture-status.json", fatal=False)
        if not self._ensure_root():
            return False
        results = [self._capture_input_manifest(report_date=report_date), self.capture_runtime_settings(
            kwargs.get("runtime_settings")
        )]

        # Freeze tracked configuration inputs before Phase 0 can refresh or
        # rewrite them.  These reads are local and bounded; a missing file is
        # recorded as unavailable and never blocks the production run.
        config_dir = kwargs.get("config_dir")
        if config_dir:
            config_root = Path(config_dir)
            results.append(self.capture_prompts(prompts_path=config_root / "prompts.yaml"))
            try:
                release_path = config_root / "model_releases.yaml"
                if release_path.is_symlink() or not release_path.is_file():
                    raise FileNotFoundError(f"model release snapshot is unavailable: {release_path}")
                results.append(self.capture_grounding(model_releases=release_path.read_bytes()))
            except Exception as exc:
                self._record_unavailable(
                    _safe_error(exc), artifact="context/model-releases-before.yaml", fatal=False
                )
                results.append(False)
        return all(results)

    start_run = begin_run
    begin = begin_run

    def capture_history(
        self,
        source_root: str | Path | None = None,
        *,
        target_date: str | date | datetime | None = None,
        used_paths: Iterable[str | Path] | None = None,
        search_documents_used: bool | None = None,
    ) -> bool:
        """Freeze only history that is eligible for the report date.

        ``source_root`` is normally the repository's ``web/data`` directory.
        We copy category JSON and ``search-documents.json`` only.  The latter is
        included based on actual presence/usage, because continuity and old
        anchor code do not necessarily read the same history representation.
        Missing dates/files are recorded in ``history-manifest.json`` rather
        than silently treated as empty history.
        """
        try:
            target = _normalise_date(target_date or self.report_date)
            if source_root is None:
                raise ValueError("history source_root is required")
            source = Path(source_root).resolve()
            if not source.exists() or not source.is_dir():
                raise FileNotFoundError(f"history source directory is unavailable: {source}")
            if Path(source_root).is_symlink():
                raise ValueError("history source directory may not be a symlink")
            data_root = source / "data" if (source / "data").is_dir() else source
            if data_root.is_symlink():
                raise ValueError("history data directory may not be a symlink")
            allowed_sources: set[Path] = set()
            restrict_sources = used_paths is not None
            if used_paths is not None:
                for raw in used_paths:
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = (data_root / candidate)
                    if candidate.is_symlink():
                        raise ValueError("history source path may not be a symlink")
                    candidate = candidate.resolve()
                    if data_root != candidate and data_root not in candidate.parents:
                        raise ValueError("history path escapes source root")
                    allowed_sources.add(candidate)

            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "report_date": target.isoformat(),
                "lookback_days": MAX_HISTORY_DAYS,
                "dates": [],
                "used": [],
                "used_hashes": {},
                "missing": [],
                "not_used": [],
                "search_documents_used": bool(search_documents_used),
            }
            # Scan exact date directories, never arbitrary recursive files.
            for offset in range(1, MAX_HISTORY_DAYS + 1):
                day = target - timedelta(days=offset)
                day_text = day.isoformat()
                manifest["dates"].append(day_text)
                day_dir = data_root / day_text
                expected = [day_dir / f"{cat}.json" for cat in self.CATEGORIES]
                # Search documents are often one corpus at data root; some
                # historical runs used a per-date file.  Include only when it
                # was explicitly used.  The normal staleness path reads the
                # root corpus; per-date search files are never inferred from
                # that fact.
                candidates = list(expected)
                date_search = day_dir / "search-documents.json"
                if date_search in allowed_sources:
                    candidates.append(date_search)
                for candidate in candidates:
                    rel = candidate.relative_to(data_root).as_posix()
                    if restrict_sources and candidate.resolve() not in allowed_sources:
                        manifest["not_used"].append(rel)
                        continue
                    if candidate.is_symlink():
                        manifest["missing"].append(rel)
                        continue
                    if candidate.exists() and candidate.is_file():
                        # Validate JSON before freezing it; malformed source
                        # evidence remains explicitly unavailable.
                        raw = candidate.read_bytes()
                        json.loads(raw.decode("utf-8"))
                        artifact = f"history/{rel}"
                        if self._write_bytes(artifact, raw, kind="history"):
                            manifest["used"].append(artifact)
                            manifest["used_hashes"][artifact] = hashlib.sha256(raw).hexdigest()
                    else:
                        manifest["missing"].append(rel)
            # A root search-documents file is used by the search indexer and is
            # allowed only when the caller observed/declared that semantics.
            root_search = data_root / "search-documents.json"
            root_search_allowed = bool(search_documents_used) or root_search.resolve() in allowed_sources
            manifest["search_documents_used"] = root_search_allowed
            root_rel = "search-documents.json"
            if root_search.is_symlink():
                manifest["missing"].append(root_rel)
            elif root_search_allowed and root_search.exists() and root_search.is_file():
                raw = root_search.read_bytes()
                json.loads(raw.decode("utf-8"))
                artifact = "history/search-documents.json"
                if self._write_bytes(artifact, raw, kind="history"):
                    manifest["used"].append(artifact)
                    manifest["used_hashes"][artifact] = hashlib.sha256(raw).hexdigest()
            elif not root_search.exists() or not root_search.is_file():
                # Even when production correctly fell back to category files,
                # preserve the fact that the preferred search corpus was
                # absent.  Replay can distinguish this explicit absence from
                # an accidentally omitted dependency.
                manifest["missing"].append(root_rel)
            else:
                manifest["not_used"].append(root_rel)
            return self._write_json("context/history-manifest.json", manifest)
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact="context/history-manifest.json", fatal=False)
            return False

    capture_history_snapshot = capture_history

    def capture_gathered(
        self,
        gathered: Mapping[str, Any] | None = None,
        *,
        collection_status: Mapping[str, Any] | None = None,
        coverage: Mapping[str, Any] | None = None,
        categories: Mapping[str, Any] | None = None,
        report_date: str | date | datetime | None = None,
        **kwargs: Any,
    ) -> bool:
        """Capture ordered Phase 1 items and source health metadata."""
        if gathered is None:
            gathered = categories
        # The orchestrator's compatibility adapter may retry a keyword call as
        # ``capture_gathered(kwargs)``.  Unpack that envelope before iterating
        # categories so the observer never records four empty lists.
        if isinstance(gathered, Mapping) and (
            "categories" in gathered or "collection_status" in gathered or "coverage" in gathered
        ):
            envelope = gathered
            if categories is None:
                categories = envelope.get("categories")
            if collection_status is None:
                collection_status = envelope.get("collection_status")
            if coverage is None:
                coverage = envelope.get("coverage")
            gathered = categories
        if report_date is not None and self.report_date is None:
            try:
                self.report_date = _normalise_date(report_date)
            except Exception:
                pass
        if not coverage and self.report_date is not None:
            zone = ZoneInfo("America/New_York")
            coverage_day = self.report_date - timedelta(days=1)
            start = datetime.combine(coverage_day, time.min, tzinfo=zone)
            end = datetime.combine(self.report_date, time.min, tzinfo=zone) - timedelta(microseconds=1)
            coverage = {
                "timezone": "America/New_York",
                "coverage_date": coverage_day.isoformat(),
                "coverage_start": start.isoformat(),
                "coverage_end": end.isoformat(),
            }
        ok = True
        for category in self.CATEGORIES:
            value = (gathered or {}).get(category, [])
            # Keep the source checkpoint's list shape.  Replay validators need
            # the exact ordered CollectedItem sequence, and a wrapper here
            # would make this artifact unlike the production checkpoint.
            payload = _jsonable(value)
            ok = self._write_json(f"gathered/{category}.json", payload) and ok
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "collection_status": _jsonable(collection_status or {}),
            "coverage": _jsonable(coverage or {}),
        }
        ok = self._write_json("gathered/metadata.json", metadata) and ok
        # Coverage is learned only once gatherers have resolved their source
        # windows.  Bind it into the pre-run lineage manifest now that it is
        # known, while retaining the manifest's immutable source fields.
        if coverage:
            ok = self._update_input_manifest(coverage=coverage) and ok
        return ok

    capture_phase1 = capture_gathered

    record_gathering = capture_gathered
    capture_gathering = capture_gathered
    record_inputs = capture_gathered

    def capture_filter_input(self, value: Any = None, *, name: str = "input.json", **kwargs: Any) -> bool:
        if name != "input.json":
            name = "input.json"
        if value is None:
            value = kwargs
        # The incumbent observer supplies a compact record shape.  Normalize it
        # into the shared renderer contract while retaining any already-frozen
        # prompt strings and hash fields from a richer caller.
        if isinstance(value, Mapping) and "records" in value:
            rows = []
            for row in value.get("records") or []:
                if not isinstance(row, Mapping):
                    continue
                rows.append({
                    "id": str(row.get("id", "")),
                    "title": str(row.get("title", "")),
                    "source": str(row.get("source", "")),
                    "snippet": str(row.get("snippet", row.get("content", ""))),
                })
            value = dict(value)
            value["records"] = rows
            value.setdefault("ordered_ids", list(value.get("input_ids") or [row["id"] for row in rows]))
            value.setdefault("input_sha256", hashlib.sha256(_canonical_json(rows)).hexdigest())
            value.setdefault("system_prompt", str(value.get("system", "")))
            value.setdefault("user_message", str(value.get("user", "")))
            self._filter_input_hash = str(value.get("input_sha256") or "")
            if self._filter_input_hash:
                self._update_input_manifest(input_sha256=self._filter_input_hash)
        elif isinstance(value, Mapping) and value.get("input_sha256"):
            self._filter_input_hash = str(value["input_sha256"])
            self._update_input_manifest(input_sha256=self._filter_input_hash)
        # ``shadow.contracts.validate_input`` intentionally validates this
        # object directly (records, ordered_ids, input_sha256 and the frozen
        # prompt strings).  Do not wrap it in an observer envelope.
        return self._write_json(f"relevance/{name}", _jsonable(value))

    capture_relevance_input = capture_filter_input
    record_relevance_input = capture_filter_input
    record_filter_input = capture_filter_input

    def capture_filter_decision(self, value: Any = None, *, name: str = "incumbent-decision.json", **kwargs: Any) -> bool:
        if name != "incumbent-decision.json" and not re.fullmatch(r"incumbent-decision(?:-[A-Za-z0-9_.-]+)?\.json", name):
            name = "incumbent-decision.json"
        if value is None:
            value = kwargs
        if isinstance(value, Mapping) and "decisions" not in value:
            value = dict(value)
            input_ids = [str(v) for v in value.get("input_ids", [])]
            kept = {str(v) for v in value.get("mapped_kept_ids", value.get("selected_ids", []))}
            fallback_superset = value.get("fallback") == "superset"
            value["decisions"] = [
                {
                    "id": article_id,
                    "decision": (
                        "abstain" if fallback_superset and article_id in kept
                        else ("keep" if article_id in kept else "reject")
                    ),
                    "effective_keep": article_id in kept,
                }
                for article_id in input_ids
            ]
            value.setdefault("input_sha256", str(value.get("input_hash", "")) or self._filter_input_hash or "")
        if isinstance(value, Mapping):
            value = dict(value)
            value.setdefault("model_identity_basis", "client_configured_not_wire_verified")
        return self._write_json(f"relevance/{name}", _jsonable(value))

    capture_relevance_decision = capture_filter_decision
    record_relevance_decision = capture_filter_decision
    record_decision = capture_filter_decision

    def capture_filter_attempts(self, attempts: Any = None, **kwargs: Any) -> bool:
        """Persist bounded structured filter-attempt usage when supplied."""
        if attempts is None:
            attempts = kwargs
        if isinstance(attempts, Mapping) and "attempts" in attempts:
            attempts = attempts["attempts"]
        if not isinstance(attempts, (list, tuple)):
            attempts = [attempts]
        return self._write_json("diagnostics/filter-attempts.json", _jsonable(list(attempts)))

    record_filter_attempts = capture_filter_attempts
    capture_filter_metrics = capture_filter_attempts

    def capture_precontinuity(self, reports: Mapping[str, Any] | None = None, **kwargs: Any) -> bool:
        if reports is None:
            reports = kwargs.get("category_reports", kwargs)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "reports": _jsonable(reports or {}),
        }
        return self._write_json("checkpoints/analysis_pre_continuity.json", payload)

    capture_analysis_pre_continuity = capture_precontinuity

    record_pre_continuity = capture_precontinuity

    def capture_original_outputs(
        self,
        outputs: Mapping[str, Any] | None = None,
        *,
        source_dir: str | Path | None = None,
    ) -> bool:
        """Capture the five report JSON outputs by value, with strict names."""
        ok = True
        outputs = outputs or {}
        for raw_name, value in outputs.items():
            name = Path(str(raw_name)).name
            if not _OUTPUT_RE.fullmatch(name):
                self._record_unavailable("output name is outside the allowlist", artifact=f"original/{name}", fatal=False)
                ok = False
                continue
            try:
                if source_dir is not None and value is None:
                    value = Path(source_dir) / name
                if isinstance(value, (str, Path)):
                    candidate = Path(value)
                    if not candidate.exists() or candidate.is_symlink():
                        raise FileNotFoundError(f"original output is unavailable: {candidate}")
                    raw = candidate.read_bytes()
                elif isinstance(value, bytes):
                    raw = value
                else:
                    raw = _canonical_json(value) + b"\n"
                ok = self._write_bytes(f"original/{name}", raw, kind="original output") and ok
            except Exception as exc:
                self._record_unavailable(_safe_error(exc), artifact=f"original/{name}", fatal=False)
                ok = False
        return ok

    capture_outputs = capture_original_outputs

    def record_finished_output(
        self,
        *,
        report_date: str | None = None,
        files: Mapping[str, Any] | None = None,
        result: Any = None,
        output_dir: str | Path | None = None,
        web_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> bool:
        """Record output hashes supplied by the post-generator observer hook.

        The hook receives hashes rather than paths in the current orchestrator;
        preserve those bounded facts in a receipt.  A caller with output bytes
        should use :meth:`capture_original_outputs` directly.
        """
        if output_dir is None and web_dir is not None and report_date:
            output_dir = Path(web_dir) / "data" / report_date
        if output_dir is not None:
            copied = self.capture_original_outputs(
                {name: Path(output_dir) / name for name in self.OUTPUTS}
            )
            hero = Path(output_dir) / "hero.webp"
            if hero.is_file() and not hero.is_symlink():
                try:
                    copied = self._write_bytes("original/hero.webp", hero.read_bytes(), kind="original hero") and copied
                except Exception as exc:
                    self._record_unavailable(_safe_error(exc), artifact="original/hero.webp", fatal=False)
                    copied = False
        else:
            copied = True
        payload = {"report_date": report_date, "files": _jsonable(files or {}), "result": _jsonable(result or {})}
        return self._write_json("original/output-receipt.json", payload) and copied

    capture_finished_output = record_finished_output
    record_output = record_finished_output

    def finish_run(self, **kwargs: Any) -> bool:
        return self.finalize_status()

    complete = finish_run
    finalize = finish_run

    def capture_pre_run(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        history_root: str | Path | None = None,
        grounding_text: str | None = None,
        model_releases: Any = None,
        resolved_ecosystem: Any = None,
        prompts: str | bytes | Path | None = None,
        prompts_path: str | Path | None = None,
        runtime_settings: Mapping[str, Any] | None = None,
        target_date: str | date | datetime | None = None,
        used_history_paths: Iterable[str | Path] | None = None,
        search_documents_used: bool | None = None,
    ) -> bool:
        """Convenience entry point for all pre-run immutable dependencies."""
        results = [
            self.capture_config(config if config is not None else runtime_settings),
            self.capture_runtime_settings(runtime_settings),
            self.capture_grounding(
                grounding_text,
                model_releases=model_releases,
                resolved_ecosystem=resolved_ecosystem,
            ),
        ]
        if prompts is not None or prompts_path is not None:
            results.append(self.capture_prompts(prompts, prompts_path=prompts_path))
        if history_root is not None:
            results.append(
                self.capture_history(
                    history_root,
                    target_date=target_date,
                    used_paths=used_history_paths,
                    search_documents_used=search_documents_used,
                )
            )
        return all(results)

    def status(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "available": self.available,
            "capture_enabled": self.capture_enabled,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "bytes_written": self._bytes_written,
            "artifacts": list(self.written.values()),
            "unavailable": list(self.failures),
        }

    capture_status = status

    def finalize_status(self) -> bool:
        """Persist observer status; status persistence itself remains best effort."""
        return self._write_json("diagnostics/capture-status.json", self.status())

    def observe(self, callback: Callable[[], Any], *, artifact: str | None = None) -> Any:
        """Run an optional observation without ever propagating its failure."""
        if not self.enabled:
            return None
        try:
            return callback()
        except Exception as exc:
            self._record_unavailable(_safe_error(exc), artifact=artifact, fatal=False)
            return None


ExperimentCapture = CaptureSession

__all__ = [
    "CaptureSession",
    "ExperimentCapture",
    "MAX_HISTORY_DAYS",
    "RUNTIME_ENV_KEYS",
    "SCHEMA_VERSION",
    "_canonical_json",
    "_jsonable",
    "_safe_relative_path",
    "_sanitize_config",
]
