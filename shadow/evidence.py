"""Frozen HTTP evidence for the existing staleness checker.

``StalenessChecker._safe_get`` has a deliberately small contract: it accepts
``(url, timeout=12, max_redirects=5)`` and returns a requests-like response
whose ``headers``, ``status_code``, ``text``, ``content``, ``is_redirect`` and
``raise_for_status`` attributes are used by the checker.  ``EvidenceStore``
records that response in production and returns an equivalent frozen response
during replay.  Replay never performs a network request.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

from .capture import SCHEMA_VERSION, _canonical_json, _safe_error, _safe_relative_path

try:  # The repository already depends on requests, but keep import-time safety.
    import requests
except Exception:  # pragma: no cover - only used in a stripped-down harness
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 4096
_SAFE_HEADERS = frozenset(
    {
        "content-type",
        "content-length",
        "content-encoding",
        "charset",
        "etag",
        "last-modified",
        "cache-control",
        "location",
        "date",
        "expires",
        "server",
    }
)
_TOKEN_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|cookie)=([^&\s]+)")


class ReplayIntegrityError(RuntimeError):
    """A required frozen dependency is absent or failed integrity checks."""


class EvidenceError(RuntimeError):
    """Capture/replay evidence operation error."""


class FrozenHTTPError(EvidenceError):
    pass


class FrozenResponse:
    """Small requests.Response-compatible value used by replay freshness checks."""

    def __init__(
        self,
        *,
        url: str,
        status_code: int,
        headers: Mapping[str, Any] | None = None,
        content: bytes = b"",
        reason: str | None = None,
        encoding: str | None = None,
        history: Iterable[Any] | None = None,
    ):
        self.url = str(url)
        self.status_code = int(status_code)
        self.headers = {str(k): str(v) for k, v in (headers or {}).items()}
        self._content = bytes(content)
        self.reason = reason or ""
        self.encoding = encoding or self._encoding_from_headers()
        self.history = list(history or [])
        self._closed = False

    def _encoding_from_headers(self) -> str | None:
        value = self.headers.get("Content-Type") or self.headers.get("content-type") or ""
        match = re.search(r"charset\s*=\s*['\"]?([\w.-]+)", value, flags=re.IGNORECASE)
        return match.group(1) if match else None

    @property
    def content(self) -> bytes:
        return self._content

    @property
    def text(self) -> str:
        encoding = self.encoding or "utf-8"
        try:
            return self._content.decode(encoding, errors="replace")
        except (LookupError, UnicodeError):
            return self._content.decode("utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def is_redirect(self) -> bool:
        return self.status_code in {301, 302, 303, 307, 308}

    @property
    def is_permanent_redirect(self) -> bool:
        return self.status_code in {308, 301}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            if requests is not None:
                raise requests.HTTPError(f"{self.status_code} response for frozen URL", response=self)
            raise FrozenHTTPError(f"{self.status_code} response for frozen URL")

    def iter_content(self, chunk_size: int = 65536, **_: Any):
        size = max(1, int(chunk_size))
        for offset in range(0, len(self._content), size):
            yield self._content[offset : offset + size]

    def close(self) -> None:
        self._closed = True

    def json(self, **kwargs: Any) -> Any:
        return json.loads(self.text, **kwargs)

    def __enter__(self) -> "FrozenResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _redact_error(value: Any) -> str:
    text = _safe_error(RuntimeError(str(value)))
    return _TOKEN_RE.sub(r"\1=<redacted>", text)[:1000]


def _response_headers(response: Any) -> dict[str, str]:
    raw = getattr(response, "headers", {})
    if not isinstance(raw, Mapping):
        return {}
    result = {}
    for key, value in raw.items():
        name = str(key)
        if name.lower() in _SAFE_HEADERS:
            result[name] = str(value)[:4096]
    return result


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("HTTP response content is not bytes")
    content = bytes(content)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("HTTP response exceeds the frozen evidence size cap")
    return content


class EvidenceStore:
    """Capture and resolve content-addressed staleness evidence.

    ``mode='capture'`` accepts observations through :meth:`record_response` or
    a wrapper around the incumbent ``_safe_get``.  ``mode='replay'`` loads only
    ``evidence/index.json`` and its referenced objects; :meth:`safe_get` then
    returns frozen data or raises :class:`ReplayIntegrityError`.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        mode: str = "capture",
        strict: bool = True,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ):
        if mode not in {"capture", "replay"}:
            raise ValueError("EvidenceStore mode must be capture or replay")
        self.root = Path(root) if root is not None else None
        self.mode = mode
        # Capture is always best-effort: a cache miss must proceed to the
        # incumbent network fetch so the observer cannot block publication.
        self.strict = bool(strict) if mode == "replay" else False
        self.max_response_bytes = min(MAX_RESPONSE_BYTES, max(1, int(max_response_bytes)))
        self.max_total_bytes = min(MAX_TOTAL_BYTES, max(1, int(max_total_bytes)))
        self.entries: dict[str, dict[str, Any]] = {}
        self._objects: dict[str, dict[str, Any]] = {}
        self._bytes_written = 0
        self._index_loaded = False
        self._index_error: ReplayIntegrityError | None = None
        if self.mode == "replay":
            # Keep loading lazy so a filter-only bundle can be opened and
            # compared even when no downstream freshness evidence was needed.
            # A later strict lookup still fails closed with the original error.
            try:
                self._load_index()
            except ReplayIntegrityError as exc:
                self._index_error = exc

    @property
    def index_path(self) -> Path | None:
        return _safe_relative_path(self.root, "evidence/index.json") if self.root is not None else None

    @property
    def replay(self) -> bool:
        return self.mode == "replay"

    @property
    def frozen(self) -> bool:
        return self.mode == "replay"

    def _load_index(self) -> None:
        if self.root is None:
            raise ReplayIntegrityError("Replay evidence root was not supplied")
        try:
            path = self.index_path
            if path is None or not path.exists() or path.is_symlink():
                raise ReplayIntegrityError("Frozen evidence index is missing")
            index = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION:
                raise ReplayIntegrityError("Unsupported frozen evidence index")
            entries = index.get("entries")
            if not isinstance(entries, dict) or len(entries) > MAX_ENTRIES:
                raise ReplayIntegrityError("Malformed frozen evidence index")
            self.entries = {str(url): dict(row) for url, row in entries.items() if isinstance(row, dict)}
            if len(self.entries) != len(entries):
                raise ReplayIntegrityError("Malformed frozen evidence entries")
            self._index_loaded = True
            self._index_error = None
        except ReplayIntegrityError:
            raise
        except Exception as exc:
            raise ReplayIntegrityError(f"Frozen evidence index cannot be read: {_redact_error(exc)}") from exc

    def _write_json(self, relative: str, value: Any) -> None:
        if self.root is None:
            return
        path = _safe_relative_path(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _canonical_json(value) + b"\n"
        # Base64 expands a maximum response by roughly one third.  Metadata is
        # still bounded, but the cap must account for that representation.
        if len(payload) > self.max_response_bytes * 2:
            raise ValueError("Frozen evidence metadata exceeds size cap")
        if path.exists() and path.read_bytes() != payload and relative != "evidence/index.json":
            raise ValueError(f"Frozen evidence artifact is immutable: {relative}")
        if not path.exists() or relative == "evidence/index.json":
            path.write_bytes(payload)

    def _persist_index(self) -> None:
        if self.root is None:
            return
        self._write_json(
            "evidence/index.json",
            {
                "schema_version": SCHEMA_VERSION,
                "entries": self.entries,
                "objects": len(self._objects),
                "bytes": self._bytes_written,
            },
        )

    def _capture_url(self, url: str) -> str:
        if not isinstance(url, str) or not url or len(url) > 8192:
            raise ValueError("Evidence URL is invalid")
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Evidence URL must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("Evidence URL must not contain credentials")
        return url

    def record_response(
        self,
        url: str,
        response: Any,
        *,
        request_url: str | None = None,
        timeout: int | float | None = None,
        max_redirects: int | None = None,
    ) -> dict[str, Any] | None:
        """Record one final response and return its index row."""
        if self.mode != "capture":
            raise EvidenceError("Cannot record HTTP evidence in replay mode")
        try:
            requested = self._capture_url(request_url or url)
            final_url = self._capture_url(str(getattr(response, "url", url) or url))
            body = _response_bytes(response)
            if len(body) > self.max_response_bytes:
                raise ValueError("HTTP response exceeds configured evidence cap")
            digest = hashlib.sha256(body).hexdigest()
            status = int(getattr(response, "status_code", 0))
            metadata = {
                "kind": "response",
                "url": requested,
                "final_url": final_url,
                "status_code": status,
                "reason": str(getattr(response, "reason", "") or "")[:200],
                "headers": _response_headers(response),
                "encoding": getattr(response, "encoding", None),
                "body_sha256": digest,
                "body_bytes": len(body),
                "body_b64": base64.b64encode(body).decode("ascii"),
            }
            if timeout is not None:
                metadata["timeout"] = float(timeout)
            if max_redirects is not None:
                metadata["max_redirects"] = int(max_redirects)
            if self.root is not None:
                object_is_new = digest not in self._objects and not _safe_relative_path(
                    self.root, f"evidence/objects/{digest}.json"
                ).exists()
                projected = self._bytes_written + (len(body) if object_is_new else 0)
                if projected > self.max_total_bytes:
                    raise ValueError("Frozen evidence exceeds total size cap")
                object_name = f"evidence/objects/{digest}.json"
                self._write_json(object_name, metadata)
                self._bytes_written = projected
            self._objects[digest] = {"sha256": digest, "bytes": len(body)}
            row = {
                "kind": "response",
                "object": digest,
                "status_code": status,
                "final_url": final_url,
                "body_sha256": digest,
                "body_bytes": len(body),
            }
            self.entries[requested] = row
            self._persist_index()
            return row
        except Exception as exc:
            self.record_failure(url, exc, request_url=request_url)
            return None

    record_http_response = record_response
    record_success = record_response

    def record_failure(
        self,
        url: str,
        error: Any,
        *,
        request_url: str | None = None,
        timeout: int | float | None = None,
        max_redirects: int | None = None,
    ) -> dict[str, Any] | None:
        if self.mode != "capture":
            return None
        try:
            requested = self._capture_url(request_url or url)
            row: dict[str, Any] = {
                "kind": "failure",
                "error_type": type(error).__name__,
                "error": _redact_error(error),
            }
            if timeout is not None:
                row["timeout"] = float(timeout)
            if max_redirects is not None:
                row["max_redirects"] = int(max_redirects)
            self.entries[requested] = row
            self._persist_index()
            return row
        except Exception as exc:
            logger.warning("Unable to record frozen evidence failure: %s", _redact_error(exc))
            return None

    record_http_failure = record_failure

    def record_http(self, url: str, record: Any) -> dict[str, Any] | None:
        """Adapter for StalenessChecker's compact ``record_http`` hook."""
        if not isinstance(record, Mapping):
            return self.record_failure(url, TypeError("unsupported evidence record"))
        error = record.get("error")
        if error:
            return self.record_failure(url, error, request_url=record.get("url"))
        body = record.get("body", record.get("text", ""))
        if isinstance(body, str):
            body = body.encode("utf-8")
        response = FrozenResponse(
            url=str(record.get("final_url") or record.get("url") or url),
            status_code=int(record.get("status_code", record.get("status", 200))),
            headers=record.get("headers") or {},
            content=bytes(body or b""),
        )
        return self.record_response(url, response, request_url=record.get("url"))

    def contains(self, url: str) -> bool:
        if self.mode == "replay" and not self._index_loaded and self._index_error is None:
            self._load_index()
        return url in self.entries

    has = contains
    has_url = contains
    contains_url = contains
    has_evidence = contains

    def _object_metadata(self, digest: str) -> dict[str, Any]:
        if self.root is None:
            raise ReplayIntegrityError("Frozen evidence root is unavailable")
        path = _safe_relative_path(self.root, f"evidence/objects/{digest}.json")
        if not path.exists() or path.is_symlink():
            raise ReplayIntegrityError(f"Frozen evidence object is missing: {digest}")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            body = base64.b64decode(metadata.get("body_b64", ""), validate=True)
            if hashlib.sha256(body).hexdigest() != digest or metadata.get("body_sha256") != digest:
                raise ReplayIntegrityError(f"Frozen evidence object hash mismatch: {digest}")
            if len(body) > self.max_response_bytes:
                raise ReplayIntegrityError("Frozen evidence object exceeds size cap")
            metadata["_body"] = body
            return metadata
        except ReplayIntegrityError:
            raise
        except Exception as exc:
            raise ReplayIntegrityError(f"Frozen evidence object is invalid: {_redact_error(exc)}") from exc

    def lookup(self, url: str, *, strict: bool | None = None) -> FrozenResponse | None:
        """Resolve a captured response, or ``None`` for non-strict lookup."""
        strict = self.strict if strict is None else bool(strict)
        if self.mode == "replay" and not self._index_loaded:
            if self._index_error is not None:
                raise self._index_error
            self._load_index()
        try:
            requested = self._capture_url(url)
        except Exception as exc:
            if strict:
                raise ReplayIntegrityError(_redact_error(exc)) from exc
            return None
        row = self.entries.get(requested)
        if row is None:
            if strict:
                raise ReplayIntegrityError(f"No frozen HTTP evidence for {requested}")
            return None
        if row.get("kind") == "failure":
            if strict:
                raise ReplayIntegrityError(
                    f"Original HTTP request failed for {requested}: {row.get('error', 'unknown failure')}"
                )
            return None
        digest = row.get("object")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ReplayIntegrityError("Malformed frozen evidence object reference")
        metadata = self._object_metadata(digest)
        return FrozenResponse(
            url=str(metadata.get("final_url") or row.get("final_url") or requested),
            status_code=int(metadata.get("status_code", row.get("status_code", 0))),
            headers=metadata.get("headers") or {},
            content=metadata["_body"],
            reason=str(metadata.get("reason") or ""),
            encoding=metadata.get("encoding"),
        )

    resolve = lookup
    get = lookup
    response_for = lookup
    resolve_http = lookup
    lookup_http = lookup
    lookup_evidence = lookup
    get_evidence = lookup
    get_response = lookup

    def safe_get(
        self,
        url: str,
        timeout: int = 12,
        max_redirects: int = 5,
    ) -> FrozenResponse:
        """Strict frozen replacement for ``StalenessChecker._safe_get``."""
        if self.mode != "replay":
            raise EvidenceError("Capture-mode safe_get requires an original fetcher wrapper")
        response = self.lookup(url, strict=True)
        if response is None:  # Defensive; strict lookup raises above.
            raise ReplayIntegrityError(f"No frozen HTTP evidence for {url}")
        return response

    def make_safe_get(self) -> Callable[..., FrozenResponse]:
        return self.safe_get

    def wrap_safe_get(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Instrument an incumbent ``_safe_get`` with the exact existing signature."""
        def observed(url: str, timeout: int = 12, max_redirects: int = 5):
            try:
                response = original(url, timeout=timeout, max_redirects=max_redirects)
            except Exception as exc:
                self.record_failure(url, exc, timeout=timeout, max_redirects=max_redirects)
                raise
            self.record_response(url, response, timeout=timeout, max_redirects=max_redirects)
            return response

        return observed

    instrument_safe_get = wrap_safe_get
    capture_safe_get = wrap_safe_get

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "entries": self.entries,
            "objects": len(self._objects),
            "bytes": self._bytes_written,
        }


__all__ = [
    "EvidenceError",
    "EvidenceStore",
    "FrozenHTTPError",
    "FrozenResponse",
    "ReplayIntegrityError",
]
