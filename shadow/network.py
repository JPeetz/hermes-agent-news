"""Fail-closed network boundary for isolated shadow model calls.

The shadow runner is deliberately a small Python process.  This module gives
that process a useful second line of defence: while :func:`model_egress_only`
is active, the HTTP clients used by a model adapter may issue only an HTTPS
``POST`` to one of the exact endpoint URLs supplied by the runner.  Redirects,
catalog/source requests, proxy environment variables, and direct socket
connections to other destinations are rejected with ``ReplayIntegrityError``.

This is an in-process guard, not an operating-system sandbox.  The runner
still needs the normal subprocess/container isolation and should clear proxy
environment variables before constructing model clients.  Keeping the guard
here avoids importing the production ``agents`` package (which initializes
the production LLM client on import).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import ipaddress
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import SplitResult, parse_qs, urlencode, urljoin, urlsplit, urlunsplit

try:  # ``budget`` is intentionally dependency-light.
    from .budget import RequestBudget
except Exception:  # pragma: no cover - useful for isolated source loading
    RequestBudget = Any  # type: ignore[misc,assignment]


try:
    # The capture/replay worker owns the canonical exception class.  Importing
    # it is safe once that module exists; the fallback keeps this module useful
    # in dependency-light guard jobs while the worker is being assembled.
    from .replay_context import ReplayIntegrityError  # type: ignore
except Exception:

    class ReplayIntegrityError(RuntimeError):
        """Fallback error used until ``shadow.replay_context`` is available."""


    # The helper below resolves the canonical class at the point of failure so
    # a network module imported before that worker module still raises the
    # canonical class once it exists.


def _integrity_error(message: str) -> BaseException:
    """Construct the canonical replay error without creating an import cycle."""

    try:
        module = importlib.import_module("shadow.replay_context")
        error_type = getattr(module, "ReplayIntegrityError", ReplayIntegrityError)
    except Exception:  # pragma: no cover - fallback for dependency-light jobs
        error_type = ReplayIntegrityError
    return error_type(message)


def _raise_integrity(message: str) -> None:
    raise _integrity_error(message)


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)

_MODEL_PATH_SUFFIXES = ("/chat/completions", "/systemone")
_HTTP_PORTS = {"http": 80, "https": 443}
_MISSING = object()


def _safe_host(value: str | None) -> str:
    """Return a comparison form for a URL/DNS hostname.

    URL parsing already strips IPv6 brackets.  A trailing DNS dot is retained
    deliberately: the allowlist is exact, so ``model.example`` and
    ``model.example.`` are separate names.
    """

    if value is None:
        return ""
    return str(value).lower()


def _port_from_url(parts: SplitResult) -> int:
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("endpoint URL has an invalid port") from exc
    return 443 if port is None else int(port)


def _is_model_path(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in _MODEL_PATH_SUFFIXES)


@dataclass(frozen=True)
class _Endpoint:
    scheme: str
    host: str
    port: int
    path: str

    @classmethod
    def parse(cls, raw: Any, *, declared: bool = False) -> "_Endpoint":
        text = str(raw)
        try:
            parts = urlsplit(text)
            # Accessing username/password/hostname can raise on malformed
            # bracketed IPv6 forms.  Treat all such forms as invalid.
            username = parts.username
            password = parts.password
            host = parts.hostname
            port = _port_from_url(parts)
        except (AttributeError, ValueError) as exc:
            raise ValueError("invalid model endpoint URL") from exc

        if parts.scheme.lower() != "https":
            raise ValueError("model endpoint must use HTTPS")
        if not host:
            raise ValueError("model endpoint must include a hostname")
        if username is not None or password is not None or parts.netloc.startswith("@"):
            raise ValueError("model endpoint may not contain URL userinfo")
        if parts.query or parts.fragment or "?" in text or "#" in text:
            raise ValueError("model endpoint may not contain query or fragment")
        if not parts.path or not _is_model_path(parts.path):
            raise ValueError("model endpoint must be chat/completions or systemone")
        if not 1 <= port <= 65535:
            raise ValueError("model endpoint port is out of range")
        if ":" in host and "%" in host:
            # Zone-scoped IPv6 destinations are a separate local-network
            # address and should never enter a portable replay allowlist.
            raise ValueError("model endpoint may not use an IPv6 zone")
        result = cls(parts.scheme.lower(), _safe_host(host), port, parts.path)
        return result

    @property
    def url(self) -> str:
        # Keep the URL useful in diagnostics while avoiding credentials (which
        # the parser has already rejected).
        netloc = self.host
        try:
            if ipaddress.ip_address(self.host).version == 6:
                netloc = f"[{self.host}]"
        except ValueError:
            pass
        if self.port != 443:
            netloc = f"{netloc}:{self.port}"
        return urlunsplit((self.scheme, netloc, self.path, "", ""))


@dataclass(frozen=True)
class _RequestBounds:
    """Conservative token bounds for one admitted model HTTP attempt."""

    input_tokens: int
    output_tokens: int


def _body_bytes(value: Any, *, kind: str) -> bytes:
    """Encode an already-buffered request body without reading streams."""

    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    if kind == "json":
        try:
            # The two encodings cover the common requests/httpx serializers;
            # retain the larger one as a conservative UTF-8 byte upper bound.
            compact = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            default = json.dumps(value, ensure_ascii=True).encode("utf-8")
            return compact if len(compact) >= len(default) else default
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("model request JSON body is not serializable") from exc
    if kind == "data" and isinstance(value, dict):
        try:
            return urlencode(value, doseq=True).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("model request form body is not serializable") from exc
    raise ValueError("model request body is not a buffered JSON/data payload")


def _request_body_from_kwargs(kwargs: dict[str, Any]) -> bytes:
    """Extract a buffered request body from requests/httpx/aiohttp kwargs."""

    if kwargs.get("files"):
        raise ValueError("multipart model request body is not bounded")
    if kwargs.get("json") is not None:
        return _body_bytes(kwargs.get("json"), kind="json")
    if kwargs.get("content") is not None:
        return _body_bytes(kwargs.get("content"), kind="content")
    if kwargs.get("data") is not None:
        return _body_bytes(kwargs.get("data"), kind="data")
    # A POST without a body is known to be empty.  It still fails below because
    # no output cap can be recovered from it, but it does not need a stream read.
    return b""


def _request_body_from_prepared(request: Any) -> bytes:
    body = getattr(request, "content", _MISSING)
    if body is _MISSING:
        body = getattr(request, "body", _MISSING)
    if body is _MISSING:
        raise ValueError("model request body is unavailable")
    return _body_bytes(body, kind="content")


def _output_bound(body: bytes) -> int:
    """Read only the already-buffered request JSON for its output ceiling."""

    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        # ``requests(data={...})`` is form encoded.  It is unusual for a model
        # API, but it is still a fully buffered body whose cap can be bounded
        # without consuming any response content.
        try:
            form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError as exc:
            raise ValueError("model request has no bounded JSON output cap") from exc
        decoded = {key: values[-1] for key, values in form.items() if values}
        form_encoded = True
    else:
        form_encoded = False
    if not isinstance(decoded, dict):
        raise ValueError("model request has no bounded JSON output cap")
    present: list[int] = []
    for key in ("max_tokens", "max_completion_tokens"):
        if key not in decoded:
            continue
        value = decoded[key]
        if form_encoded and isinstance(value, str):
            try:
                value = int(value)
            except ValueError as exc:
                raise ValueError("model request output cap is invalid") from exc
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("model request output cap is invalid")
        present.append(value)
    if not present:
        raise ValueError("model request has no max_tokens/max_completion_tokens cap")
    return max(present)


def _request_bounds(body: bytes) -> _RequestBounds:
    try:
        input_bytes = len(body)
        output_tokens = _output_bound(body)
    except ValueError:
        # Preserve one stable error path for callers.  The body itself is never
        # included in the exception, since it may contain authorization data.
        raise
    return _RequestBounds(input_tokens=input_bytes, output_tokens=output_tokens)


def _load_endpoints(allowed_urls: Iterable[str]) -> tuple[_Endpoint, ...]:
    if isinstance(allowed_urls, (str, bytes)):
        raise ValueError("allowed_urls must be an iterable of endpoint URLs")
    result: list[_Endpoint] = []
    seen: set[_Endpoint] = set()
    for raw in allowed_urls:
        endpoint = _Endpoint.parse(raw, declared=True)
        if endpoint not in seen:
            result.append(endpoint)
            seen.add(endpoint)
    if not result:
        raise ValueError("at least one model endpoint is required")
    return tuple(result)


def _method_text(method: Any) -> str:
    if isinstance(method, bytes):
        return method.decode("ascii", "replace").upper()
    return str(method).upper()


def _url_text(url: Any) -> str:
    # httpx.URL, yarl.URL and requests.PreparedRequest all provide a useful
    # string representation.  PreparedRequest.url is preferred when present.
    if hasattr(url, "url") and not isinstance(url, (str, bytes)):
        try:
            candidate = getattr(url, "url")
            if candidate:
                return str(candidate)
        except Exception:
            pass
    if isinstance(url, bytes):
        return url.decode("utf-8", "replace")
    return str(url)


def _join_base_url(base: Any, url: Any) -> str:
    """Resolve a client-relative URL without weakening endpoint validation."""

    raw = _url_text(url)
    parts = urlsplit(raw)
    if parts.scheme or parts.netloc:
        return raw
    if base is None:
        return raw
    try:
        join = getattr(base, "join", None)
        if callable(join):
            return str(join(raw))
    except Exception:
        pass
    return urljoin(str(base), raw)


class _NetworkGuard:
    """Mutable state shared by transport and socket wrappers."""

    def __init__(self, endpoints: tuple[_Endpoint, ...], budget: Any = None):
        self.endpoints = endpoints
        self.budget = budget
        self._endpoint_keys = {(e.scheme, e.host, e.port, e.path) for e in endpoints}
        self._hosts = {e.host for e in endpoints}
        self._ports_by_host: dict[str, set[int]] = {}
        for endpoint in endpoints:
            self._ports_by_host.setdefault(endpoint.host, set()).add(endpoint.port)
        self._literal_hosts = {
            e.host for e in endpoints if _is_ip_literal(e.host)
        }
        self._resolved_ips: set[tuple[str, int]] = set()
        self._reserved: list[Any] = []

    def _endpoint_for(self, method: Any, url: Any) -> _Endpoint:
        if _method_text(method) != "POST":
            _raise_integrity("shadow model egress permits HTTPS POST only")
        text = _url_text(url)
        try:
            endpoint = _Endpoint.parse(text)
        except ValueError as exc:
            _raise_integrity(f"undeclared or invalid model endpoint: {type(exc).__name__}")
        if (endpoint.scheme, endpoint.host, endpoint.port, endpoint.path) not in self._endpoint_keys:
            _raise_integrity("model endpoint is not in the declared shadow allowlist")
        return endpoint

    def check_http(self, method: Any, url: Any) -> _Endpoint:
        return self._endpoint_for(method, url)

    def check_redirects(self, value: Any, *, name: str) -> None:
        if value is True:
            _raise_integrity(f"shadow model egress forbids redirects ({name}=true)")

    def check_proxy(self, value: Any, *, name: str = "proxy") -> None:
        if value:
            _raise_integrity(f"shadow model egress forbids explicit proxies ({name})")

    def bounds_from_kwargs(self, kwargs: dict[str, Any]) -> _RequestBounds | None:
        if self.budget is None:
            return None
        try:
            return _request_bounds(_request_body_from_kwargs(kwargs))
        except Exception as exc:
            _raise_integrity(f"shadow model request is not bounded: {type(exc).__name__}")

    def bounds_from_prepared(self, request: Any) -> _RequestBounds | None:
        if self.budget is None:
            return None
        try:
            return _request_bounds(_request_body_from_prepared(request))
        except Exception as exc:
            _raise_integrity(f"shadow model request is not bounded: {type(exc).__name__}")

    def reserve_attempt(self, bounds: _RequestBounds | None = None) -> Any:
        """Consume one request slot without consuming a streaming response.

        Reservations use conservative request-body bytes and the explicit
        output cap from the request.  Usage is intentionally left unknown: a
        model adapter may reconcile measured usage later, while the guard
        never reads response content and therefore cannot corrupt streaming
        bodies.
        """

        if self.budget is None:
            return None
        if bounds is None:
            _raise_integrity("shadow model request has no bounded token reservation")
        reserve = getattr(self.budget, "reserve", None)
        settle = getattr(self.budget, "settle", None)
        if not callable(reserve) or not callable(settle):
            _raise_integrity("shadow network budget does not implement RequestBudget")
        reservation = reserve(
            input_tokens=bounds.input_tokens,
            output_tokens=bounds.output_tokens,
        )
        self._reserved.append(reservation)
        return reservation

    def settle_attempt(self, reservation: Any) -> None:
        if reservation is None or self.budget is None:
            return
        settle = getattr(self.budget, "settle", None)
        if callable(settle):
            try:
                settle(reservation)
            except ValueError:
                # A model adapter may have already reconciled measured usage.
                # Do not let cleanup hide the original response/error.
                pass

    def attempt(
        self,
        method: Any,
        url: Any,
        operation: Callable[[], Any],
        bounds: _RequestBounds | None = None,
    ) -> Any:
        self.check_http(method, url)
        reservation = self.reserve_attempt(bounds)
        try:
            result = operation()
        except BaseException:
            self.settle_attempt(reservation)
            raise
        # Do not inspect/read result.  For streamed responses this is an
        # unknown-usage reservation against the conservative bounds, which is
        # the only safe settlement here.
        self.settle_attempt(reservation)
        return result

    def check_dns(self, host: Any, port: Any = None) -> None:
        if host is None:
            # ``getaddrinfo(None, ...)`` is used for local bind/listen setup and
            # does not represent model egress.
            return
        name = _safe_host(host.decode("ascii", "replace") if isinstance(host, bytes) else host)
        if not name:
            _raise_integrity("shadow model egress rejected an empty DNS name")
        if name not in self._hosts and not _is_allowed_ip(name, self._literal_hosts):
            _raise_integrity("shadow model egress rejected undeclared DNS access")
        requested_port = _normalise_port(port)
        if requested_port is not None:
            allowed = self._ports_by_host.get(name)
            if allowed is None and _is_allowed_ip(name, self._literal_hosts):
                allowed = {port_value for endpoint in self.endpoints if endpoint.host == name
                           for port_value in (endpoint.port,)}
            if allowed and requested_port not in allowed:
                _raise_integrity("shadow model egress rejected an undeclared destination port")

    def record_resolution(self, host: Any, port: Any, infos: Any) -> None:
        name = _safe_host(host.decode("ascii", "replace") if isinstance(host, bytes) else host)
        requested_port = _normalise_port(port)
        allowed_ports = self._ports_by_host.get(name, set())
        for info in infos or ():
            try:
                sockaddr = info[4]
                ip = _safe_host(sockaddr[0])
                resolved_port = _normalise_port(sockaddr[1] if len(sockaddr) > 1 else requested_port)
            except (IndexError, TypeError, ValueError):
                continue
            if not ip:
                continue
            if resolved_port is not None:
                if not allowed_ports or resolved_port in allowed_ports:
                    self._resolved_ips.add((ip, resolved_port))
            else:
                # A caller may resolve first with ``port=None`` and connect
                # later.  Preserve the endpoint's permitted ports while still
                # rejecting a different port at connect time.
                for allowed_port in allowed_ports:
                    self._resolved_ips.add((ip, allowed_port))

    def check_connect(self, address: Any, family: Any = None) -> None:
        if not isinstance(address, tuple) or len(address) < 2:
            _raise_integrity("shadow model egress rejected a non-IP socket destination")
        host = _safe_host(address[0])
        port = _normalise_port(address[1])
        if port is None:
            _raise_integrity("shadow model egress rejected an invalid socket port")
        if host in self._hosts and port in self._ports_by_host.get(host, set()):
            return
        if (host, port) in self._resolved_ips:
            return
        if _is_allowed_ip(host, self._literal_hosts):
            for endpoint in self.endpoints:
                if endpoint.host == host and endpoint.port == port:
                    return
        _raise_integrity("shadow model egress rejected an undeclared socket destination")


def _is_ip_literal(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _is_allowed_ip(value: str, literals: set[str]) -> bool:
    if value in literals:
        return True
    return False


def _normalise_port(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        if value.isdigit():
            value = int(value)
        else:
            value = _HTTP_PORTS.get(value.lower())
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 1 <= value <= 65535 else None


_CURRENT_GUARD: contextvars.ContextVar[_NetworkGuard | None] = contextvars.ContextVar(
    "shadow_network_guard", default=None
)
_TRANSPORT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "shadow_network_transport_depth", default=0
)


class _Patcher:
    def __init__(self):
        self._originals: list[tuple[Any, str, Any]] = []

    def set(self, target: Any, name: str, replacement: Any) -> None:
        try:
            original = getattr(target, name)
            setattr(target, name, replacement)
        except (AttributeError, TypeError):
            return
        self._originals.append((target, name, original))

    def restore(self) -> None:
        for target, name, original in reversed(self._originals):
            try:
                setattr(target, name, original)
            except (AttributeError, TypeError):  # pragma: no cover - exotic C API
                pass
        self._originals.clear()


def _invoke(
    guard: _NetworkGuard,
    method: Any,
    url: Any,
    original: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    bounds: _RequestBounds | None = None,
) -> Any:
    """Validate and reserve only the outer transport invocation."""

    depth = _TRANSPORT_DEPTH.get()
    if depth:
        # A high-level ``request`` usually delegates to ``send``.  Validate the
        # nested request too, but reserve exactly one budget attempt.
        guard.check_http(method, url)
        return original(*args, **kwargs)
    token = _TRANSPORT_DEPTH.set(depth + 1)
    try:
        return guard.attempt(method, url, lambda: original(*args, **kwargs), bounds)
    finally:
        _TRANSPORT_DEPTH.reset(token)


async def _invoke_async(
    guard: _NetworkGuard,
    method: Any,
    url: Any,
    original: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    bounds: _RequestBounds | None = None,
) -> Any:
    """Async counterpart to :func:`_invoke`.

    Keeping the depth token installed until the coroutine returns is important:
    an ``httpx.AsyncClient.request`` normally awaits a lower-level ``send``;
    both wrappers must count one attempt and both must remain inside the same
    guard while the actual socket is opened.
    """

    depth = _TRANSPORT_DEPTH.get()
    if depth:
        guard.check_http(method, url)
        return await original(*args, **kwargs)
    token = _TRANSPORT_DEPTH.set(depth + 1)
    try:
        guard.check_http(method, url)
        reservation = guard.reserve_attempt(bounds)
        try:
            result = await original(*args, **kwargs)
        except BaseException:
            guard.settle_attempt(reservation)
            raise
        guard.settle_attempt(reservation)
        return result
    finally:
        _TRANSPORT_DEPTH.reset(token)


def _install_socket_guards(patcher: _Patcher, guard: _NetworkGuard) -> None:
    original_getaddrinfo = socket.getaddrinfo

    @functools.wraps(original_getaddrinfo)
    def guarded_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any):
        guard.check_dns(host, port)
        result = original_getaddrinfo(host, port, *args, **kwargs)
        guard.record_resolution(host, port, result)
        return result

    patcher.set(socket, "getaddrinfo", guarded_getaddrinfo)

    for name in ("gethostbyname", "gethostbyname_ex"):
        original = getattr(socket, name, None)
        if original is None:
            continue

        @functools.wraps(original)
        def guarded_host_lookup(host: Any, _original: Callable[..., Any] = original, **kwargs: Any):
            guard.check_dns(host, None)
            result = _original(host, **kwargs)
            if isinstance(result, str):
                for endpoint in guard.endpoints:
                    if endpoint.host == _safe_host(host):
                        guard._resolved_ips.add((_safe_host(result), endpoint.port))
            elif isinstance(result, tuple) and len(result) >= 3:
                for item in result[2] or ():
                    for endpoint in guard.endpoints:
                        if endpoint.host == _safe_host(host):
                            guard._resolved_ips.add((_safe_host(item), endpoint.port))
            return result

        patcher.set(socket, name, guarded_host_lookup)

    original_gethostbyaddr = getattr(socket, "gethostbyaddr", None)
    if original_gethostbyaddr is not None:

        @functools.wraps(original_gethostbyaddr)
        def guarded_reverse_lookup(host: Any, *args: Any, **kwargs: Any):
            name = _safe_host(host.decode("ascii", "replace") if isinstance(host, bytes) else host)
            known_ips = {ip for ip, _port in guard._resolved_ips}
            if name not in guard._literal_hosts and name not in known_ips:
                _raise_integrity("shadow model egress rejected undeclared reverse DNS access")
            return original_gethostbyaddr(host, *args, **kwargs)

        patcher.set(socket, "gethostbyaddr", guarded_reverse_lookup)

    original_getnameinfo = getattr(socket, "getnameinfo", None)
    if original_getnameinfo is not None:

        @functools.wraps(original_getnameinfo)
        def guarded_nameinfo(address: Any, flags: Any, *args: Any, **kwargs: Any):
            guard.check_connect(address)
            return original_getnameinfo(address, flags, *args, **kwargs)

        patcher.set(socket, "getnameinfo", guarded_nameinfo)

    original_create_connection = socket.create_connection

    @functools.wraps(original_create_connection)
    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any):
        guard.check_connect(address)
        return original_create_connection(address, *args, **kwargs)

    patcher.set(socket, "create_connection", guarded_create_connection)

    original_connect = socket.socket.connect

    @functools.wraps(original_connect)
    def guarded_connect(sock: socket.socket, address: Any, *args: Any, **kwargs: Any):
        guard.check_connect(address, getattr(sock, "family", None))
        return original_connect(sock, address, *args, **kwargs)

    patcher.set(socket.socket, "connect", guarded_connect)

    original_connect_ex = getattr(socket.socket, "connect_ex", None)
    if original_connect_ex is not None:

        @functools.wraps(original_connect_ex)
        def guarded_connect_ex(sock: socket.socket, address: Any, *args: Any, **kwargs: Any):
            guard.check_connect(address, getattr(sock, "family", None))
            return original_connect_ex(sock, address, *args, **kwargs)

        patcher.set(socket.socket, "connect_ex", guarded_connect_ex)

    original_sendto = getattr(socket.socket, "sendto", None)
    if original_sendto is not None:

        @functools.wraps(original_sendto)
        def guarded_sendto(sock: socket.socket, data: Any, *args: Any, **kwargs: Any):
            # sendto(data, flags, address) and sendto(data, address) are both
            # valid Python spellings.
            address = args[-1] if args and isinstance(args[-1], tuple) else kwargs.get("address")
            if address is not None:
                guard.check_connect(address, getattr(sock, "family", None))
            return original_sendto(sock, data, *args, **kwargs)

        patcher.set(socket.socket, "sendto", guarded_sendto)


def _install_requests_guards(patcher: _Patcher, guard: _NetworkGuard) -> None:
    try:
        requests = importlib.import_module("requests")
        sessions = importlib.import_module("requests.sessions")
    except Exception:
        return

    original_request = sessions.Session.request

    @functools.wraps(original_request)
    def guarded_request(session: Any, method: Any, url: Any, *args: Any, **kwargs: Any):
        guard.check_redirects(kwargs.get("allow_redirects"), name="allow_redirects")
        guard.check_proxy(kwargs.get("proxies"), name="proxies")
        bounds = guard.bounds_from_kwargs(kwargs)
        kwargs["allow_redirects"] = False
        kwargs["proxies"] = {}
        return _invoke(
            guard,
            method,
            url,
            original_request,
            (session, method, url, *args),
            kwargs,
            bounds,
        )

    patcher.set(sessions.Session, "request", guarded_request)

    original_send = sessions.Session.send

    @functools.wraps(original_send)
    def guarded_send(session: Any, request: Any, *args: Any, **kwargs: Any):
        method = getattr(request, "method", None)
        url = getattr(request, "url", request)
        bounds = guard.bounds_from_prepared(request)
        return _invoke(
            guard,
            method,
            url,
            original_send,
            (session, request, *args),
            kwargs,
            bounds,
        )

    patcher.set(sessions.Session, "send", guarded_send)

    original_merge = sessions.Session.merge_environment_settings

    @functools.wraps(original_merge)
    def guarded_merge(session: Any, url: Any, proxies: Any, stream: Any, verify: Any, cert: Any):
        guard.check_proxy(proxies, name="proxies")
        result = original_merge(session, url, {}, stream, verify, cert)
        if isinstance(result, dict):
            result["proxies"] = {}
        return result

    patcher.set(sessions.Session, "merge_environment_settings", guarded_merge)

    # Requests' top-level helper can be replaced by an embedding application;
    # patching it makes the boundary explicit even though it normally delegates
    # into Session.request above.
    api = getattr(requests, "api", None)
    original_api_request = getattr(api, "request", None)
    if api is not None and original_api_request is not None:

        @functools.wraps(original_api_request)
        def guarded_api_request(method: Any, url: Any, *args: Any, **kwargs: Any):
            guard.check_redirects(kwargs.get("allow_redirects"), name="allow_redirects")
            guard.check_proxy(kwargs.get("proxies"), name="proxies")
            kwargs["allow_redirects"] = False
            kwargs["proxies"] = {}
            return original_api_request(method, url, *args, **kwargs)

        patcher.set(api, "request", guarded_api_request)


def _install_httpx_guards(patcher: _Patcher, guard: _NetworkGuard) -> None:
    try:
        httpx = importlib.import_module("httpx")
    except Exception:
        return

    def patch_client_class(cls: Any, *, async_client: bool) -> None:
        if cls is None:
            return
        original_init = getattr(cls, "__init__", None)
        if original_init is not None:

            @functools.wraps(original_init)
            def guarded_init(client: Any, *args: Any, **kwargs: Any):
                for key in ("proxy", "proxies"):
                    if kwargs.get(key):
                        guard.check_proxy(kwargs.get(key), name=key)
                kwargs["trust_env"] = False
                return original_init(client, *args, **kwargs)

            patcher.set(cls, "__init__", guarded_init)

        original_request = getattr(cls, "request", None)
        if original_request is not None:

            if async_client:

                @functools.wraps(original_request)
                async def guarded_request(client: Any, method: Any, url: Any, *args: Any, **kwargs: Any):
                    guard.check_redirects(kwargs.get("follow_redirects"), name="follow_redirects")
                    if hasattr(client, "_trust_env"):
                        client._trust_env = False
                    if getattr(client, "follow_redirects", False):
                        _raise_integrity("shadow model egress forbids client redirects")
                    kwargs["follow_redirects"] = False
                    effective_url = _join_base_url(getattr(client, "base_url", None), url)
                    bounds = guard.bounds_from_kwargs(kwargs)
                    return await _invoke_async(
                        guard,
                        method,
                        effective_url,
                        original_request,
                        (client, method, url, *args),
                        kwargs,
                        bounds,
                    )

            else:

                @functools.wraps(original_request)
                def guarded_request(client: Any, method: Any, url: Any, *args: Any, **kwargs: Any):
                    guard.check_redirects(kwargs.get("follow_redirects"), name="follow_redirects")
                    if hasattr(client, "_trust_env"):
                        client._trust_env = False
                    if getattr(client, "follow_redirects", False):
                        _raise_integrity("shadow model egress forbids client redirects")
                    kwargs["follow_redirects"] = False
                    effective_url = _join_base_url(getattr(client, "base_url", None), url)
                    bounds = guard.bounds_from_kwargs(kwargs)
                    return _invoke(
                        guard,
                        method,
                        effective_url,
                        original_request,
                        (client, method, url, *args),
                        kwargs,
                        bounds,
                    )

            patcher.set(cls, "request", guarded_request)

        original_send = getattr(cls, "send", None)
        if original_send is not None:

            if async_client:

                @functools.wraps(original_send)
                async def guarded_send(client: Any, request: Any, *args: Any, **kwargs: Any):
                    follow = kwargs.get("follow_redirects")
                    guard.check_redirects(follow, name="follow_redirects")
                    if getattr(client, "follow_redirects", False):
                        _raise_integrity("shadow model egress forbids client redirects")
                    kwargs["follow_redirects"] = False
                    method = getattr(request, "method", None)
                    url = getattr(request, "url", request)
                    bounds = guard.bounds_from_prepared(request)
                    return await _invoke_async(
                        guard,
                        method,
                        url,
                        original_send,
                        (client, request, *args),
                        kwargs,
                        bounds,
                    )

            else:

                @functools.wraps(original_send)
                def guarded_send(client: Any, request: Any, *args: Any, **kwargs: Any):
                    follow = kwargs.get("follow_redirects")
                    guard.check_redirects(follow, name="follow_redirects")
                    if getattr(client, "follow_redirects", False):
                        _raise_integrity("shadow model egress forbids client redirects")
                    kwargs["follow_redirects"] = False
                    method = getattr(request, "method", None)
                    url = getattr(request, "url", request)
                    bounds = guard.bounds_from_prepared(request)
                    return _invoke(
                        guard,
                        method,
                        url,
                        original_send,
                        (client, request, *args),
                        kwargs,
                        bounds,
                    )

            patcher.set(cls, "send", guarded_send)

    patch_client_class(getattr(httpx, "Client", None), async_client=False)
    patch_client_class(getattr(httpx, "AsyncClient", None), async_client=True)


def _install_aiohttp_guards(patcher: _Patcher, guard: _NetworkGuard) -> None:
    try:
        aiohttp = importlib.import_module("aiohttp")
    except Exception:
        return
    session_cls = getattr(aiohttp, "ClientSession", None)
    if session_cls is None:
        return

    original_init = getattr(session_cls, "__init__", None)
    if original_init is not None:

        @functools.wraps(original_init)
        def guarded_init(session: Any, *args: Any, **kwargs: Any):
            kwargs["trust_env"] = False
            return original_init(session, *args, **kwargs)

        patcher.set(session_cls, "__init__", guarded_init)

    original_request = getattr(session_cls, "_request", None)
    if original_request is None:
        return

    @functools.wraps(original_request)
    async def guarded_request(session: Any, method: Any, url: Any, *args: Any, **kwargs: Any):
        guard.check_redirects(kwargs.get("allow_redirects"), name="allow_redirects")
        if hasattr(session, "_trust_env"):
            session._trust_env = False
        if kwargs.get("params"):
            _raise_integrity("shadow model egress forbids URL query parameters")
        guard.check_proxy(kwargs.get("proxy"), name="proxy")
        guard.check_proxy(kwargs.get("proxy_auth"), name="proxy_auth")
        kwargs["allow_redirects"] = False
        kwargs.pop("proxy", None)
        kwargs.pop("proxy_auth", None)
        bounds = guard.bounds_from_kwargs(kwargs)
        depth = _TRANSPORT_DEPTH.get()
        if depth:
            guard.check_http(method, _join_base_url(getattr(session, "_base_url", None), url))
            return await original_request(session, method, url, *args, **kwargs)
        token = _TRANSPORT_DEPTH.set(depth + 1)
        reservation = None
        try:
            effective_url = _join_base_url(getattr(session, "_base_url", None), url)
            guard.check_http(method, effective_url)
            reservation = guard.reserve_attempt(bounds)
            result = await original_request(session, method, url, *args, **kwargs)
            guard.settle_attempt(reservation)
            return result
        except BaseException:
            guard.settle_attempt(reservation)
            raise
        finally:
            _TRANSPORT_DEPTH.reset(token)

    patcher.set(session_cls, "_request", guarded_request)


def model_egress_only(
    allowed_urls: Iterable[str], budget: RequestBudget | None = None
) -> contextlib.AbstractContextManager[_NetworkGuard]:
    """Allow only bounded HTTPS model POSTs for the duration of a context.

    ``allowed_urls`` must contain exact HTTPS endpoint URLs.  Each endpoint's
    path must end in ``/chat/completions`` or ``/systemone``; URLs with
    credentials, query strings, or fragments are rejected before any patches
    are installed.  A request budget, when supplied, counts attempts
    (including retries) and reserves the UTF-8 request-body byte length plus
    ``max_tokens``/``max_completion_tokens`` before opening the connection.
    Unbounded request bodies fail closed.  Response bodies are never read by
    this guard.  The endpoint and budget declarations are validated when this
    function is called; process-wide patches and proxy removal begin only when
    the returned context is entered.  The context is synchronous but its
    wrappers support async HTTP clients and are task-local through
    ``contextvars``.
    """

    endpoints = _load_endpoints(allowed_urls)
    if budget is not None and not hasattr(budget, "reserve"):
        raise TypeError("budget must provide RequestBudget.reserve/settle")
    # Keep declaration validation outside the generator.  A function decorated
    # with ``contextmanager`` does not execute its body until ``__enter__``;
    # callers should learn about a malformed allowlist at declaration time,
    # before they retain or schedule an invalid guard.  No process-wide state
    # is changed until the returned context manager is entered.
    @contextlib.contextmanager
    def activate() -> Iterator[_NetworkGuard]:
        guard = _NetworkGuard(endpoints, budget)
        patcher = _Patcher()
        env_before = {name: os.environ.get(name) for name in _PROXY_ENV_VARS}
        env_present = set(os.environ).intersection(_PROXY_ENV_VARS)
        for name in _PROXY_ENV_VARS:
            os.environ.pop(name, None)
        current_token = _CURRENT_GUARD.set(guard)
        try:
            _install_socket_guards(patcher, guard)
            _install_requests_guards(patcher, guard)
            _install_httpx_guards(patcher, guard)
            _install_aiohttp_guards(patcher, guard)
            yield guard
        finally:
            patcher.restore()
            _CURRENT_GUARD.reset(current_token)
            for name in _PROXY_ENV_VARS:
                if name in env_present:
                    value = env_before[name]
                    if value is not None:
                        os.environ[name] = value
                else:
                    os.environ.pop(name, None)

    return activate()


__all__ = ["ReplayIntegrityError", "model_egress_only"]
