"""Focused tests for the shadow model-only network boundary.

These tests use mocked HTTP transports and never contact a provider.  The
guard is deliberately tested at both the client and socket layers because an
HTTP-client-only patch would leave a raw socket escape hatch.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shadow.budget import BudgetExceeded, BudgetLimits, RequestBudget  # noqa: E402
try:  # The capture worker owns the canonical replay error class.
    from shadow.replay_context import ReplayIntegrityError  # noqa: E402
except ImportError:  # pragma: no cover - before the capture worker lands
    from shadow.network import ReplayIntegrityError  # noqa: E402
from shadow.network import model_egress_only  # noqa: E402


MODEL_URL = "https://model.example.test/v1/chat/completions"
SYSTEMONE_URL = "https://api.typesafe.test/v1/systemone"


class DeclarationTests(unittest.TestCase):
    def test_allowlist_is_https_model_endpoint_without_url_metadata(self):
        for url in (
            "http://model.example.test/v1/chat/completions",
            "https://user:pass@model.example.test/v1/chat/completions",
            "https://model.example.test/v1/chat/completions?token=secret",
            "https://model.example.test/v1/chat/completions#fragment",
            "https://model.example.test/v1/catalog",
            "https://model.example.test/v1/systemone/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                model_egress_only([url])

    def test_type_safe_systemone_endpoint_is_accepted(self):
        with model_egress_only([SYSTEMONE_URL]) as guard:
            self.assertEqual(len(guard.endpoints), 1)


class SocketBoundaryTests(unittest.TestCase):
    def test_undeclared_dns_and_socket_destinations_fail_before_io(self):
        with model_egress_only([MODEL_URL]):
            with self.assertRaises(ReplayIntegrityError):
                socket.getaddrinfo("source.example.test", 443)
            with self.assertRaises(ReplayIntegrityError):
                socket.create_connection(("source.example.test", 443), timeout=0.01)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                with self.assertRaises(ReplayIntegrityError):
                    sock.connect_ex(("198.51.100.22", 443))

    def test_proxy_environment_is_removed_only_inside_context(self):
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.example.test:8080"}, clear=False):
            with model_egress_only([MODEL_URL]):
                self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.example.test:8080")


class RequestsBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("requests is not installed")

    def test_get_and_other_host_are_rejected_without_transport(self):
        import requests

        with model_egress_only([MODEL_URL]):
            with self.assertRaises(ReplayIntegrityError):
                requests.get(MODEL_URL)
            with self.assertRaises(ReplayIntegrityError):
                requests.post("https://source.example.test/v1/chat/completions")
            with self.assertRaises(ReplayIntegrityError):
                requests.post(MODEL_URL, allow_redirects=True)

    def test_declared_post_uses_no_redirects_and_preserves_body(self):
        import requests

        class StubSession(requests.Session):
            def send(self, request, **kwargs):
                response = requests.Response()
                response.status_code = 200
                response.url = request.url
                response.request = request
                response._content = b"stream-like body"
                response._content_consumed = False
                return response

        with model_egress_only([MODEL_URL]):
            with StubSession() as session:
                response = session.post(MODEL_URL)
                self.assertEqual(response.content, b"stream-like body")


class HttpxBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("httpx is not installed")

    def test_sync_mock_transport_accepts_exact_post_only(self):
        import httpx

        seen = []

        def handle(request):
            seen.append(request)
            return httpx.Response(200, request=request, content=b"response")

        with model_egress_only([MODEL_URL]):
            with httpx.Client(transport=httpx.MockTransport(handle)) as client:
                response = client.post(MODEL_URL, json={"ok": True})
                self.assertEqual(response.content, b"response")
                self.assertEqual(len(seen), 1)
                with self.assertRaises(ReplayIntegrityError):
                    client.get(MODEL_URL)
                with self.assertRaises(ReplayIntegrityError):
                    client.post(MODEL_URL, follow_redirects=True)

    def test_async_mock_transport_accepts_relative_path_after_base_url(self):
        import httpx

        async def run():
            def handle(request):
                return httpx.Response(200, request=request, content=b"async response")

            with model_egress_only([MODEL_URL]):
                async with httpx.AsyncClient(
                    base_url="https://model.example.test",
                    transport=httpx.MockTransport(handle),
                ) as client:
                    response = await client.post("/v1/chat/completions")
                    self.assertEqual(response.content, b"async response")
                    with self.assertRaises(ReplayIntegrityError):
                        await client.post("/v1/chat/completions", follow_redirects=True)

        asyncio.run(run())


class BudgetBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("httpx is not installed")

    def test_budget_counts_attempts_without_reading_response_content(self):
        import httpx

        # The network guard reserves the buffered request-body byte length as
        # its conservative input bound.  Keep the request-count ceiling tight
        # while allowing this small JSON body to reach the mocked transport.
        budget = RequestBudget(BudgetLimits(max_requests=1, max_input_tokens=64, max_output_tokens=1))
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, request=request, content=b"body")
        )
        with model_egress_only([MODEL_URL], budget=budget):
            with httpx.Client(transport=transport) as client:
                response = client.post(MODEL_URL, json={"max_tokens": 1, "messages": []})
                self.assertEqual(response.content, b"body")
                with self.assertRaises(BudgetExceeded):
                    client.post(MODEL_URL, json={"max_tokens": 1, "messages": []})
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["unknown_usage_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
