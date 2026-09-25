#!/usr/bin/env python3
"""Tiny HTTP server for newsletter subscription (Buttondown proxy)."""

import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib import request as urllib_req
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("subscribe")

BUTTONDOWN_API_KEY = os.environ.get("BUTTONDOWN_API_KEY", "").strip()
BUTTONDOWN_API = "https://api.buttondown.com/v1/subscribers"


class SubscribeHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path != "/subscribe":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        email = (data.get("email") or "").strip()
        if not email or "@" not in email:
            self._json(400, {"error": "valid email required"})
            return

        if not BUTTONDOWN_API_KEY:
            logger.error("BUTTONDOWN_API_KEY not set")
            self._json(500, {"error": "server config error"})
            return

        try:
            req = urllib_req.Request(
                BUTTONDOWN_API,
                data=json.dumps({"email": email}).encode(),
                headers={
                    "Authorization": f"Token {BUTTONDOWN_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp = urllib_req.urlopen(req, timeout=10)
            resp_data = json.loads(resp.read())
            logger.info(f"Subscribed: {email} (id={resp_data.get('id')})")
            self._json(200, {"ok": True, "message": "Subscribed!"})
        except HTTPError as e:
            err_body = e.read().decode()
            logger.warning(f"Buttondown API error {e.code} for {email}: {err_body}")
            self._json(e.code, {"error": "subscription failed"})

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        logger.info(f"{self.client_address[0]} - {fmt % args}")


def main():
    port = int(os.environ.get("SUBSCRIBE_PORT", "8080"))
    server = HTTPServer(("127.0.0.1", port), SubscribeHandler)
    logger.info(f"Subscribe server listening on 127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()