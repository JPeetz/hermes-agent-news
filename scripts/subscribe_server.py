#!/usr/bin/env python3
"""Tiny HTTP server for newsletter subscription (Buttondown proxy)."""

import json
import logging
import os
import sys
import hmac
import hashlib
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib import request as urllib_req
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("subscribe")

BUTTONDOWN_API_KEY = os.environ.get("BUTTONDOWN_API_KEY", "").strip()
BUTTONDOWN_API = "https://api.buttondown.com/v1/subscribers"
WEBHOOK_HMAC_KEY = os.environ.get("KIE_WEBHOOK_HMAC_KEY", "").strip()
WEB_DIR = os.environ.get("PIPELINE_WEB_DIR", "/app/web")


class SubscribeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(f"{self.client_address[0]} - {fmt % args}")

    def _json(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _verify_webhook(self) -> bool:
        """Verify Kie callback HMAC signature if key is configured."""
        if not WEBHOOK_HMAC_KEY:
            return True  # No key = skip verification
        timestamp = self.headers.get("x-webhook-timestamp", "")
        sig = self.headers.get("x-webhook-signature", "")
        if not timestamp or not sig:
            logger.warning("Callback missing webhook headers")
            return False
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        task_id = body.get("data", {}).get("taskId", "")
        message = f"{task_id}.{timestamp}"
        expected = base64.b64encode(
            hmac.new(WEBHOOK_HMAC_KEY.encode(), message.encode(), hashlib.sha256).digest()
        ).decode()
        return hmac.compare_digest(sig, expected)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length)

        if self.path == "/api/subscribe":
            self._handle_subscribe(body_raw)
        elif self.path == "/api/image-callback":
            self._handle_image_callback(body_raw)
        else:
            self._json(404, {"error": "not found"})


    def _handle_subscribe(self, body_raw: bytes):
        try:
            data = json.loads(body_raw) if body_raw else {}
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

    def _handle_image_callback(self, body_raw: bytes):
        """Handle Kie.ai callback for hero image generation."""
        if WEBHOOK_HMAC_KEY and not self._verify_webhook():
            self._json(401, {"error": "invalid signature"})
            return

        try:
            body = json.loads(body_raw) if body_raw else {}
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return

        code = body.get("code")
        task_id = body.get("data", {}).get("taskId", "")
        info = body.get("data", {}).get("info")

        if code != 200 or not info:
            msg = body.get("msg", "unknown error")
            logger.warning(f"Image callback failed for task {task_id}: {msg}")
            self._json(200, {"ok": False})  # Ack Kie so it doesn't retry
            return

        result_urls = info.get("result_urls", [])
        if not result_urls:
            logger.warning(f"Image callback for {task_id}: no result_urls")
            self._json(200, {"ok": False})
            return

        img_url = result_urls[0]
        logger.info(f"Image callback for {task_id}: {img_url}")

        # Download and save the image
        try:
            img_resp = urllib_req.urlopen(img_url, timeout=180)
            img_data = img_resp.read()

            # Determine the date from the task_id or use today
            from datetime import datetime
            date_str = datetime.now().strftime("%Y-%m-%d")

            hero_dir = Path(WEB_DIR) / "data" / date_str
            hero_dir.mkdir(parents=True, exist_ok=True)
            dest_path = hero_dir / "hero.webp"
            dest_path.write_bytes(img_data)
            logger.info(f"Saved hero image from callback: {dest_path}")

            # Update summary.json hero_image_url
            summary_path = hero_dir / "summary.json"
            if summary_path.exists():
                import json as _json
                summary = _json.loads(summary_path.read_text())
                summary["hero_image_url"] = f"/data/{date_str}/hero.webp"
                summary_path.write_text(_json.dumps(summary, indent=2))
                logger.info(f"Updated summary.json hero_image_url")
        except Exception as e:
            logger.error(f"Failed to save callback image: {e}")

        self._json(200, {"ok": True})


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