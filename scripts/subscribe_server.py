#!/usr/bin/env python3
"""Tiny HTTP server for newsletter subscriptions with local SQLite storage + Buttondown sync."""

import json
import logging
import os
import sys
import sqlite3
import threading
import time
import hmac
import hashlib
import base64
from datetime import datetime
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
DB_PATH = os.environ.get("SUBSCRIBE_DB_PATH", "/app/data/subscribers.db")

# ── SQLite setup ─────────────────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT NOT NULL UNIQUE,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            buttondown_id TEXT,
            synced      INTEGER NOT NULL DEFAULT 0,
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error  TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT NOT NULL,
            action      TEXT NOT NULL,
            status      TEXT NOT NULL,
            details     TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def store_subscriber(email: str) -> bool:
    """Insert or ignore subscriber into SQLite. Returns True if new."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO subscribers (email) VALUES (?)",
            (email,)
        )
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def mark_synced(email: str, buttondown_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE subscribers SET synced=1, buttondown_id=?, retry_count=0, last_error=NULL WHERE email=?",
        (buttondown_id, email)
    )
    conn.commit()
    conn.close()


def mark_pending(email: str, error: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE subscribers SET retry_count=retry_count+1, last_error=? WHERE email=?",
        (error[:500], email)
    )
    conn.commit()
    conn.close()


def log_sync(email: str, action: str, status: str, details: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sync_log (email, action, status, details) VALUES (?, ?, ?, ?)",
        (email, action, status, details[:500])
    )
    conn.commit()
    conn.close()


def get_pending_syncs(limit: int = 50) -> list:
    """Get subscribers not yet synced to Buttondown, oldest first, max 3 retries."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT email, retry_count FROM subscribers WHERE synced=0 AND retry_count < 3 ORDER BY created_at ASC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM subscribers").fetchone()[0]
    synced = conn.execute("SELECT COUNT(*) FROM subscribers WHERE synced=1").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM subscribers WHERE synced=0").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM subscribers WHERE synced=0 AND retry_count >= 3").fetchone()[0]
    today = conn.execute(
        "SELECT COUNT(*) FROM subscribers WHERE date(created_at) = date('now')"
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "synced": synced,
        "pending_sync": pending,
        "failed": failed,
        "subscribed_today": today,
        "db_path": DB_PATH,
    }


# ── Buttondown sync ──────────────────────────────────────────────────────

def sync_to_buttondown(email: str) -> bool:
    """Push a single subscriber to Buttondown. Returns True on success."""
    try:
        req = urllib_req.Request(
            BUTTONDOWN_API,
            data=json.dumps({"email_address": email}).encode(),
            headers={
                "Authorization": f"Token {BUTTONDOWN_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        resp = urllib_req.urlopen(req, timeout=10)
        resp_data = json.loads(resp.read())
        bd_id = resp_data.get("id", "unknown")
        mark_synced(email, bd_id)
        log_sync(email, "buttondown_subscribe", "success", f"buttondown_id={bd_id}")
        return True
    except HTTPError as e:
        err_body = e.read().decode()[:300]
        log_sync(email, "buttondown_subscribe", "error", f"HTTP {e.code}: {err_body}")
        mark_pending(email, f"HTTP {e.code}: {err_body}")
        return False
    except Exception as e:
        log_sync(email, "buttondown_subscribe", "error", str(e)[:300])
        mark_pending(email, str(e)[:300])
        return False


def retry_pending_syncs():
    """Background task: retry unsynced subscribers every 5 minutes."""
    while True:
        if not BUTTONDOWN_API_KEY:
            time.sleep(300)
            continue
        try:
            pending = get_pending_syncs()
            if pending:
                logger.info(f"Retry sync: {len(pending)} pending subscribers")
                for email, retries in pending:
                    logger.info(f"Syncing {email} (retry {retries+1})")
                    sync_to_buttondown(email)
                    time.sleep(1)  # Rate limit: 1 req/s
        except Exception as e:
            logger.error(f"Retry sync error: {e}")
        time.sleep(300)  # 5 minutes between retry batches


# ── HTTP handler ──────────────────────────────────────────────────────────

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
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _verify_webhook(self) -> bool:
        if not WEBHOOK_HMAC_KEY:
            return True
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

    def do_GET(self):
        if self.path == "/api/stats":
            self._json(200, get_stats())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length)

        if self.path == "/subscribe":
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

        email = (data.get("email") or "").strip().lower()
        if not email or "@" not in email or "." not in email.split("@")[-1]:
            self._json(400, {"error": "valid email required"})
            return

        # Store locally first — always
        is_new = store_subscriber(email)
        if not is_new:
            logger.info(f"Duplicate subscription: {email}")
            self._json(200, {"ok": True, "message": "Already subscribed!"})
            return

        logger.info(f"New subscriber stored: {email}")

        # Try Buttondown sync immediately
        if BUTTONDOWN_API_KEY:
            synced = sync_to_buttondown(email)
            if synced:
                logger.info(f"Subscribed + synced to Buttondown: {email}")
                self._json(200, {"ok": True, "message": "Subscribed!"})
            else:
                logger.warning(f"Stored locally, Buttondown sync pending: {email}")
                self._json(200, {
                    "ok": True,
                    "message": "Subscribed! Sync pending.",
                    "note": "Saved locally, will retry Buttondown sync automatically."
                })
        else:
            logger.info(f"Stored locally (no Buttondown key): {email}")
            self._json(200, {
                "ok": True,
                "message": "Subscribed!",
                "note": "No email provider configured."
            })

    def _handle_image_callback(self, body_raw: bytes):
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
            self._json(200, {"ok": False})
            return

        result_urls = info.get("result_urls", [])
        if not result_urls:
            logger.warning(f"Image callback for {task_id}: no result_urls")
            self._json(200, {"ok": False})
            return

        img_url = result_urls[0]
        logger.info(f"Image callback for {task_id}: {img_url}")

        try:
            img_resp = urllib_req.urlopen(img_url, timeout=180)
            img_data = img_resp.read()

            date_str = datetime.now().strftime("%Y-%m-%d")
            hero_dir = Path(WEB_DIR) / "data" / date_str
            hero_dir.mkdir(parents=True, exist_ok=True)
            dest_path = hero_dir / "hero.webp"
            dest_path.write_bytes(img_data)
            logger.info(f"Saved hero image from callback: {dest_path}")

            summary_path = hero_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                summary["hero_image_url"] = f"/data/{date_str}/hero.webp"
                summary_path.write_text(json.dumps(summary, indent=2))
                logger.info(f"Updated summary.json hero_image_url")
        except Exception as e:
            logger.error(f"Failed to save callback image: {e}")

        self._json(200, {"ok": True})


def main():
    init_db()
    port = int(os.environ.get("SUBSCRIBE_PORT", "8080"))

    # Start background retry thread
    retry_thread = threading.Thread(target=retry_pending_syncs, daemon=True)
    retry_thread.start()
    logger.info("Background sync retry thread started (5 min interval)")

    # Retry any existing pending syncs on startup
    pending = get_pending_syncs()
    if pending:
        logger.info(f"Retrying {len(pending)} pending subscribers from previous run")
        for email, retries in pending:
            if BUTTONDOWN_API_KEY:
                sync_to_buttondown(email)
                time.sleep(0.5)

    server = HTTPServer(("127.0.0.1", port), SubscribeHandler)
    logger.info(f"Subscribe server listening on 127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()