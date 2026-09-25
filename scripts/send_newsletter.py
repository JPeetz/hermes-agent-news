#!/usr/bin/env python3
"""Send daily news digest via Buttondown as an Agent N-branded HTML email."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("send_newsletter")


def build_newsletter_html(summary: dict, report_date: str, base_url: str = "https://hermesnews.xyz") -> str:
    """Render the HTML email with Agent N branding."""
    exec_summary = summary.get("executive_summary_html") or summary.get("executive_summary", "")
    topics = summary.get("top_topics", [])[:6]
    categories = summary.get("categories", {})
    hero_url = summary.get("hero_image_url")
    total_items = summary.get("total_items_analyzed", 0)

    topic_rows = ""
    for t in topics:
        title = t.get("title", "Topic")
        importance = t.get("importance", "")
        cat = t.get("category", "general")
        # Map category to color
        color_map = {"news": "#0066FF", "research": "#22C55E", "social": "#F97316", "reddit": "#EF4444"}
        dot_color = color_map.get(cat, "#888")
        topic_rows += f"""
          <tr>
            <td style="padding:6px 0;border-bottom:1px solid #2A2A2A;">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot_color};margin-right:8px;"></span>
              <span style="color:#E0E0E0;font-size:14px;">{title}</span>
              {f'<span style="color:#888;font-size:12px;margin-left:8px;">({importance})</span>' if importance else ''}
            </td>
          </tr>"""

    cat_counts = ", ".join(f"{k}: {v.get('count', 0)}" for k, v in categories.items() if v.get("count", 0) > 0)

    # Build hero HTML if image exists
    hero_html = ""
    if hero_url:
        full_hero = f"{base_url}{hero_url}" if hero_url.startswith("/") else hero_url
        hero_html = f"""
          <tr>
            <td style="padding:0 0 20px 0;">
              <a href="{base_url}"><img src="{full_hero}" alt="Agent N's Hermes News" style="width:100%;max-width:600px;height:auto;border-radius:8px;display:block;" /></a>
            </td>
          </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body style="margin:0;padding:0;background-color:#0A0A0A;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0A0A0A;">
    <tr>
      <td align="center" style="padding:20px 10px;">
        <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">
          <!-- Header -->
          <tr>
            <td style="text-align:center;padding:30px 20px 10px 20px;">
              <span style="color:#FFD700;font-size:28px;font-weight:bold;letter-spacing:1px;">Agent N's Hermes News</span>
              <p style="color:#00FFFF;font-size:14px;margin:4px 0 0 0;">Powered by GLM-5.3-Flash</p>
            </td>
          </tr>
          <tr><td style="height:2px;background:linear-gradient(90deg,#FFD700,#00FFFF,#FFD700);margin:0 20px;"></td></tr>
          <!-- Date header -->
          <tr>
            <td style="padding:20px 20px 10px 20px;text-align:center;">
              <p style="color:#E0E0E0;font-size:16px;margin:0;">{report_date}</p>
              <p style="color:#888;font-size:12px;margin:4px 0 0 0;">{total_items} items analyzed | <a href="{base_url}" style="color:#00FFFF;">View online</a></p>
            </td>
          </tr>
          {hero_html}
          <!-- Executive Summary -->
          <tr>
            <td style="padding:10px 20px;">
              <h2 style="color:#FFD700;font-size:18px;margin:0 0 10px 0;">Executive Summary</h2>
              <div style="color:#CCCCCC;font-size:14px;line-height:1.6;">
                {exec_summary[:800]}
              </div>
            </td>
          </tr>
          <!-- Topics -->
          <tr>
            <td style="padding:10px 20px;">
              <h2 style="color:#FFD700;font-size:18px;margin:0 0 10px 0;">Top Stories</h2>
              <table width="100%" cellpadding="0" cellspacing="0">
                {topic_rows}
              </table>
            </td>
          </tr>
          <!-- Category counts -->
          <tr>
            <td style="padding:10px 20px;">
              <div style="color:#888;font-size:12px;border-top:1px solid #2A2A2A;padding-top:10px;">
                Coverage: {cat_counts}
              </div>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding:20px;text-align:center;border-top:1px solid #2A2A2A;margin-top:20px;">
              <p style="color:#666;font-size:11px;margin:0;">
                <a href="{base_url}" style="color:#00FFFF;">Agent N's Hermes News</a> &mdash;
                Daily Hermes Agent &amp; Hermes Desktop news<br/>
                Hero images by <a href="https://kie.ai?ref=7c7e62a37e5bbed684c789bbd7d6f0dd" style="color:#00FFFF;">kie.ai</a>
              </p>
              <p style="color:#666;font-size:10px;margin:10px 0 0 0;">
                You received this because you subscribed. 
                <a href="%unsubscribe_url%" style="color:#888;">Unsubscribe</a>
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    return html


def send_newsletter(api_key: str, html_body: str, report_date: str):
    """Post newsletter to Buttondown as a sent email."""
    url = "https://api.buttondown.com/v1/emails"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "subject": f"Agent N's Hermes News — {report_date}",
        "body": html_body,
        "status": "sent",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    data = resp.json()
    if resp.status_code in (200, 201):
        logger.info(f"Buttondown email sent: id={data.get('id')} status={data.get('status')}")
        return True
    else:
        logger.error(f"Buttondown API error {resp.status_code}: {data}")
        return False


def main():
    api_key = os.environ.get("BUTTONDOWN_API_KEY", "").strip()
    if not api_key:
        logger.warning("BUTTONDOWN_API_KEY not set; skipping newsletter")
        return

    data_dir = os.environ.get("DATA_DIR", "/app/web/data")
    date = os.environ.get("TARGET_DATE", datetime.now().strftime("%Y-%m-%d"))
    base_url = os.environ.get("PIPELINE_BASE_URL", "https://hermesnews.xyz")

    summary_path = Path(data_dir) / date / "summary.json"
    if not summary_path.exists():
        logger.warning(f"Summary not found: {summary_path}; skipping")
        return

    summary = json.loads(summary_path.read_text())
    html = build_newsletter_html(summary, date, base_url)
    send_newsletter(api_key, html, date)


if __name__ == "__main__":
    main()