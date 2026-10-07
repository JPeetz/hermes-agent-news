#!/usr/bin/env python3
"""Send daily news digest via Buttondown as an Agent N-branded HTML email."""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # cron children don't inherit docker-compose env; read /app/.env
except ImportError:
    pass

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("send_newsletter")


def build_newsletter_html(summary: dict, report_date: str, base_url: str = "https://hermesnews.xyz") -> str:
    """Render the HTML email with Agent N branding."""
    exec_summary = summary.get("executive_summary_html") or summary.get("executive_summary", "")

    # Make internal relative links absolute so they work in email. The website's
    # exec summary carries links like href="/?date=...&category=..." which render
    # fine on the site but are dead in an email client (no origin to resolve against).
    # We rewrite every href="/... to href="<base_url>/... before sending. This is safe:
    # external links are already absolute (http/https) and untouched.
    import re as _re
    exec_summary = _re.sub(
        r'href="/(?!/)', f'href="{base_url}/', exec_summary
    ) if exec_summary else ""

    topics = summary.get("top_topics", [])[:6]
    categories = summary.get("categories", {})
    hero_url = summary.get("hero_image_url")
    total_items = summary.get("total_items_analyzed", 0)

    topic_rows = ""
    for t in topics:
        # top_topics objects carry `name` (+ rich `description`), NOT `title`. Reading
        # `title` made every row render the literal fallback "Topic" + a bare importance
        # number with no content. Use name for the label and a short clean description
        # snippet so each topic row actually shows what the story is about.
        title = t.get("name") or t.get("title") or "Top Story"
        importance = t.get("importance", "")
        cat = t.get("category", "general")
        # Category color map
        color_map = {"news": "#0066FF", "research": "#22C55E", "social": "#F97316", "reddit": "#EF4444"}
        dot_color = color_map.get(cat, "#888")
        # Clean description for email: strip markdown links/bold to plain text, cap length.
        import re as _re
        desc_raw = t.get("description", "") or ""
        desc_clean = _re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", desc_raw)
        desc_clean = desc_clean.replace("**", "").replace("*", "")
        desc_clean = _re.sub(r"\s+", " ", desc_clean).strip()
        if len(desc_clean) > 180:
            desc_clean = desc_clean[:177].rstrip() + "..."
        importance_badge = f' <span style="color:#999999;font-size:12px;">({importance})</span>' if importance else ""
        desc_html = f'<div style="color:#666666;font-size:13px;line-height:1.5;margin-top:2px;">{desc_clean}</div>' if desc_clean else ""
        topic_rows += f"""
          <tr>
            <td style="padding:8px 0;border-bottom:1px solid #E0E0E0;">
              <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot_color};margin-right:8px;"></span>
              <span style="color:#333333;font-size:14px;font-weight:600;">{title}</span>{importance_badge}
              {desc_html}
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
<body style="margin:0;padding:0;background-color:#F4F4F4;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#F4F4F4;">
    <tr>
      <td align="center" style="padding:30px 10px;">
        <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#FFFFFF;border-radius:12px;">
          <!-- Header -->
          <tr>
            <td style="text-align:center;padding:32px 24px 12px 24px;">
              <span style="color:#D4A843;font-size:26px;font-weight:700;letter-spacing:0.5px;">Agent N's Hermes News</span>
              <p style="color:#0088CC;font-size:13px;margin:4px 0 0 0;font-weight:500;">Powered by GLM-5.3-Flash</p>
            </td>
          </tr>
          <tr><td style="height:2px;background:linear-gradient(90deg,#D4A843,#0088CC,#D4A843);margin:0 24px;"></td></tr>
          <!-- Date header -->
          <tr>
            <td style="padding:20px 24px 10px 24px;text-align:center;">
              <p style="color:#555555;font-size:15px;margin:0;font-weight:500;">{report_date}</p>
              <p style="color:#888888;font-size:12px;margin:4px 0 0 0;">{total_items} items analyzed | <a href="{base_url}" style="color:#0088CC;text-decoration:underline;">View online</a></p>
            </td>
          </tr>
          {hero_html}
          <!-- Executive Summary -->
          <tr>
            <td style="padding:8px 24px 16px 24px;">
              <h2 style="color:#D4A843;font-size:17px;margin:0 0 10px 0;font-weight:700;">Executive Summary</h2>
              <div style="color:#333333;font-size:14px;line-height:1.7;">
                {exec_summary}
              </div>
            </td>
          </tr>
          <!-- Topics -->
          <tr>
            <td style="padding:8px 24px 16px 24px;">
              <h2 style="color:#D4A843;font-size:17px;margin:0 0 10px 0;font-weight:700;">Top Stories</h2>
              <table width="100%" cellpadding="0" cellspacing="0">
                {topic_rows}
              </table>
            </td>
          </tr>
          <!-- Category counts -->
          <tr>
            <td style="padding:8px 24px;">
              <div style="color:#888888;font-size:12px;border-top:1px solid #E0E0E0;padding-top:10px;">
                Coverage: {cat_counts}
              </div>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding:20px 24px;text-align:center;border-top:1px solid #E0E0E0;margin-top:16px;">
              <p style="color:#888888;font-size:11px;margin:0;">
                <a href="{base_url}" style="color:#0088CC;text-decoration:underline;">Agent N's Hermes News</a> &mdash;
                Daily Hermes Agent &amp; Hermes Desktop news<br/>
                Hero images by <a href="https://kie.ai?ref=7c7e62a37e5bbed684c789bbd7d6f0dd" style="color:#0088CC;text-decoration:underline;">kie.ai</a>
              </p>
              <p style="color:#999999;font-size:10px;margin:10px 0 0 0;">
                You received this because you subscribed. 
                <a href="%unsubscribe_url%" style="color:#999999;">Unsubscribe</a>
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
        "X-Buttondown-Live-Dangerously": "true",
    }
    payload = {
        "subject": f"Agent N's Hermes News — {report_date}",
        "body": html_body,
        "status": "about_to_send",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    data = resp.json()
    if resp.status_code in (200, 201):
        logger.info(f"Buttondown email sent: id={data.get('id')} status={data.get('status')}")
        return True
    elif resp.status_code == 400 and data.get("code") == "email_duplicate":
        # Same subject/date already exists on Buttondown (e.g. a re-run of a
        # date that already sent). This is not a failure — the email is out.
        logger.info(f"Buttondown duplicate for {report_date}; email already exists — nothing to send")
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
    date = os.environ.get("TARGET_DATE", "").strip() or datetime.now().strftime("%Y-%m-%d")
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