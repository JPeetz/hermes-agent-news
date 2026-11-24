#!/usr/bin/env python3
"""
llms Generator for AI News Aggregator

Generates llms.txt + llms-full.txt from index.json and the search corpus so AI
assistants (ChatGPT, Perplexity, Claude, Gemini) can discover every edition and
read machine-readable content — the GEO half of the site's discoverability.

Follows the llms.txt convention (https://llmstxt.org):
  - llms.txt      : one line per key resource, grouped by ## sections
  - llms-full.txt : the full expanded content for the latest edition

Written for GEO: Google Search Console barely knew the site; llms.txt gives
AI assistants a complete, current map of every published edition + the raw
JSON endpoints they can read directly.

Outputs:
  - web/data/llms.txt      (aliased to /llms.txt in nginx, same as sitemap.xml)
  - web/data/llms-full.txt (aliased to /llms-full.txt)
"""

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "https://hermesnews.xyz"


class LlmsGenerator:
    def __init__(self, web_dir, base_url=BASE_URL):
        # web_dir is the web ROOT; data lives in /data (the mounted volume).
        self.web_dir = Path(web_dir)
        self.data_dir = self.web_dir / "data"
        self.base_url = base_url.rstrip("/")

    def _manifest(self) -> list[dict]:
        idx = self.data_dir / "index.json"
        if not idx.exists():
            return []
        try:
            d = json.loads(idx.read_text(encoding="utf-8"))
            return d.get("dates", [])
        except Exception:
            return []

    def _corpus(self) -> list[dict]:
        p = self.data_dir / "search-corpus.json"
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    def generate(self) -> tuple[str, str]:
        dates = sorted(self._manifest(), key=lambda x: x.get("date", ""), reverse=True)
        latest = dates[0]["date"] if dates else None
        corpus = self._corpus()

        lines = []
        lines.append(f"# Agent N's Hermes News")
        lines.append("")
        lines.append(
            "> Daily curated news about the Hermes Agent / Hermes Desktop ecosystem — "
            "multi-agent pipeline that collects, analyzes, and summarizes Hermes Agent "
            "releases, community updates, research, and ecosystem developments. "
            "Machine-readable editions below; every article links back to its source."
        )
        lines.append("")
        lines.append("## Navigation")
        lines.append("")
        lines.append(f"- [Home — Latest Edition]({self.base_url}/) — Today's curated Hermes Agent news")
        lines.append(f"- [About]({self.base_url}/about) — How the pipeline works and Agent N's mission")
        lines.append(f"- [Archive]({self.base_url}/archive) — Browse all past editions")
        lines.append(f"- [RSS Feeds]({self.base_url}/feeds) — Atom RSS feeds for every category")
        lines.append("")

        lines.append("## Current Edition")
        lines.append("")
        if latest:
            lines.append(f"- [{latest} — Latest Edition]({self.base_url}/?date={latest})")
            lines.append(f"- [Summary JSON]({self.base_url}/data/{latest}/summary.json)")
            lines.append(f"- [News JSON]({self.base_url}/data/{latest}/news.json)")
            lines.append(f"- [Research JSON]({self.base_url}/data/{latest}/research.json)")
            lines.append(f"- [Social JSON]({self.base_url}/data/{latest}/social.json)")
            lines.append(f"- [Reddit JSON]({self.base_url}/data/{latest}/reddit.json)")
        lines.append("")

        lines.append("## All Editions")
        lines.append("")
        lines.append("Chronological archive (newest first). Each edition has its own")
        lines.append("summary.json / news.json / research.json / social.json / reddit.json:")
        lines.append("")
        for e in dates:
            d = e.get("date", "")
            if d:
                lines.append(f"- [{d}]({self.base_url}/?date={d})")
        lines.append("")

        lines.append("## Data Sources (Machine-Readable)")
        lines.append("")
        lines.append("- [/data/index.json]({}/data/index.json) — Date manifest (all editions)".format(self.base_url))
        lines.append("- [/data/search-corpus.json]({}/data/search-corpus.json) — Full 30-day search corpus (all items)".format(self.base_url))
        lines.append("- [/data/quick-answer.html]({}/data/quick-answer.html) — Current day's distilled answer".format(self.base_url))
        lines.append("")

        lines.append("## RSS Feeds")
        lines.append("")
        for feed in ("main", "news", "research-25", "social-25", "reddit-25", "reddit-100", "research-100"):
            lines.append(f"- [{feed}.xml]({self.base_url}/data/feeds/{feed}.xml)")
        lines.append("")

        lines.append("## Topic Coverage")
        lines.append("")
        lines.append(
            "Agent N's Hermes News tracks: Hermes Agent releases, Hermes Desktop updates, "
            "community tools (Awesome Hermes Agent list, Hermes Bible), the wider AI agent "
            "ecosystem (MCP, tool-use standards, agent frameworks), major AI model releases "
            "(OpenAI, Anthropic, Google, Meta, DeepSeek, Mistral, Kimi, MiniMax), and daily "
            "notable papers from HuggingFace/arXiv relevant to AI agents and LLMs."
        )
        lines.append("")
        lines.append("## Branding")
        lines.append("")
        lines.append("- Color palette: Gold (#FFD700), Cyan (#00FFFF), deep black backgrounds")
        lines.append("- Mascot: Agent N — a young woman with short dark hair, brown eyes, black over-ear headphones with 'N' logo")
        lines.append("- Powered by [JP Technologies](https://jptechnologies.vercel.app/)")
        lines.append("")
        lines.append("## llms-full.txt")
        lines.append("")
        lines.append(f"[Full latest-edition content]({self.base_url}/llms-full.txt) — the latest edition's items in plain text.")

        llms_txt = "\n".join(lines)

        # llms-full.txt: latest edition items + recent corpus, plain text.
        full_lines = [f"# Agent N's Hermes News — Full Content", ""]
        if latest:
            full_lines.append(f"## Latest Edition ({latest})")
            full_lines.append("")
            for fname, label in (
                ("summary.json", "Executive Summary"),
                ("news.json", "News"),
                ("research.json", "Research"),
                ("social.json", "Social"),
                ("reddit.json", "Reddit"),
            ):
                p = self.data_dir / latest / fname
                if not p.exists():
                    continue
                full_lines.append(f"### {label}")
                full_lines.append("")
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    items = d if isinstance(d, list) else d.get("items", [])
                    if fname == "summary.json":
                        full_lines.append(str(d)[:2000])
                    else:
                        for it in items[:50]:
                            title = it.get("title", "")
                            url = it.get("url", "")
                            summary = (it.get("summary") or "")[:300]
                            full_lines.append(f"- {title}")
                            if url:
                                full_lines.append(f"  {url}")
                            if summary:
                                full_lines.append(f"  {summary}")
                except Exception as e:
                    full_lines.append(f"(could not read {fname}: {e})")
                full_lines.append("")
        if corpus:
            full_lines.append(f"## Recent Corpus ({len(corpus)} items)")
            full_lines.append("")
            for it in corpus[:100]:
                title = it.get("title", "")
                url = it.get("url", "")
                date = it.get("date", "")
                summary = (it.get("summary") or "")[:200]
                full_lines.append(f"- [{date}] {title}")
                if url:
                    full_lines.append(f"  {url}")
                if summary:
                    full_lines.append(f"  {summary}")
        full_lines.append("")

        llms_full = "\n".join(full_lines)

        (self.data_dir / "llms.txt").write_text(llms_txt, encoding="utf-8")
        (self.data_dir / "llms-full.txt").write_text(llms_full, encoding="utf-8")
        logger.info("llms: wrote llms.txt (%d lines) + llms-full.txt (%d lines)", len(lines), len(full_lines))
        return llms_txt, llms_full


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate llms.txt + llms-full.txt from index.json")
    p.add_argument("--web-dir", default="web", help="Web root (data lives in its /data subdir)")
    p.add_argument("--base-url", default=BASE_URL)
    args = p.parse_args()
    gen = LlmsGenerator(args.web_dir, args.base_url)
    llms, full = gen.generate()
    print(f"OK -> {args.web_dir}/data/llms.txt + llms-full.txt")
