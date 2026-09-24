#!/usr/bin/env python3
"""
Hugging Face Daily Papers Filter — reduces 50 arxiv papers/day to the ~8 most notable for AI news.

Run: python3 scripts/hf_daily_filter.py

Output: prints Atom RSS XML to stdout (pipe to a file, or consume directly).
Set HF_DAILY_OUTPUT to a file path to write RSS to disk.

Inclusion criteria (any match = keep):
  1. Author/institution mentions major AI orgs (OpenAI, Google DeepMind, Meta, etc.)
  2. Title mentions core agent/LLM topics: agent, framework, reasoning, tool use, etc.
  3. Title suggests a major release, benchmark, or breakthrough

Usage:
  python3 scripts/hf_daily_filter.py --max 8
  python3 scripts/hf_daily_filter.py > /tmp/hf_daily_filtered.xml
"""

import json
import sys
import os
import argparse
import urllib.request
from datetime import datetime, timezone

# Major AI organizations whose papers are automatically notable
MAJOR_ORGS = [
    "openai", "anthropic", "google deepmind", "meta", "microsoft",
    "deepseek", "mistral", "moonshot", "minimax", "alibaba",
    "tencent", "baidu", "bytedance", "stability ai", "midjourney",
    "cohere", "ai2", "eleuthera", "together", "reka", "01.ai",
    "google research", "ibm research", "apple",
]

# High-value topic keywords — papers mentioning these in their title are notable
HIGH_VALUE_KEYWORDS = [
    "agent", "framework", "llm", "large language model", "foundation model",
    "reasoning", "tool use", "tool-use", "function calling",
    "code generation", "open source", "model release", "benchmark",
    "multimodal", "vision-language", "VL model", "video generation",
    "state-of-the-art", "SOTA", "frontier", "breakthrough",
    "alignment", "safety", "evaluation", "RLHF", "DPO", "GRPO",
    "fine-tuning", "training recipe", "scaling law", "architecture",
    "mixture of experts", "MoE", "transformer", "attention",
    "retrieval", "RAG", "knowledge graph", "embedding",
    "reinforcement learning", "RL", "self-play",
    "instruction following", "steerability", "prompt",
    "test-time compute", "inference", "distillation",
    "context window", "long context", "memory",
]

LOW_VALUE_KEYWORDS = [
    "survey", "review", "comprehensive study", "preliminary",
    "toward", "towards", "small", "low-resource", "low resource",
    "pedagogical", "tutorial", "rebuttal", "discussion",
    "detection", "adversarial", "attack", "poison",
]


def score_paper(paper):
    """Score a paper from 0-100 on how notable it is for AI news."""
    title = (paper.get("title", "") or "").lower()
    summary = (paper.get("summary", "") or "").lower()
    text = f"{title} {summary}"

    score = 0
    reasons = []

    # +30 if from major org
    for org in MAJOR_ORGS:
        if org in text:
            score += 30
            reasons.append(f"org:{org}")
            break

    # +10 per high-value keyword hit (max +50)
    kw_hits = 0
    for kw in HIGH_VALUE_KEYWORDS:
        if kw in title:
            score += 15
            kw_hits += 1
            reasons.append(f"kw-title:{kw}")
        elif kw in summary[:300]:
            score += 5
            kw_hits += 1

    # Penalize low-value keywords (-20 per hit in title)
    for kw in LOW_VALUE_KEYWORDS:
        if kw in title:
            score -= 20
            reasons.append(f"penalty:{kw}")

    # Penalize very narrow/specialized topics
    narrow_terms = ["protein", "molecule", "chemistry", "drug", "medical imaging",
                    "spectroscopy", "crystal", "quantum", "material", "biology"]
    for nt in narrow_terms:
        if nt in title and "agent" not in title and "llm" not in title:
            score -= 15

    return max(0, score), reasons


def fetch_daily_papers():
    """Fetch the daily papers list from Hugging Face."""
    url = "https://huggingface.co/api/daily_papers"
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Agent-News/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def generate_rss(papers, title_prefix="HF Daily Notable"):
    """Generate Atom RSS XML from scored papers."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    entries = []
    for paper, score, reasons in papers:
        pid = paper["paper"]["id"] if "paper" in paper else paper.get("id", "unknown")
        title = paper.get("title", paper.get("paper", {}).get("title", "Untitled"))
        summary = paper.get("summary", paper.get("paper", {}).get("summary", ""))[:500]
        url = f"https://arxiv.org/abs/{pid}"
        pub_date = paper.get("publishedAt", paper.get("paper", {}).get("publishedAt", now))

        entry = f"""  <entry>
    <id>{url}</id>
    <title>{escape_xml(title)}</title>
    <link href="{url}" rel="alternate"/>
    <summary type="html">{escape_xml(summary[:400])}</summary>
    <published>{pub_date}</published>
    <updated>{now}</updated>
    <category term="research"/>
    <author><name>HF Daily Papers</name></author>
    <score>{score}</score>
  </entry>"""
        entries.append(entry)

    rss = f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>{title_prefix} — Agent N's Hermes News</title>
  <link href="https://huggingface.co/api/daily_papers" rel="self"/>
  <updated>{now}</updated>
  <author><name>Agent N's Hermes News</name></author>
  <id>tag:hf-daily,2026://filtered</id>
  <generator>hf_daily_filter.py</generator>
{chr(10).join(entries)}
</feed>"""
    return rss


def escape_xml(s):
    """Escape XML special characters."""
    if not s:
        return ""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


def main():
    parser = argparse.ArgumentParser(description="Filter HF daily papers to notable ones")
    parser.add_argument("--max", type=int, default=8, help="Max papers to include (default: 8)")
    parser.add_argument("--min-score", type=int, default=10, help="Minimum score threshold (default: 10)")
    parser.add_argument("--output", "-o", type=str, default="", help="Output file path")
    args = parser.parse_args()

    try:
        papers = fetch_daily_papers()
    except Exception as e:
        print(f"Error fetching daily papers: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(papers)} daily papers", file=sys.stderr)

    # Score and filter
    scored = []
    for p in papers:
        paper_data = p.get("paper", p)
        score, reasons = score_paper(paper_data)
        if score >= args.min_score:
            scored.append((p, score, reasons))

    # Sort by score descending
    scored.sort(key=lambda x: -x[1])

    print(f"Passed score filter: {len(scored)} papers", file=sys.stderr)
    for p, score, reasons in scored[:args.max]:
        title = p.get("title", p.get("paper", {}).get("title", "?"))
        print(f"  [{score:2d}] {title[:70]}", file=sys.stderr)

    # Trim to max
    final = scored[:args.max]

    # Generate RSS
    rss = generate_rss([(p, s, r) for p, s, r in final])

    if args.output:
        with open(args.output, "w") as f:
            f.write(rss)
        print(f"Written {len(final)} papers to {args.output}", file=sys.stderr)
    else:
        print(rss)


if __name__ == "__main__":
    main()