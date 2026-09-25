"""
Reddit Gatherer — DISABLED (replaced by native Reddit RSS feeds).

Reddit content is now ingested via RSS feeds (no proxy required),
added to `config/rss_feeds.txt`:

  https://www.reddit.com/r/hermesagent+LocalLLaMA+nousresearch/.rss
  https://www.reddit.com/user/NousResearch/.rss
  https://www.reddit.com/r/hermesagent/search.rss?q=Hermes+Desktop&restrict_sr=1

The news gatherer (feedparser) handles these. This gatherer stays as a
placeholder so the pipeline's import chain doesn't break, but returns empty.
"""

import asyncio
import logging
from typing import List, Optional

from ..base import BaseGatherer, CollectedItem

logger = logging.getLogger(__name__)


class RedditGatherer(BaseGatherer):
    """Reddit gatherer — no-op. Reddit is now collected via RSS feeds."""

    def __init__(
        self,
        config_dir: str = './config',
        data_dir: str = './data',
        lookback_hours: int = 24,
        target_date: Optional[str] = None
    ):
        super().__init__(config_dir, data_dir, lookback_hours, target_date)

    @property
    def category(self) -> str:
        return 'reddit'

    async def gather(self) -> List[CollectedItem]:
        """No-op: Reddit collected via RSS. Returns empty list."""
        logger.info("Reddit gatherer disabled — Reddit content now collected via RSS feeds in news gatherer")
        return []