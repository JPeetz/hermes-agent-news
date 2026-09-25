"""
Reddit Gatherer - Collects posts from Reddit subreddits via the ScraperAPI proxy.

Reddit's free unauthenticated ``.json`` endpoint is blocked at the endpoint level
(HTTP 403 from every exit IP) and OAuth is also unavailable. This gatherer uses
ScraperAPI (``api.scraperapi.com``) as a proxy to Reddit's native JSON API, which
unblocks Reddit server-side and returns native ``.json``-equivalent data.

Two endpoint patterns are used (both proxied through ScraperAPI):
  * ``GET /r/{subreddit}/.json?sort=...&after=...``  -> listing (discovery + ranking)
  * ``GET /r/{subreddit}/comments/{id}/.json``        -> per-post body (selftext) + top comments

Collection strategy: ``sort=new`` (strictly reverse-chronological) is paged
newest -> oldest and stopped once the coverage window is passed. This is both
credit-cheap (only pages that overlap the window are fetched) and complete for a
date-bounded run. Top-scoring posts are then enriched: self posts get their body
text; high-discussion link posts get a digest of the top community comments
(the same call returns both, so comments come "for free").
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

from ..base import BaseGatherer, CollectedItem, deduplicate_items

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Read a non-negative int from the environment, falling back on bad input."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


# ScraperAPI configuration
SCRAPECREATORS_API_KEY = os.getenv("SCRAPECREATORS_API_KEY", "")
SCRAPECREATORS_BASE = os.getenv("SCRAPECREATORS_BASE", "https://api.scraperapi.com")

# Egress: ScraperAPI unblocks Reddit server-side, so its calls go DIRECT and must
# NOT be captured by the pipeline-wide HTTPS_PROXY/ALL_PROXY exports (Mullvad).
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "AI-News-Aggregator/1.0")

# Tunables (all env-overridable)
REDDIT_SORT = os.getenv("REDDIT_SORT", "new")
REDDIT_MAX_PAGES = _env_int("REDDIT_MAX_PAGES", 20, minimum=1)
REDDIT_BODY_TOP_N = _env_int("REDDIT_BODY_TOP_N", 12, minimum=0)
REDDIT_MIN_COMMENTS_FOR_DIGEST = _env_int("REDDIT_MIN_COMMENTS_FOR_DIGEST", 8, minimum=0)
REDDIT_CREDIT_BUDGET = _env_int("REDDIT_CREDIT_BUDGET", 600, minimum=1)
REDDIT_FETCH_WORKERS = _env_int("REDDIT_FETCH_WORKERS", 6, minimum=1)
# Consecutive older-than-window posts that trigger a stop. >1 absorbs out-of-order
# pinned/stickied posts at the top of a listing.
REDDIT_OLDER_STOP_THRESHOLD = _env_int("REDDIT_OLDER_STOP_THRESHOLD", 3, minimum=1)
REDDIT_REQUEST_TIMEOUT = _env_int("REDDIT_REQUEST_TIMEOUT", 60, minimum=5)

# Listing/detail status codes that are retryable transient failures
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)


class FatalScrapeError(Exception):
    """Non-recoverable error - abort the run."""


class RedditGatherer(BaseGatherer):
    """Gathers posts from Reddit subreddits via ScraperAPI proxy."""

    def __init__(
        self,
        config_dir: str = './config',
        data_dir: str = './data',
        lookback_hours: int = 24,
        target_date: Optional[str] = None
    ):
        super().__init__(config_dir, data_dir, lookback_hours, target_date)
        self.subreddits = self.load_config_list('reddit_subreddits.txt')

        # Config snapshot (instance copies so tests/overrides are explicit)
        self.sort = REDDIT_SORT
        self.max_pages = REDDIT_MAX_PAGES
        self.body_top_n = REDDIT_BODY_TOP_N
        self.min_comments_for_digest = REDDIT_MIN_COMMENTS_FOR_DIGEST
        self.credit_budget = REDDIT_CREDIT_BUDGET
        self.fetch_workers = REDDIT_FETCH_WORKERS
        self.older_stop_threshold = REDDIT_OLDER_STOP_THRESHOLD
        self.timeout = REDDIT_REQUEST_TIMEOUT

        # Shared, thread-safe run state (gathering runs across a thread pool)
        self._lock = Lock()
        self._calls_made = 0            # HTTP attempts issued this run (budget unit)
        self._credits_remaining: Optional[int] = None  # last observed balance
        self._stop_calls = False        # set when budget hit or a fatal error occurs
        self._scraperapi_dead = False   # circuit-breaker: set True after first failure

        if not self.subreddits:
            # Default subreddits if none configured
            self.subreddits = [
                'MachineLearning',
                'artificial',
                'LocalLLaMA',
                'ChatGPT',
                'OpenAI'
            ]

    @property
    def category(self) -> str:
        return 'reddit'

    def note_degradation(self, reason: str) -> None:
        with self._lock:
            super().note_degradation(reason)

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    async def gather(self) -> List[CollectedItem]:
        """Gather posts from configured subreddits."""
        if not SCRAPECREATORS_API_KEY:
            self.note_degradation('SCRAPECREATORS_API_KEY is missing (required for ScraperAPI proxy)')
            logger.error(
                "SCRAPECREATORS_API_KEY is not set - Reddit collection is disabled. "
                "Set the env var / GitHub secret to restore Reddit data."
            )
            self.save_to_file([], f'reddit_{self.target_date}.json')
            return []

        # Backfill is depth-limited: sort=new pages from "now" backwards, so a coverage
        # window many days in the past costs a lot of pages and may hit the page cap.
        days_back = (datetime.now().date() - datetime.strptime(self.coverage_date, '%Y-%m-%d').date()).days
        if days_back > 2:
            logger.warning(
                f"Reddit coverage date {self.coverage_date} is {days_back} days back; "
                f"sort=new backfill is depth-limited (max_pages={self.max_pages}) and may under-collect."
            )

        logger.info(f"Starting Reddit collection from {len(self.subreddits)} subreddits (sort={self.sort})")

        # Run blocking code in an owned thread pool so Reddit's work doesn't consume the
        # shared default executor used by the concurrently-running social/research gatherers.
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='reddit-driver') as driver:
            with self.time_source('reddit'):
                all_posts = await loop.run_in_executor(driver, self._gather_sync)

        logger.info(f"Collected {len(all_posts)} posts from Reddit")
        if not all_posts:
            self.note_degradation('Reddit returned no posts for the coverage window')
        self.save_to_file(all_posts, f'reddit_{self.target_date}.json')
        return all_posts

    # ------------------------------------------------------------------ #
    # Orchestration (synchronous, thread-pool based)
    # ------------------------------------------------------------------ #

    def _gather_sync(self) -> List[CollectedItem]:
        """Fetch all subreddits concurrently; runs in a worker thread."""
        start_balance = self._fetch_credit_balance()
        if start_balance is not None:
            logger.info(f"ScraperAPI balance at start: {start_balance}")
            self._credits_remaining = start_balance
            if start_balance <= 0:
                self._stop_calls = True
                self.note_degradation(f'ScraperAPI credits exhausted (balance={start_balance})')
                raise FatalScrapeError(self.get_degradation())

        all_posts: List[CollectedItem] = []

        def fetch_timed(sub: str) -> List[CollectedItem]:
            """One subreddit, timed as its own replay step.

            Timed inside the worker so the span is the fetch itself, not the wait for
            a pool slot -- with `fetch_workers` below the subreddit count, queued subs
            would otherwise all appear to start at once.
            """
            with self.time_step('reddit', f'r/{sub}') as step:
                posts = self._fetch_subreddit(sub)
                step.items = len(posts)
                # `_fetch_subreddit` catches everything -- a fatal proxy error
                # or a blown call budget returns a short list rather than
                # raising, so `time_step` cannot infer failure from an exception.
                # Without this, an aborted run draws 15 clean green bars.
                with self._lock:
                    stopped = self._stop_calls
                if stopped:
                    step.status = 'partial'
                return posts

        with ThreadPoolExecutor(max_workers=self.fetch_workers, thread_name_prefix='reddit-sub') as ex:
            future_to_sub = {ex.submit(fetch_timed, sub): sub for sub in self.subreddits}
            for future in as_completed(future_to_sub):
                sub = future_to_sub[future]
                try:
                    all_posts.extend(future.result())
                except Exception as e:  # defensive: a worker should not crash the run
                    logger.error(f"r/{sub} worker failed: {e}")
                    self.note_degradation(f'r/{sub} worker failed: {type(e).__name__}')

        # Belt-and-suspenders dedup across subs (per-sub dedup already applied).
        all_posts = deduplicate_items(all_posts)

        # Concurrent responses can arrive out of order; use a free final probe
        # as the authoritative end balance rather than whichever response won.
        final_balance = self._fetch_credit_balance()
        if final_balance is not None:
            self._credits_remaining = final_balance

        with self._lock:
            calls = self._calls_made
            remaining = self._credits_remaining
            stopped = self._stop_calls
        consumed = max(0, start_balance - remaining) if (start_balance is not None and remaining is not None) else None
        if stopped:
            self.note_degradation('Reddit collection stopped before completion (budget or provider failure)')
        logger.info(
            f"ScraperAPI usage: {calls} calls this run; "
            f"credits_remaining={remaining}; credits_consumed={consumed}"
            + ("; STOPPED EARLY (budget/fatal)" if stopped else "")
        )

        # Surface credit usage/balance in the end-of-run cost summary.
        try:
            from ..cost_tracker import get_tracker
            # ScraperAPI bills per call; no per-credit balance available.
            billed = consumed if consumed is not None else calls
            get_tracker().record_external_api(
                "ScraperAPI (Reddit)",
                calls=calls,
                credits_consumed=consumed,
                balance=remaining,
                est_cost_usd=round((billed or 0) * 0.99 / 1000, 4),
                note=self.get_degradation(),
            )
        except Exception as e:  # never let reporting break collection
            logger.debug(f"Could not record ScraperAPI usage: {e}")

        return all_posts

    # ------------------------------------------------------------------ #
    # Per-subreddit collection
    # ------------------------------------------------------------------ #

    def _fetch_subreddit(self, subreddit: str) -> List[CollectedItem]:
        """Fetch in-window posts for one subreddit, then enrich the top-scoring ones."""
        session = self._make_session()
        pairs: List[tuple] = []  # (CollectedItem, raw_post_dict) kept aligned for enrichment
        seen_ids: set = set()
        after = None
        pages = 0
        consecutive_older = 0

        try:
            while pages < self.max_pages:
                if self._stop_calls:
                    break

                params = {"subreddit": subreddit, "sort": self.sort}
                if after:
                    params["after"] = after

                data = self._api_get(session, "/v1/reddit/subreddit", params)
                if data is None:  # soft failure or stop_calls
                    break

                listing = data.get("data") if isinstance(data, dict) else None
                if not isinstance(listing, dict):
                    self.note_degradation(f'r/{subreddit}: malformed listing response')
                    break
                children = listing.get("children") or []
                if not isinstance(children, list):
                    self.note_degradation(f'r/{subreddit}: children is not a list')
                    break
                if not children:
                    break

                for child in children:
                    post = child.get("data", {}) if isinstance(child, dict) else {}
                    if not isinstance(post, dict):
                        continue
                    post_id = post.get("id", "")
                    if not post_id or post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)

                    # Reddit JSON flags stickied posts natively; skip them.
                    if post.get("stickied"):
                        continue

                    created = post.get("created_utc")
                    if not created:
                        continue
                    try:
                        pub_dt = datetime.fromtimestamp(created)
                    except (ValueError, OSError, OverflowError):
                        continue

                    if pub_dt > self.end_time:
                        # "Today overhang": newer than the coverage window. Keep paging.
                        consecutive_older = 0
                        continue
                    if pub_dt < self.start_time:
                        # Older than the window. With sort=new everything below is older too,
                        # but absorb a few out-of-order pinned posts before committing to stop.
                        consecutive_older += 1
                        continue

                    consecutive_older = 0
                    pairs.append((self._build_item(subreddit, post, pub_dt), post))

                if consecutive_older >= self.older_stop_threshold:
                    break

                after = listing.get("after")
                if not after:
                    break
                pages += 1

            logger.info(f"r/{subreddit}: collected {len(pairs)} in-window posts across {pages + 1} page(s)")
            if pages >= self.max_pages:
                self.note_degradation(f'r/{subreddit}: max_pages={self.max_pages} reached before coverage completed')
                logger.warning(
                    f"r/{subreddit}: hit max_pages={self.max_pages} before exhausting the window; "
                    f"may be under-collecting."
                )

            if pairs and not self._stop_calls:
                self._enrich_pairs(session, subreddit, pairs)

        except FatalScrapeError as e:
            with self._lock:
                self._stop_calls = True
            logger.error(f"Aborting Reddit collection (fatal): {e}")
            self.note_degradation(str(e))
        except Exception as e:
            logger.error(f"Error fetching r/{subreddit}: {e}")
            self.note_degradation(f'r/{subreddit}: {type(e).__name__}')
        finally:
            session.close()

        return [item for item, _ in pairs]

    def _build_item(self, subreddit: str, post: Dict[str, Any], pub_dt: datetime) -> CollectedItem:
        """Map a Reddit native JSON post to a CollectedItem (body filled in later)."""
        post_id = post.get("id", "")
        title = post.get("title", "") or ""
        domain = (post.get("domain") or "").lower()
        return CollectedItem(
            id=self.generate_id('reddit', post_id),
            title=title,
            content="",  # enriched later for top-N posts
            url=f"https://reddit.com{post.get('permalink', '')}",
            author=f"u/{post.get('author', '')}",
            published=pub_dt.isoformat(),
            source=f"r/{subreddit}",
            source_type='reddit',
            tags=[],  # flair not exposed by Reddit .json listings
            metadata={
                'platform_id': post_id,
                'subreddit': subreddit,
                'external_url': post.get('url', ''),
                'is_self': domain.startswith('self.'),
                'engagement': {
                    'score': post.get('score', 0) or 0,
                    'upvote_ratio': post.get('upvote_ratio', 0) or 0,
                    'num_comments': post.get('num_comments', 0) or 0,
                },
            },
            keywords=self.extract_keywords(title),
        )

    # ------------------------------------------------------------------ #
    # Body / comment enrichment
    # ------------------------------------------------------------------ #

    def _enrich_pairs(self, session: requests.Session, subreddit: str, pairs: List[tuple]) -> None:
        """Enrich the top-N posts (by score) with body text and/or a comment digest."""
        ranked = sorted(pairs, key=lambda p: p[1].get("score", 0) or 0, reverse=True)
        for item, post in ranked[:self.body_top_n]:
            if self._stop_calls:
                break
            try:
                self._enrich_one(session, item, post)
            except FatalScrapeError:
                raise  # bubble to _fetch_subreddit to stop the whole run
            except Exception as e:
                logger.warning(f"Enrichment failed for {item.url}: {e}")

    def _enrich_one(self, session: requests.Session, item: CollectedItem, post: Dict[str, Any]) -> None:
        """One post/comments call: self -> selftext; link -> top-comment discussion digest."""
        is_self = (post.get("domain") or "").lower().startswith("self.")
        num_comments = post.get("num_comments", 0) or 0

        # Link posts with little discussion aren't worth a credit (their substance is the
        # linked article, captured elsewhere). Self posts are always enriched (body text).
        if not is_self and num_comments < self.min_comments_for_digest:
            return

        permalink = post.get("permalink", "")
        if not permalink:
            return

        data = self._api_get(session, "/v1/reddit/post/comments", {"url": f"https://www.reddit.com{permalink}"})
        if data is None:
            return

        # Reddit native JSON comments: [post_listing, comments_listing]
        if isinstance(data, list) and len(data) >= 2:
            post_children = data[0].get("data", {}).get("children", []) if isinstance(data[0], dict) else []
            detail = post_children[0].get("data", {}) if post_children else {}
            comment_children = data[1].get("data", {}).get("children", []) if isinstance(data[1], dict) else []
            comments = [c.get("data", {}) for c in comment_children if isinstance(c, dict)]
        else:
            detail = {}
            comments = []

        content = ""
        if is_self:
            content = (detail.get("selftext") or "").strip()
        if not content:
            # Link post (or empty self post) -> digest the community discussion.
            content = self._build_comment_digest(comments)

        if content:
            item.content = content
            item.keywords = self.extract_keywords(f"{item.title} {content}")

    @staticmethod
    def _build_comment_digest(comments: List[Dict[str, Any]], max_comments: int = 6, max_len: int = 220) -> str:
        """Build a compact, markdown (HTML-safe via downstream nh3) top-comments digest."""
        cleaned = []
        for c in comments:
            body = (c.get("body") or "").strip()
            if not body:
                continue
            author = (c.get("author") or "").lower()
            if author in ("automoderator", "[deleted]"):
                continue
            low = body.lower()
            if "i am a bot" in low or "performed automatically" in low:
                continue
            body = " ".join(body.split())  # collapse newlines so each comment is one bullet
            if len(body) > max_len:
                body = body[:max_len].rstrip() + "…"
            cleaned.append((c.get("score", 0) or 0, body))

        if not cleaned:
            return ""

        cleaned.sort(key=lambda x: x[0], reverse=True)
        lines = ["**Top community comments:**", ""]
        for score, body in cleaned[:max_comments]:
            lines.append(f"- (▲{score}) {body}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # HTTP layer
    # ------------------------------------------------------------------ #

    def _make_session(self) -> requests.Session:
        """Create a session that ignores ambient proxy env vars (direct egress by default)."""
        session = requests.Session()
        session.headers.update({"User-Agent": REDDIT_USER_AGENT})
        # Ignore HTTPS_PROXY/ALL_PROXY exported pipeline-wide (Mullvad) - ScraperAPI
        # unblocks server-side and must go direct.
        session.trust_env = False
        return session

    def _fetch_credit_balance(self) -> Optional[int]:
        """ScraperAPI has no credit balance endpoint."""
        return None

    def _api_get(self, session: requests.Session, path: str, params: dict) -> Optional[Dict[str, Any]]:
        """
        Budgeted GET through ScraperAPI proxy to Reddit's native JSON API.

        Builds the target Reddit JSON URL from params, wraps it in a ScraperAPI
        proxy call (``?api_key=...&url=...``), and returns the parsed native JSON
        on success. Returns None on a soft failure / when the call budget is exhausted.
        No credit-balance endpoint is available.
        """
        # Quick health check: skip fast if the API key is missing or dead
        if not SCRAPECREATORS_API_KEY or len(SCRAPECREATORS_API_KEY) < 10:
            logger.warning("SCRAPECREATORS_API_KEY not set or too short; skipping Reddit")
            self.note_degradation('SCRAPECREATORS_API_KEY missing or short')
            return None

        # Build the target Reddit native JSON URL from the endpoint params.
        if "url" in params:
            # Comments/detail endpoint: params["url"] is a Reddit post URL
            target_url = params["url"].rstrip("/") + ".json"
        else:
            # Listing endpoint: params has subreddit, sort, optional after
            subreddit = params.get("subreddit", "")
            sort = params.get("sort", "new")
            after = params.get("after")
            target_url = f"https://www.reddit.com/r/{subreddit}/.json?sort={sort}"
            if after:
                target_url += f"&after={after}"

        # Wrap through ScraperAPI
        proxy_params = {
            "api_key": SCRAPECREATORS_API_KEY,
            "url": target_url,
        }

        for attempt in range(3):
            with self._lock:
                if self._stop_calls or self._scraperapi_dead:
                    return None
                if self._calls_made >= self.credit_budget:
                    self._stop_calls = True
                    logger.warning('Reddit call budget (%s calls) reached', self.credit_budget)
                    return None
                self._calls_made += 1
            try:
                resp = session.get(SCRAPECREATORS_BASE, params=proxy_params, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                delay = 2 ** attempt
                logger.warning(f"ScraperAPI request error for {target_url} ({e}); retrying in {delay}s")
                time.sleep(delay)
                continue

            status = resp.status_code

            if status in _RETRYABLE_STATUS:
                delay = 2 ** attempt
                logger.warning(f"ScraperAPI HTTP {status} for {target_url}; retrying in {delay}s")
                time.sleep(delay)
                continue

            if status not in (200, *range(429, 600)):
                break
            if status != 200:
                with self._lock:
                    self._scraperapi_dead = True
                logger.warning(f"ScraperAPI HTTP {status} for {target_url}; skipping, circuit-breaker set")
                self.note_degradation(f'ScraperAPI HTTP {status} for {target_url}')
                return None

            try:
                data = resp.json()
            except ValueError:
                logger.warning(f"ScraperAPI returned non-JSON for {target_url}; skipping")
                self.note_degradation(f'ScraperAPI returned non-JSON for {target_url}')
                return None

            # Reddit native JSON — no success/credit fields; response IS the data.
            return data

        logger.warning(f"ScraperAPI request to {target_url} failed after retries; circuit-breaker set")
        with self._lock:
            self._scraperapi_dead = True
        self.note_degradation(f'ScraperAPI {path} failed after retries')
        return None
