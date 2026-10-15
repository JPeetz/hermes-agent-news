"""
Shared Reddit gate policy (2026-10-15).

Both the orchestrator's pre-analysis gate and the publish validator in
validate_report.py enforce "Reddit is a required daily source". They must
agree, or a run dies before analysis on a day the validator would have
published (or vice versa: an analysis the validator rejects wastes ~$8 and
20 minutes). This module is the single definition of "usable Reddit".

History:
- b001bff (2026-09-12) made ANY self-reported degradation fatal, added
  because a silent arXiv-style gap had published as green (2026-08-17).
- 2026-10-15 proved that's too strict: one intermittent ScrapeCreators 404
  on one busy subreddit's pagination cursor left r/ChatGPT at ~20% of its
  normal yield, and the whole day died four times over it (537 posts
  collected, 14/15 subreddits at full volume, credits healthy).

Policy now:
- 'failed' (or missing / zero posts) is still fatal: Reddit is a required
  source and an empty category is an outage, not a quiet day.
- 'partial' is publishable when the category cleared an absolute floor.
  The degradation stays VISIBLE: the source row keeps status 'partial' and
  the reason, so the day's data says exactly what was collected.
- A partial below the floor is fatal, because a real source-level collapse
  (the arXiv pattern: 12 items against a 334 median) shows up as a low
  count, and a silent low-count day must not publish.
"""

import os

# Absolute minimum Reddit posts the day may publish with. Recent healthy
# days run 550-690 (2026-10-08..14: 690/684/636/557/609/606); the 2026-10-15
# partial-failure runs still collected 529-542. A floor of 300 tolerates one
# dead mid-size subreddit but still catches a source-level collapse (the
# arXiv precedent was 12 items against a 334-median day).
_DEFAULT_MIN_PUBLISH_COUNT = 300

try:
    REDDIT_MIN_PUBLISH_COUNT = max(1, int(os.environ.get("REDDIT_MIN_PUBLISH_COUNT", _DEFAULT_MIN_PUBLISH_COUNT)))
except ValueError:
    REDDIT_MIN_PUBLISH_COUNT = _DEFAULT_MIN_PUBLISH_COUNT


def reddit_gate_ok(status: str | None, count: int | None) -> tuple[bool, str | None]:
    """
    Decide whether a Reddit collection row may proceed to analysis/publish.

    Returns (ok, reason). ``reason`` is None when ok, else a human-readable
    explanation suitable for the RuntimeError message / validation failure.
    """
    count = count or 0
    if status == "success":
        if count > 0:
            return True, None
        return False, "reddit collection reported success with zero posts"
    if status == "partial":
        if count >= REDDIT_MIN_PUBLISH_COUNT:
            return True, None
        return False, (
            f"reddit collection degraded below the publish floor "
            f"({count} posts < {REDDIT_MIN_PUBLISH_COUNT})"
        )
    if status is None:
        return False, "no reddit collection status"
    return False, f"reddit collection status is '{status}'"
