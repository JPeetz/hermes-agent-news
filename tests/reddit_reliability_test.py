"""Regression cases for Reddit publication gating, credit health and warnings.

The scraper-API path (RedditGatherer with credit budgets, `_api_get`,
subreddit fetches, FatalScrapeError) was REMOVED when Reddit moved to native
RSS feeds (2026-11): `agents/gatherers/reddit_gatherer.py` is now a no-op
placeholder and Reddit content flows through the news gatherer's feedparser
path. Tests for the removed machinery were retired with it; what remains is the
still-live gating logic (publication validation, credit reserve accounting,
pipeline alerts). The RSS-path recovery behavior is covered by
`reddit_analysis_recovery_test` and `analysis_identity_test`.

Run explicitly: venv/bin/python -m unittest tests.reddit_reliability_test -v
"""
from unittest.mock import patch
import unittest
from scripts.validate_report import validate
from scripts.reddit_credit_health import reserve
from scripts.pipeline_alert import post_alert


class RedditReliabilityTests(unittest.TestCase):
    def test_empty_reddit_is_optional_and_does_not_block_publish(self):
        # Reddit is optional since the scraper API was retired (validate_report
        # downgraded its reddit check from failure to warning). An edition that
        # collected zero Reddit items but analyzed other content must validate.
        summary = {'date': '2026-09-12', 'executive_summary': 'A' * 500,
                   'top_topics': [{'name': 'AI'}], 'total_items_analyzed': 100,
                   'categories': {'reddit': {'count': 0}},
                   'collection_status': {'sources': [{'name': 'reddit', 'status': 'success', 'count': 0}]}}
        result = validate(summary, '2026-09-12')
        self.assertTrue(result['valid'])

    def test_zero_analyzed_items_still_blocks_publish(self):
        # The analysis-wipeout guard in validate_report is the real hard gate:
        # collect everything, analyze nothing, and the edition must not publish.
        summary = {'date': '2026-09-12', 'executive_summary': 'A' * 500,
                   'top_topics': [{'name': 'AI'}], 'total_items_analyzed': 0,
                   'categories': {'news': {'count': 42}},
                   'collection_status': {'sources': [{'name': 'news', 'status': 'success', 'count': 42}]}}
        result = validate(summary, '2026-09-12')
        self.assertFalse(result['valid'])
        self.assertTrue(any('analyzed' in f for f in result['failures']))

    def test_reserve_uses_complete_daily_consumption_only(self):
        rows = [dict(date='2026-09-09', consumed=250, complete=True),
                dict(date='2026-09-10', consumed=85, complete=False)]
        self.assertEqual(reserve(rows), (750, 250))
        self.assertEqual(reserve([]), (1800, 600))

    def test_alert_bypasses_collection_proxy_and_rejects_failed_delivery(self):
        with patch.dict('os.environ', {'PIPELINE_ALERT_TOKEN': 'test-only'}), \
             patch('scripts.pipeline_alert.urllib.request.ProxyHandler') as proxy, \
             patch('scripts.pipeline_alert.urllib.request.build_opener') as opener:
            reply = opener.return_value.open.return_value.__enter__.return_value
            reply.status = 403
            reply.read.return_value = b'{"ok": false}'
            self.assertFalse(post_alert({'status': 'degraded'}))
            proxy.assert_called_once_with({})


if __name__ == '__main__':
    unittest.main()
