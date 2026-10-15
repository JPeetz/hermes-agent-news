"""Regression cases for empty Reddit publication, credit exhaustion and warnings.

Run explicitly: venv/bin/python -m unittest tests.reddit_reliability_test -v
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from agents.gatherers.reddit_gatherer import RedditGatherer, FatalScrapeError
from scripts.validate_report import validate
from scripts.reddit_credit_health import reserve
from scripts.pipeline_alert import post_alert


class RedditReliabilityTests(unittest.TestCase):
    def test_credit_exhaustion_prevents_any_listing_requests(self):
        with TemporaryDirectory() as temp:
            g = RedditGatherer(config_dir=temp, data_dir=temp, target_date='2026-09-12')
            with patch.object(g, '_fetch_credit_balance', return_value=-5), patch.object(g, '_fetch_subreddit') as fetch:
                with self.assertRaises(FatalScrapeError):
                    g._gather_sync()
            fetch.assert_not_called()
            self.assertIn('exhausted', g.get_degradation())

    def test_retries_count_against_credit_budget_and_report_degradation(self):
        with TemporaryDirectory() as temp:
            g = RedditGatherer(config_dir=temp, data_dir=temp, target_date='2026-09-12')
            g.credit_budget = 2
            session = Mock()
            session.get.return_value.status_code = 503
            with patch('agents.gatherers.reddit_gatherer.time.sleep'):
                self.assertIsNone(g._api_get(session, '/v1/reddit/subreddit', {}))
            self.assertEqual(session.get.call_count, 2)
            self.assertTrue(g._stop_calls)

    def test_empty_reddit_cannot_publish_even_with_success_status(self):
        summary = {'date': '2026-09-12', 'executive_summary': 'A' * 500,
                   'top_topics': [{'name': 'AI'}], 'total_items_analyzed': 100,
                   'categories': {'reddit': {'count': 0}},
                   'collection_status': {'sources': [{'name': 'reddit', 'status': 'success', 'count': 0}]}}
        result = validate(summary, '2026-09-12')
        self.assertFalse(result['valid'])
        self.assertTrue(any('Reddit' in f for f in result['failures']))

    # ---- 2026-10-15: intermittent HTTP 404 on after-cursor listing pages ----
    # Three runs died because ONE 404 on a busy subreddit's page 2 ended
    # pagination mid-window and marked the whole Reddit source degraded,
    # tripping the orchestrator's strict-Reddit gate. Listing calls now retry
    # 404s and, if those fail, restart the page walk once before degrading.

    @staticmethod
    def _listing_session(responses_by_call, window_start, window_end):
        """Fake requests.Session returning scripted listing/detail responses."""
        from unittest.mock import Mock

        def _post(pid, ts):
            return {'id': pid, 'title': f'post {pid}', 'permalink': f'/r/test/comments/{pid}/x',
                    'created_utc': ts, 'num_comments': 0, 'stickied': None, 'selftext': 'body',
                    'is_self': True, 'url': '', 'score': 1, 'author': 'a'}

        mid = window_start + (window_end - window_start) / 2
        page = {'success': True, 'posts': [_post('p1', mid), _post('p2', mid + 1), _post('p3', mid + 2)],
                'after': 't3_cursor', 'credits_remaining': 1000}
        # A real cursor page that returns 200 but no posts ends the walk.
        end_page = {'success': True, 'posts': [], 'after': None, 'credits_remaining': 1000}

        state = {'listing_calls': 0}

        def fake_get(url, params=None, headers=None, timeout=None):
            resp = Mock()
            resp.json.return_value = page
            if '/post/comments' in url:
                resp.status_code = 200
                resp.json.return_value = {'success': True, 'post': {}, 'comments': []}
                return resp
            state['listing_calls'] += 1
            has_after = bool((params or {}).get('after'))
            scripted = responses_by_call(state['listing_calls'], has_after)
            resp.status_code = scripted
            if has_after:
                resp.json.return_value = end_page if scripted == 200 else {'success': False, 'message': 'not found'}
            return resp

        session = Mock()
        session.get.side_effect = fake_get
        return session

    def _gatherer_with_fake_pages(self, temp, responses_by_call):
        from unittest.mock import patch
        g = RedditGatherer(config_dir=temp, data_dir=temp, target_date='2026-09-12')
        g.subreddits = ['test']
        session = self._listing_session(responses_by_call, g.start_time.timestamp(), g.end_time.timestamp())
        with patch.object(g, '_make_session', return_value=session), \
             patch('agents.gatherers.reddit_gatherer.time.sleep'), \
             patch.object(g, '_fetch_credit_balance', return_value=1000):
            items = g._gather_sync()
        return g, items

    def test_listing_404_on_cursor_page_is_retried_and_recovers(self):
        """A single 404 on an after-cursor page must be retried, not terminal."""
        with TemporaryDirectory() as temp:
            def script(call, has_after):
                return 404 if (has_after and call == 2) else 200
            g, items = self._gatherer_with_fake_pages(temp, script)
            self.assertIsNone(g.get_degradation())
            self.assertEqual(len(items), 3)

    def test_persistent_cursor_404_restarts_walk_and_dedupes(self):
        """If cursor pages keep 404ing, restart once from page 1 and dedupe."""
        with TemporaryDirectory() as temp:
            def script(call, has_after):
                return 404 if has_after else 200
            g, items = self._gatherer_with_fake_pages(temp, script)
            # Restart happened; the first-page posts re-walked are deduped.
            self.assertEqual(len(items), 3)
            self.assertIn('pagination restart', g.get_degradation() or '')

    def test_non_listing_404_still_degrades_without_retry(self):
        """404 on a non-listing call keeps the old non-retryable behaviour."""
        from unittest.mock import Mock
        with TemporaryDirectory() as temp:
            g = RedditGatherer(config_dir=temp, data_dir=temp, target_date='2026-09-12')
            session = Mock()
            session.get.return_value.status_code = 404
            session.get.return_value.json.return_value = {'success': False, 'message': 'gone'}
            session.get.return_value.text = 'gone'
            self.assertIsNone(g._api_get(session, '/v1/reddit/post/comments', {}))
            self.assertEqual(session.get.call_count, 1)
            self.assertIn('HTTP 404', g.get_degradation() or '')

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
