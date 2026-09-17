"""Offline contracts for the bounded news shadow coordinator.

These tests intentionally use only the coordinator's JSON/state layer.  They
do not invoke a source API, an evaluator, a model, or a workflow.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadow.coordinator import (  # noqa: E402
    Coordinator,
    DiscoveryError,
    SelectorError,
    TrustedRefError,
    classify_source_run,
    discover_metadata,
    parse_run_attempts,
    parse_selector,
    validate_trusted_ref,
)
from shadow.state import (  # noqa: E402
    ExperimentIdentity,
    StateConflict,
    StateStore,
    STATUS_COMPLETED,
    STATUS_RUNNING,
)


def _run(run_id: int, *, attempt: int = 1, report_date: str = "2026-09-10", healthy=True):
    return {
        "id": run_id,
        "run_attempt": attempt,
        "path": ".github/workflows/daily-pipeline.yml",
        "status": "completed",
        "conclusion": "success",
        "report_date": report_date,
        "source_health": {"overall": "success" if healthy else "partial"},
        "publication": {"status": "published"},
    }


class SelectorTests(unittest.TestCase):
    def test_attempt_is_part_of_selector(self):
        self.assertEqual(parse_run_attempts("123:1, 123:2")[1].attempt, 2)
        self.assertEqual(parse_selector(source_runs="123").source_runs[0].attempt, 1)

    def test_mixed_and_oversized_selectors_are_rejected(self):
        with self.assertRaises(SelectorError):
            parse_selector(source_runs="123:1", from_date="2026-09-10", to_date="2026-09-10")
        with self.assertRaises(SelectorError):
            parse_selector(from_date="2026-09-01", to_date="2026-09-15")

    def test_only_main_is_trusted(self):
        self.assertEqual(validate_trusted_ref("refs/heads/main"), "main")
        with self.assertRaises(TrustedRefError):
            validate_trusted_ref("refs/pull/42/merge")


class DiscoveryTests(unittest.TestCase):
    def test_report_date_is_explicit_and_health_is_separate(self):
        healthy = classify_source_run(_run(10))
        degraded = classify_source_run(_run(11, healthy=False))
        self.assertTrue(healthy.eligible)
        self.assertFalse(degraded.metadata_eligible)
        self.assertEqual(healthy.report_date, "2026-09-10")

    def test_workflow_timestamp_cannot_select_a_report_date(self):
        metadata = _run(10)
        metadata.pop("report_date")
        metadata["created_at"] = "2026-09-10T12:00:00Z"
        rows = discover_metadata(
            [metadata], parse_selector(from_date="2026-09-10", to_date="2026-09-10")
        )
        self.assertEqual(rows, [])


class StateTests(unittest.TestCase):
    def _identity(self, run_id=1):
        return ExperimentIdentity(
            repository="flyryan/ai-news-aggregator",
            source_run_id=run_id,
            source_attempt=1,
            bundle_sha256="bundle",
            replay_sha="replay",
            dependency_lock="lock",
            container_digest="container",
            mode="filter",
            candidate_policy="news-relevance-v1-dev",
            rubric_version="r1",
            candidate_model="jev-1.13.0",
            judge_version="judge-v1",
            judge_model="deepseek",
            sampling_plan="all",
            repeat_plan="none",
        )

    def test_discovery_deduplicates_and_revision_is_optimistic(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(directory)
            row = state.ensure_discovered(self._identity())
            self.assertEqual(state.ensure_discovered(self._identity())["experiment_id"], row["experiment_id"])
            revision = state.revision
            with self.assertRaises(StateConflict):
                state.record_status(row["experiment_id"], STATUS_RUNNING, expected_revision=revision - 1)

    def test_serial_lease_is_bounded_to_one_active_record(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(directory)
            state.ensure_discovered(self._identity(1))
            state.ensure_discovered(self._identity(2))
            first = state.claim_next(owner="worker", limit=2)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["status"], STATUS_RUNNING)
            self.assertEqual(state.claim_next(owner="other", limit=2), [])
            state.release_lease(first[0]["experiment_id"], owner="worker", status=STATUS_COMPLETED)
            second = state.claim_next(owner="other", limit=2)
            self.assertEqual(len(second), 1)

    def test_coordinator_claims_only_bundles_present_in_the_current_job(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator(directory)
            rows = [coordinator.state.ensure_discovered(
                self._identity(run_id),
                eligibility={"auto_eligible": True, "importer_verified": True},
                capabilities={"filter_replay": True},
            ) for run_id in (1, 2)]
            claimed = coordinator.claim_next(owner="worker", experiment_ids={rows[1]["experiment_id"]})
            self.assertEqual([row["experiment_id"] for row in claimed], [rows[1]["experiment_id"]])

    def test_terminal_failure_needs_a_new_version_before_paid_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = Coordinator(directory)
            row = coordinator.state.ensure_discovered(
                self._identity(),
                eligibility={"auto_eligible": True, "importer_verified": True},
                capabilities={"filter_replay": True},
            )
            coordinator.state.record_status(row["experiment_id"], "failed")
            self.assertEqual(coordinator.claim_next(owner="worker"), [])

    def test_resealed_source_cannot_create_unrequested_paid_experiment(self):
        manifest = {"source": {"repository": "flyryan/ai-news-aggregator", "run_id": 123, "run_attempt": 1},
                    "report_date": "2026-09-17", "bundle_sha256": "a" * 64,
                    "capabilities": {"filter_replay": True},
                    "eligibility": {"healthy": True, "successful_run": True},
                    "publication": {"status": "published"}}
        for status in ("discovered", "running", "completed", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory, \
                    patch("shadow.coordinator.verify_bundle", return_value=manifest):
                coordinator = Coordinator(directory)
                first = coordinator.record_bundle("first", evaluator_sha="b" * 40,
                                                   repeat_plan="1", status=status)
                with patch("shadow.coordinator.verify_bundle", return_value={**manifest, "bundle_sha256": "c" * 64}):
                    with self.assertRaisesRegex(DiscoveryError, "already sealed"):
                        coordinator.record_bundle("second", evaluator_sha="b" * 40, repeat_plan="1")
                    retried = coordinator.record_bundle("second", evaluator_sha="b" * 40, repeat_plan="2")
                    self.assertNotEqual(first["experiment_id"], retried["experiment_id"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
