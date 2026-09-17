"""Offline integrity/budget cases; never contacts a model or source."""
import tempfile
import unittest
from pathlib import Path

from shadow.budget import BudgetExceeded, BudgetLimits, RequestBudget
from shadow.contracts import (
    BundleValidationError, bundle_path, read_json, seal_bundle, sha256_json,
    validate_decisions, verify_bundle, write_json,
)


class BundleContractsTest(unittest.TestCase):
    def test_tampered_and_extra_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "original.json", {"value": 1})
            seal_bundle(root, {"report_date": "2026-09-17", "capabilities": {}})
            verify_bundle(root, capability=None)
            write_json(root / "unexpected.json", {})
            with self.assertRaises(BundleValidationError):
                verify_bundle(root, capability=None)
            (root / "unexpected.json").unlink()
            write_json(root / "original.json", {"value": 2})
            with self.assertRaises(BundleValidationError):
                verify_bundle(root, capability=None)

    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for body in ('{"id":1,"id":2}', '{"p":NaN}'):
                path.write_text(body)
                with self.assertRaises(BundleValidationError):
                    read_json(path)

    def test_archive_paths_and_symlinks_do_not_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../outside", "/outside", "a//b", "a/./b", "a\\b"):
                with self.assertRaises(BundleValidationError):
                    bundle_path(directory, name)
            (Path(directory) / "link").symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(BundleValidationError):
                bundle_path(directory, "link/file")

    def test_abstention_retains_input_and_unknown_ids_fail(self):
        records = [{"id": "abc", "title": "title", "source": "source", "snippet": "snippet"}]
        good = [{"id": "abc", "decision": "abstain", "effective_keep": True}]
        self.assertEqual(validate_decisions(records, good), good)
        with self.assertRaises(BundleValidationError):
            validate_decisions(records, [{**good[0], "effective_keep": False}])
        with self.assertRaises(BundleValidationError):
            validate_decisions(records, [{**good[0], "id": "other"}])


class BudgetContractsTest(unittest.TestCase):
    def test_unknown_usage_consumes_reservation_not_zero(self):
        budget = RequestBudget(BudgetLimits(max_input_tokens=20, max_output_tokens=10))
        reservation = budget.reserve(input_tokens=15, output_tokens=5)
        budget.settle(reservation)
        self.assertEqual(budget.snapshot()["input_tokens_accounted"], 15)
        self.assertIsNone(budget.snapshot()["cost_usd"])
        with self.assertRaises(BudgetExceeded):
            budget.reserve(input_tokens=6, output_tokens=1)

    def test_usage_reconciles_but_retries_count_and_cannot_double_settle(self):
        budget = RequestBudget(BudgetLimits(max_requests=2))
        first = budget.reserve(input_tokens=100, output_tokens=10)
        budget.settle(first, input_tokens=5, output_tokens=3, cost_usd=0.01)
        second = budget.reserve(input_tokens=100, output_tokens=10)
        budget.settle(second, input_tokens=6, output_tokens=2, cost_usd=0.02)
        self.assertEqual(budget.snapshot()["input_tokens_accounted"], 11)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(input_tokens=1, output_tokens=1)
        with self.assertRaises(ValueError):
            budget.settle(first)

    def test_time_limit_is_enclosing(self):
        now = [0.0]
        budget = RequestBudget(BudgetLimits(deadline_seconds=10), clock=lambda: now[0])
        now[0] = 11
        with self.assertRaises(BudgetExceeded):
            budget.reserve(input_tokens=1, output_tokens=1)


if __name__ == "__main__":
    unittest.main()
