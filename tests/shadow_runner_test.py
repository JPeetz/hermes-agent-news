"""Offline integration checks for immutable experiments and protected cohorts."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from shadow.contracts import BundleValidationError, seal_bundle, sha256_json, write_json
from shadow.budget import BudgetLimits, RequestBudget
from shadow.experiment import control_repeat_evidence, run_experiment
from shadow.runtime import ROOT, inspect_run, model_child_environment, validate_output_root


def make_bundle(path, report_date="2026-09-17"):
    records = [{"id": "a1", "title": "AI model", "source": "Example", "snippet": "A new frontier model."}]
    frozen = {"records": records, "ordered_ids": ["a1"], "input_sha256": sha256_json(records),
              "system_prompt": "Select frontier AI news.", "user_message": "ID: a1\nAI model"}
    decision = {"model": "deepseek-v4.1-flash", "input_sha256": frozen["input_sha256"],
                "decisions": [{"id": "a1", "decision": "keep", "effective_keep": True}]}
    write_json(path / "relevance/input.json", frozen)
    write_json(path / "relevance/incumbent-decision.json", decision)
    seal_bundle(path, {"report_date": report_date, "capabilities": {"filter_replay": True, "pipeline_replay": False},
        "eligibility": {"healthy": True, "successful_run": True},
        "publication": {"status": "published", "original_publication_verified": True},
        "source": {"repository": "flyryan/ai-news-aggregator", "run_id": 123, "run_attempt": 1}})
    return records, decision


class FakeAdapter:
    result = None

    def __init__(self, *args, **kwargs):
        pass

    async def evaluate(self, *args, **kwargs):
        return {**self.result, "status": "complete", "requests": []}


class FakeJudge:
    records = None
    result = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def adjudicate_inputs(self, records, output_path=None):
        result = {**self.result, "status": "complete", "adjudications": self.result.get("adjudications", []),
                  "requests": [], "errors": []}
        write_json(output_path, result)
        return result


class ShadowRunnerTest(unittest.TestCase):
    def test_repeat_checks_model_response_identity_not_gateway_request_identity(self):
        def artifact(response_id, request_id):
            return {"status": "complete", "requests": [{"attempts": [
                {"status": "success", "response_id": response_id, "request_id": request_id}]}]}
        control = artifact("completion-1", "gateway-1")
        reused = control_repeat_evidence(control, artifact("completion-1", "gateway-2"))
        self.assertFalse(reused["distinct_response_ids"])
        self.assertEqual(reused["basis"], "reused_model_response_id")
        self.assertFalse(control_repeat_evidence(control, artifact(None, "gateway-3"))["distinct_response_ids"])
        self.assertTrue(control_repeat_evidence(control, artifact("completion-2", "gateway-4"))["distinct_response_ids"])

    def test_development_budget_covers_five_maximum_output_judge_batches(self):
        from shadow.judge import JudgeConfig
        from shadow.runtime import budget_for
        from shadow.settings import load_policy
        config = JudgeConfig()
        budget = budget_for(load_policy(ROOT / "config/shadow/news-relevance-v1-dev.json"), "judge")
        for _ in range(5):
            reservation = budget.reserve(input_tokens=10000, output_tokens=config.max_output_tokens)
            budget.settle(reservation, input_tokens=10000, output_tokens=config.max_output_tokens)
        self.assertEqual(budget.snapshot()["requests"], 5)

    def test_failed_access_probe_preserves_safe_diagnostics(self):
        from shadow.preflight import probe_models

        env = {"TYPESAFE_API_KEY": "candidate-secret", "RDSEC_API_KEY": "judge-secret",
               "SHADOW_INCUMBENT_MODEL": "deepseek-v4.1-flash"}
        failure = {"status": "degraded", "requests": [{"status": "transport_error",
                   "error": "ReadTimeout"}], "actual_models": []}
        success = {"status": "complete", "actual_models": ["jev-1.13.0"], "requests": []}
        judge_failure = {"status": "failed", "requests": [],
                         "errors": [{"error": "JudgeError"}]}
        with patch.dict(os.environ, env, clear=True), \
             patch("shadow.incumbent.IncumbentAdapter") as incumbent, \
             patch("shadow.typesafe.TypeSafeAdapter") as candidate, \
             patch("shadow.judge.JudgeClient") as judge:
            incumbent.return_value.evaluate = AsyncMock(return_value=failure)
            candidate.return_value.evaluate = AsyncMock(return_value=success)
            judge.return_value.__enter__.return_value.adjudicate_inputs.return_value = judge_failure
            result = asyncio.run(probe_models())
            judge_config = judge.call_args.args[0]
            judge_budget = judge.call_args.kwargs["budget"]
            judge_budget.reserve(input_tokens=1000, output_tokens=judge_config.max_output_tokens)
        self.assertFalse(result["model_access_verified"])
        self.assertEqual(result["diagnostics"]["incumbent"]["requests"][0]["error"], "ReadTimeout")
        self.assertEqual(result["diagnostics"]["judge"]["errors"], judge_failure["errors"])
        self.assertEqual(result["actual_models"]["candidate"], ["jev-1.13.0"])
        self.assertNotIn("candidate-secret", json.dumps(result))
        self.assertNotIn("judge-secret", json.dumps(result))

    def test_worker_environment_excludes_repository_source_and_proxy_credentials(self):
        result = model_child_environment({"PATH": "/bin", "HOME": "/tmp/home", "RDSEC_API_KEY": "model",
            "GITHUB_TOKEN": "git", "GH_TOKEN": "git", "HTTP_PROXY": "proxy", "SCRAPECREATORS_API_KEY": "source",
            "ANTHROPIC_API_KEY": "production", "NEWS_SHADOW_CAPTURE_DIR": "production-capture"})
        self.assertEqual(result["RDSEC_API_KEY"], "model")
        for name in ("GITHUB_TOKEN", "GH_TOKEN", "HTTP_PROXY", "SCRAPECREATORS_API_KEY", "ANTHROPIC_API_KEY", "NEWS_SHADOW_CAPTURE_DIR"):
            self.assertNotIn(name, result)

    def test_output_cannot_overlap_production_or_source_bundle(self):
        for output in (ROOT, ROOT / "web/data/shadow", ROOT / "config/shadow", ROOT.parent):
            with self.assertRaises(BundleValidationError):
                validate_output_root(output)
        with tempfile.TemporaryDirectory() as temp:
            # macOS exposes the temporary directory through /var, a symlink to
            # /private/var.  The runtime correctly rejects symlink traversal,
            # so exercise it with the canonical fixture path.
            temp_root = Path(temp).resolve()
            bundle = temp_root / "bundle"
            with self.assertRaises(BundleValidationError):
                validate_output_root(bundle / "results", bundle)

    def test_reserved_holdout_cannot_be_relabelled_as_engineering(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "bundle"
            make_bundle(bundle, "2026-09-16")
            with self.assertRaises(BundleValidationError):
                inspect_run(bundle, cohort="engineering", environ={})
            with self.assertRaises(BundleValidationError):
                inspect_run(bundle, cohort="holdout", environ={})

    def test_complete_filter_run_writes_reviewable_result_and_refuses_overwrite(self):
        env = {"TYPESAFE_API_KEY": "test", "RDSEC_API_KEY": "test", "SHADOW_INCUMBENT_MODEL": "deepseek-v4.1-flash"}
        with tempfile.TemporaryDirectory() as temp:
            # macOS exposes the temporary directory through /var, a symlink to
            # /private/var.  The runtime correctly rejects symlink traversal,
            # so exercise it with the canonical fixture path.
            temp_root = Path(temp).resolve()
            bundle = temp_root / "bundle"
            records, decision = make_bundle(bundle)
            FakeAdapter.result = decision
            judged_row = {"article_id": "a1", "relevance": "relevant", "evidence_sufficiency": "sufficient",
                          "critical_story": True, "reason": "A new frontier model release.",
                          "rubric_category": "model", "evidence_ids": ["a1:snippet"],
                          "quotes": [{"evidence_id": "a1:snippet", "quote": "A new frontier model."}]}
            FakeJudge.result = {**decision, "adjudications": [judged_row]}
            identity = {"git_sha": "a" * 40, "source_sha256": "b" * 64, "files": {}}
            policy = ROOT / "config/shadow/news-relevance-v1-dev.json"
            before = (bundle / "manifest.json").read_bytes()
            output = temp_root / "out"
            now = [0.0]
            class SlowControl(FakeAdapter):
                async def evaluate(self, *args, **kwargs):
                    now[0] += 1801
                    return await super().evaluate(*args, **kwargs)

            class ReservingCandidate(FakeAdapter):
                async def evaluate(self, *args, **kwargs):
                    budget = kwargs["budget"]
                    reservation = budget.reserve(input_tokens=10, output_tokens=10)
                    budget.settle(reservation, input_tokens=10, output_tokens=10)
                    return await super().evaluate(*args, **kwargs)

            with patch.dict(os.environ, env, clear=True), \
                 patch("shadow.incumbent.IncumbentAdapter", SlowControl), \
                 patch("shadow.typesafe.TypeSafeAdapter", ReservingCandidate), \
                 patch("shadow.judge.JudgeClient", FakeJudge), \
                 patch("shadow.experiment.budget_for", side_effect=lambda policy, role:
                       RequestBudget(BudgetLimits(**policy[f"{role}_budget"]), clock=lambda: now[0])), \
                 patch("shadow.experiment.code_identity", return_value=identity):
                result = asyncio.run(run_experiment(bundle, output, policy))
                self.assertEqual(result["status"], "complete")
                self.assertTrue((Path(result["path"]) / "assessment.md").is_file())
                judge_result = json.loads((Path(result["path"]) / "judge-adjudication.json").read_text())
                self.assertIn("adjudications", judge_result)
                self.assertEqual(judge_result["adjudications"], [judged_row])
                self.assertEqual(judge_result["input_sha256"], decision["input_sha256"])
                saved = json.loads((Path(result["path"]) / "experiment.json").read_text())
                self.assertEqual(saved["metrics"]["population"]["input_count"], 1)
                self.assertTrue(saved["human_review_required"])
                with self.assertRaises(FileExistsError):
                    asyncio.run(run_experiment(bundle, output, policy))
            self.assertEqual((bundle / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
