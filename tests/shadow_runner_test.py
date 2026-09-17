"""Offline integration checks for immutable experiments and protected cohorts."""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shadow.contracts import BundleValidationError, seal_bundle, sha256_json, write_json
from shadow.experiment import run_experiment
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
        result = {**self.result, "status": "complete", "adjudications": [], "requests": [], "errors": []}
        write_json(output_path, result)
        return result


class ShadowRunnerTest(unittest.TestCase):
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
            bundle = Path(temp) / "bundle"
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
            bundle = Path(temp) / "bundle"
            records, decision = make_bundle(bundle)
            FakeAdapter.result = decision
            FakeJudge.result = decision
            identity = {"git_sha": "a" * 40, "source_sha256": "b" * 64, "files": {}}
            policy = ROOT / "config/shadow/news-relevance-v1-dev.json"
            before = (bundle / "manifest.json").read_bytes()
            with patch.dict(os.environ, env, clear=True), \
                 patch("shadow.incumbent.IncumbentAdapter", FakeAdapter), \
                 patch("shadow.typesafe.TypeSafeAdapter", FakeAdapter), \
                 patch("shadow.judge.JudgeClient", FakeJudge), \
                 patch("shadow.experiment.code_identity", return_value=identity):
                result = asyncio.run(run_experiment(bundle, Path(temp) / "out", policy))
                self.assertEqual(result["status"], "complete")
                self.assertTrue((Path(result["path"]) / "assessment.md").is_file())
                saved = json.loads((Path(result["path"]) / "experiment.json").read_text())
                self.assertEqual(saved["metrics"]["population"]["input_count"], 1)
                self.assertTrue(saved["human_review_required"])
                with self.assertRaises(FileExistsError):
                    asyncio.run(run_experiment(bundle, Path(temp) / "out", policy))
            self.assertEqual((bundle / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
