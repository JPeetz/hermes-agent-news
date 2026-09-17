"""Offline contract tests for the dedicated shadow judge.

These tests intentionally use a fake transport and never contact RDSec.  The
repository owner runs them explicitly; they are included here as the focused
mocked coverage requested by the shadow plan.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from h11 import LocalProtocolError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shadow.contracts import validate_input, validate_decisions  # noqa: E402
from shadow.judge import (  # noqa: E402
    CATEGORY_VALUES,
    DEEPSEEK_MODEL,
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DIMENSIONS,
    DIMENSION_VALUES,
    FINDING_KINDS,
    RDSEC_ENDPOINT,
    RELEVANCE_VALUES,
    SEVERITIES,
    SUFFICIENCY_VALUES,
    JudgeClient,
    JudgeConfig,
    JudgeError,
    OVERALL_VALUES,
    TransportResponse,
    build_blinded_output_prompt,
    build_input_artifact,
    validate_input_adjudication,
    validate_output_comparison,
)
from shadow.metrics import compare_decisions, summarize_requests  # noqa: E402
from shadow.report import build_assessment, render_assessment  # noqa: E402


RECORDS = [
    {
        "id": "a1",
        "title": "A useful model release",
        "source": "Example source",
        "snippet": "A new AI model is available.",
    },
    {
        "id": "a2",
        "title": "A cooking story",
        "source": "Example source",
        "snippet": "A recipe with no model evidence.",
    },
]


def _input_payload(records=RECORDS, *, model=DEEPSEEK_MODEL, finish="stop", usage=None):
    rows = []
    for record in records:
        article_id = record["id"]
        rows.append(
            {
                "article_id": article_id,
                "evidence_ids": [f"{article_id}:snippet"],
                "relevance": "relevant" if article_id == "a1" else "irrelevant",
                "evidence_sufficiency": "sufficient",
                "rubric_category": "model" if article_id == "a1" else "none",
                "critical_story": article_id == "a1",
                "reason": "The bounded snippet supports this classification.",
                "quotes": [
                    {"evidence_id": f"{article_id}:snippet", "quote": record["snippet"]}
                ],
            }
        )
    return {
        "id": "mock-request",
        "model": model,
        "choices": [
            {
                "finish_reason": finish,
                "message": {"role": "assistant", "content": json.dumps({"adjudications": rows})},
            }
        ],
        "usage": usage or {"prompt_tokens": 20, "completion_tokens": 80, "total_tokens": 100},
    }


class RecordingBudget:
    def __init__(self):
        self.reservations = []
        self.settlements = []

    def reserve(self, **kwargs):
        reservation = len(self.reservations) + 1
        self.reservations.append((reservation, kwargs))
        return reservation

    def settle(self, reservation, **kwargs):
        self.settlements.append((reservation, kwargs))


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class BlockingTransport(FakeTransport):
    def post(self, url, **kwargs):
        time.sleep(0.04)
        return super().post(url, **kwargs)


class JudgeContractTests(unittest.TestCase):
    def test_config_is_pinned_and_secret_safe(self):
        config = JudgeConfig(api_key="secret")
        self.assertEqual(config.endpoint, RDSEC_ENDPOINT)
        self.assertEqual(config.model, DEEPSEEK_MODEL)
        self.assertEqual(config.max_output_tokens, DEEPSEEK_MAX_OUTPUT_TOKENS)
        self.assertEqual(config.to_public_dict()["max_output_tokens"], 384_000)
        self.assertEqual(config.timeout_seconds, 900.0)
        self.assertNotIn("api_key", config.to_public_dict())
        with self.assertRaises(ValueError):
            JudgeConfig(endpoint="https://other.example/v1/chat/completions")
        with self.assertRaises(ValueError):
            JudgeConfig(model="some-other-model")

    def test_input_artifact_matches_shared_contract_and_excludes_labels(self):
        artifact = build_input_artifact(
            [
                {**RECORDS[0], "decision": "keep", "branch": "candidate", "confidence": 0.9},
                RECORDS[1],
            ]
        )
        validate_input(artifact)
        payload = json.loads(artifact["user_message"])
        self.assertEqual(set(payload), {"schema", "articles"})
        encoded = json.dumps(payload)
        self.assertNotIn('"decision"', encoded)
        self.assertNotIn('"branch"', encoded)
        self.assertNotIn('"confidence"', encoded)

    def test_prompts_declare_the_complete_typed_output_contract(self):
        input_prompt = build_input_artifact(RECORDS)["system_prompt"]
        for field in (
            '"article_id"',
            '"evidence_ids"',
            '"relevance"',
            '"evidence_sufficiency"',
            '"rubric_category"',
            '"critical_story"',
            '"reason"',
            '"quotes"',
            'relevant|irrelevant|insufficient_evidence',
        ):
            self.assertIn(field, input_prompt)
        self.assertIn("never the pipe character", input_prompt)

        decoder = json.JSONDecoder()
        input_example, _ = decoder.raw_decode(input_prompt[input_prompt.index("{"):])
        input_row = input_example["adjudications"][0]
        self.assertEqual(set(input_example), {"adjudications"})
        self.assertEqual(set(input_row), {
            "article_id", "evidence_ids", "relevance", "evidence_sufficiency",
            "rubric_category", "critical_story", "reason", "quotes",
        })
        self.assertEqual(set(input_row["relevance"].split("|")), RELEVANCE_VALUES)
        self.assertEqual(set(input_row["evidence_sufficiency"].split("|")), SUFFICIENCY_VALUES)
        self.assertEqual(set(input_row["rubric_category"].split("|")), CATEGORY_VALUES)
        input_example["adjudications"] = [{
            "article_id": "a1",
            "evidence_ids": ["a1:snippet"],
            "relevance": "relevant",
            "evidence_sufficiency": "sufficient",
            "rubric_category": "model",
            "critical_story": True,
            "reason": "The supplied snippet supports the classification.",
            "quotes": [{"evidence_id": "a1:snippet", "quote": RECORDS[0]["snippet"]}],
        }]
        self.assertEqual(validate_input_adjudication(input_example, RECORDS[:1])[0]["article_id"], "a1")

        output_prompt, _ = build_blinded_output_prompt(
            {"summary": "A"},
            {"summary": "B"},
            pair_id="p1",
            seed=1,
        )
        for field in (
            '"pair_id"',
            '"dimensions"',
            '"important_story_coverage"',
            '"irrelevant_inclusions"',
            '"safety_policy_omissions"',
            '"supported_claims"',
            '"duplicates"',
            '"ranking_usefulness"',
            '"summary_quality"',
            '"findings"',
            '"overall"',
            'gained_story|lost_story|unsupported_claim|duplicate|other',
        ):
            self.assertIn(field, output_prompt["system_prompt"])
        self.assertIn("never the pipe character", output_prompt["system_prompt"])
        output_example, _ = decoder.raw_decode(
            output_prompt["system_prompt"][output_prompt["system_prompt"].index("{"):]
        )
        self.assertEqual(set(output_example), {"pair_id", "dimensions", "findings", "overall"})
        self.assertEqual(set(output_example["dimensions"]), set(DIMENSIONS))
        self.assertEqual(
            set(output_example["dimensions"]["important_story_coverage"].split("|")),
            DIMENSION_VALUES,
        )
        finding_example = output_example["findings"][0]
        self.assertEqual(set(finding_example), {
            "kind", "side", "severity", "article_id", "evidence_ids", "quote", "reason",
        })
        self.assertEqual(set(finding_example["kind"].split("|")), FINDING_KINDS)
        self.assertEqual(set(finding_example["severity"].split("|")), SEVERITIES)
        self.assertEqual(set(finding_example["side"].split("|")), {"A", "B"})
        self.assertEqual(set(output_example["overall"].split("|")), OVERALL_VALUES)
        output_example.update(
            pair_id="p1",
            dimensions={key: "tie" for key in DIMENSIONS},
            findings=[],
            overall="tie",
        )
        self.assertEqual(
            validate_output_comparison(output_example, pair_id="p1", evidence={})["overall"],
            "tie",
        )

    def test_all_records_are_adjudicated_and_decisions_validate(self):
        fake = FakeTransport([TransportResponse(200, _input_payload())])
        budget = RecordingBudget()
        with JudgeClient(JudgeConfig(batch_size=16), transport=fake, budget=budget) as client:
            result = client.adjudicate_inputs(RECORDS)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([row["id"] for row in result["decisions"]], ["a1", "a2"])
        validate_decisions(RECORDS, result["decisions"])
        self.assertEqual(len(budget.reservations), 1)
        self.assertEqual(len(budget.settlements), 1)
        self.assertEqual(fake.calls[0][0], RDSEC_ENDPOINT)
        self.assertEqual(fake.calls[0][1]["json"]["model"], DEEPSEEK_MODEL)
        self.assertEqual(
            fake.calls[0][1]["json"]["cache"],
            {"no-cache": True, "no-store": True},
        )

    def test_transport_retry_is_bounded_and_each_attempt_is_budgeted(self):
        fake = FakeTransport(
            [
                TransportResponse(429, {"error": {"message": "busy"}}, text="busy"),
                TransportResponse(200, _input_payload()),
            ]
        )
        budget = RecordingBudget()
        config = JudgeConfig(max_attempts=3)
        with JudgeClient(config, transport=fake, budget=budget, sleep=lambda _: None) as client:
            result = client.adjudicate_inputs(RECORDS)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(budget.reservations), 2)
        self.assertEqual(len(budget.settlements), 2)

    def test_end_turn_is_a_terminal_judge_response(self):
        fake = FakeTransport([TransportResponse(200, _input_payload(finish="end_turn"))])
        with JudgeClient(JudgeConfig(max_attempts=1), transport=fake) as client:
            result = client.adjudicate_inputs(RECORDS)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["requests"][0]["finish_reason"], "end_turn")

    def test_judge_progress_emits_heartbeat_while_request_is_blocked(self):
        stderr = io.StringIO()
        fake = BlockingTransport([TransportResponse(200, _input_payload())])
        with JudgeClient(
            JudgeConfig(max_attempts=1),
            transport=fake,
            progress_interval_seconds=0.01,
            progress_stream=stderr,
        ) as client:
            result = client.adjudicate_inputs(RECORDS)
        events = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(events[0]["event"], "start")
        self.assertTrue(any(event["event"] == "heartbeat" for event in events))
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["status"], "success")
        self.assertEqual(events[-1]["input_tokens"], 20)
        self.assertEqual(events[-1]["output_tokens"], 80)

    def test_finish_length_fails_without_identical_retries_or_substitution(self):
        bad = _input_payload(finish="length")
        fake = FakeTransport([TransportResponse(200, bad)] * 3)
        budget = RecordingBudget()
        stderr = io.StringIO()
        with JudgeClient(
            JudgeConfig(max_attempts=3),
            transport=fake,
            budget=budget,
            sleep=lambda _: None,
            progress_stream=stderr,
        ) as client:
            result = client.adjudicate_inputs(RECORDS)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([row["decision"] for row in result["decisions"]], ["abstain", "abstain"])
        self.assertTrue(all(row["fallback_reason"] == "judge_error" for row in result["decisions"]))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(budget.reservations), 1)
        self.assertEqual(result["requests"][0]["finish_reason"], "length")
        events = [json.loads(line) for line in stderr.getvalue().splitlines()]
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(events[-1]["input_tokens"], 20)
        self.assertEqual(events[-1]["output_tokens"], 80)

    def test_model_identity_and_usage_are_validated(self):
        fake = FakeTransport([TransportResponse(200, _input_payload(model="other"))] * 3)
        with JudgeClient(JudgeConfig(max_attempts=3), transport=fake, sleep=lambda _: None) as client:
            result = client.adjudicate_inputs(RECORDS)
        self.assertEqual(result["status"], "failed")
        self.assertIn("returned model", result["errors"][0]["error"])

    def test_transport_exception_does_not_echo_credentials_in_artifacts(self):
        credential = "synthetic-judge-secret"
        transport_error = LocalProtocolError(
            f"Illegal header value b'Bearer {credential}\\n'"
        )
        fake = FakeTransport([transport_error] * 3)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "judge-artifact.json"
            with JudgeClient(
                JudgeConfig(api_key=credential, max_attempts=3),
                transport=fake,
                sleep=lambda _: None,
            ) as client:
                result = client.adjudicate_inputs(RECORDS, output_path=output_path)
            persisted = output_path.read_text(encoding="utf-8")

        artifacts = json.dumps(
            {
                "result": result,
                "requests": result["requests"],
                "errors": result["errors"],
                "persisted": persisted,
            }
        )
        self.assertNotIn(credential, artifacts)
        self.assertTrue(result["errors"])
        self.assertTrue(
            all(request["error"] == "LocalProtocolError" for request in result["requests"])
        )


class BlindingAndValidationTests(unittest.TestCase):
    def test_blinding_is_deterministic_and_mapping_is_outside_prompt(self):
        prompt_one, mapping_one = build_blinded_output_prompt(
            {"branch": "control", "summary": "control text"},
            {"branch": "candidate", "summary": "candidate text"},
            pair_id="p1",
            seed=17,
            evidence={"e1": "A supported claim."},
            reversal_probability=0.5,
        )
        prompt_two, mapping_two = build_blinded_output_prompt(
            {"branch": "control", "summary": "control text"},
            {"branch": "candidate", "summary": "candidate text"},
            pair_id="p1",
            seed=17,
            evidence={"e1": "A supported claim."},
            reversal_probability=0.5,
        )
        self.assertEqual(mapping_one, mapping_two)
        self.assertEqual(prompt_one, prompt_two)
        encoded = prompt_one["user_message"]
        self.assertNotIn('"branch"', encoded)
        self.assertNotIn("control_label", encoded)
        self.assertNotIn("candidate_label", encoded)

    def test_comparison_rejects_fabricated_quote_and_unknown_reference(self):
        valid = {
            "pair_id": "p1",
            "dimensions": {key: "tie" for key in (
                "important_story_coverage", "irrelevant_inclusions", "safety_policy_omissions",
                "supported_claims", "duplicates", "ranking_usefulness", "summary_quality",
            )},
            "findings": [
                {
                    "kind": "gained_story",
                    "side": "A",
                    "severity": "major",
                    "article_id": "a1",
                    "evidence_ids": ["e1"],
                    "quote": "A supported claim.",
                    "reason": "The supplied evidence supports the finding.",
                }
            ],
            "overall": "tie",
        }
        self.assertEqual(validate_output_comparison(valid, pair_id="p1", evidence={"e1": "A supported claim."})["overall"], "tie")
        fabricated = dict(valid)
        fabricated["findings"] = [dict(valid["findings"][0], quote="not in evidence")]
        with self.assertRaises(JudgeError):
            validate_output_comparison(fabricated, pair_id="p1", evidence={"e1": "A supported claim."})
        unknown = dict(valid)
        unknown["findings"] = [dict(valid["findings"][0], evidence_ids=["missing"])]
        with self.assertRaises(JudgeError):
            validate_output_comparison(unknown, pair_id="p1", evidence={"e1": "A supported claim."})


class MetricsAndReportTests(unittest.TestCase):
    def test_metrics_keep_set_differences_and_null_denominators(self):
        original = [{"id": "a1", "decision": "keep", "effective_keep": True}]
        control = [{"id": "a1", "decision": "keep", "effective_keep": True}]
        candidate = []
        metrics = compare_decisions(RECORDS, original, control, candidate)
        self.assertEqual(metrics["set_differences"]["candidate_vs_control"]["removed_ids"], ["a1"])
        self.assertIsNone(metrics["branches"]["candidate"]["coverage_rate"])
        self.assertIsNone(metrics["branches"]["candidate"]["abstention_rate"])
        self.assertIsNone(metrics["usage"]["judge"]["cost_usd_known"])

    def test_unknown_cost_is_not_zero(self):
        summary = summarize_requests([{"status": "success", "usage_known": True, "input_tokens": 10, "output_tokens": 5, "cost_known": False, "cost_usd": None}])
        self.assertEqual(summary["cost_status"], "unknown")
        self.assertIsNone(summary["cost_usd_known"])
        self.assertEqual(summary["unknown_cost_request_count"], 1)

    def test_report_calls_judge_quality_an_estimate_and_requires_review(self):
        metrics = compare_decisions(
            RECORDS,
            [{"id": "a1", "decision": "keep", "effective_keep": True}, {"id": "a2", "decision": "reject", "effective_keep": False}],
            [{"id": "a1", "decision": "keep", "effective_keep": True}, {"id": "a2", "decision": "reject", "effective_keep": False}],
            [{"id": "a1", "decision": "reject", "effective_keep": False}, {"id": "a2", "decision": "reject", "effective_keep": False}],
            judge_adjudications=[
                {"article_id": "a1", "relevance": "relevant", "critical_story": True},
                {"article_id": "a2", "relevance": "irrelevant", "critical_story": False},
            ],
        )
        report = build_assessment({"experiment_id": "x", "status": "complete", "metrics": metrics})
        markdown = render_assessment(report)
        self.assertIn("judge-estimated, not ground truth", markdown)
        self.assertEqual(report["gate"], "requires_human_review")
        self.assertTrue(report["human_review_required"])


if __name__ == "__main__":
    unittest.main()
