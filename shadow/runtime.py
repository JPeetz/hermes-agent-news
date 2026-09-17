"""Explicit shadow execution settings; no production dotenv or provider lookup."""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .budget import BudgetLimits, RequestBudget
from .contracts import BundleValidationError, read_json, sha256_json, verify_bundle
from .settings import RDSEC_BASE, load_policy, safe_api_base

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "config/shadow/news-relevance-v4-dev.json"
COHORTS = ("engineering", "development", "holdout", "prospective", "recovery")


def budget_for(policy: dict, role: str) -> RequestBudget:
    value = policy.get(f"{role}_budget")
    if not isinstance(value, dict):
        raise BundleValidationError(f"Missing bounded {role} budget")
    limits = BudgetLimits(**value)
    # A policy file can lower these ceilings, never create an unbounded run.
    if (limits.max_requests > 256 or limits.max_input_tokens > 8_000_000 or
            limits.max_output_tokens > 2_000_000 or limits.deadline_seconds > 7200):
        raise BundleValidationError(f"{role} exceeds reviewed experiment ceilings")
    return RequestBudget(limits)


def explicit_route(environ=None, *, require_credentials=False) -> dict:
    env = os.environ if environ is None else environ
    base = safe_api_base(env.get("SHADOW_INCUMBENT_BASE_URL") or RDSEC_BASE,
                         allowed_hosts={"api.rdsec.trendmicro.com", "openrouter.ai"})
    if base not in {RDSEC_BASE, "https://openrouter.ai/api/v1"}:
        raise BundleValidationError("Incumbent endpoint is outside reviewed routes")
    model = env.get("SHADOW_INCUMBENT_MODEL", "").strip()
    # Do not inherit ANTHROPIC_MODEL: the judge and replay control are independent.
    keys = {
        "typesafe": bool(env.get("TYPESAFE_API_KEY")),
        "incumbent": bool(env.get("SHADOW_INCUMBENT_API_KEY") or env.get("RDSEC_API_KEY")),
        "judge": bool(env.get("RDSEC_API_KEY")),
    }
    missing = [name for name, present in keys.items() if not present]
    if not model:
        missing.append("SHADOW_INCUMBENT_MODEL")
    if require_credentials and missing:
        raise BundleValidationError("Missing shadow configuration: " + ", ".join(missing))
    return {"base_url": base, "model": model or None, "max_output_tokens": 65536,
            "max_attempts": 3, "timeout_seconds": 900, "credentials_present": keys,
            "missing_configuration": missing}


def inspect_run(bundle, policy_path=DEFAULT_POLICY, *, mode="filter", cohort="engineering",
                require_credentials=False, environ=None) -> dict:
    if mode not in {"filter", "pipeline"} or cohort not in COHORTS:
        raise BundleValidationError("Unsupported mode or cohort")
    policy = load_policy(policy_path)
    if cohort in {"holdout", "prospective"} and not policy["frozen"]:
        raise BundleValidationError("Holdout and prospective runs require a frozen policy")
    for role in ("candidate", "incumbent", "judge"):
        budget_for(policy, role)
    route = explicit_route(environ, require_credentials=require_credentials)
    cohorts = read_json(ROOT / "config/shadow/cohorts-2026-09-17.json")
    prospective_start = None
    if cohort == "prospective":
        try:
            prospective_start = date.fromisoformat(policy.get("prospective_start_date", ""))
        except (ValueError, TypeError):
            raise BundleValidationError("Prospective policy requires an explicit prospective_start_date")
        historical = [day for key in ("engineering", "development", "holdout", "recovery")
                      for day in cohorts[key]]
        if prospective_start <= date.fromisoformat(max(historical)):
            raise BundleValidationError("Prospective start must follow the declared historical cohorts")
    result = {"schema_version": "news-shadow-preflight/v1", "mode": mode, "cohort": cohort,
              "policy_version": policy["version"], "policy_frozen": policy["frozen"],
              "route": route, "model_access_verified": False,
              "source_access_verified": False, "ready": not route["missing_configuration"],
              "cohort_sha256": sha256_json(cohorts)}
    if bundle is not None:
        manifest = verify_bundle(bundle, capability=f"{mode}_replay")
        if prospective_start and date.fromisoformat(manifest["report_date"]) < prospective_start:
            raise BundleValidationError("Report precedes the frozen prospective start date")
        if manifest["report_date"] in cohorts["holdout"] and cohort != "holdout":
            raise BundleValidationError("Reserved historical holdout must use the frozen holdout policy")
        if cohort in {"development", "holdout"} and manifest["report_date"] not in cohorts[cohort]:
            raise BundleValidationError("Bundle date is outside the predeclared cohort")
        if cohort != "recovery":
            eligibility = manifest.get("eligibility", {})
            if (eligibility.get("healthy") is not True or
                    eligibility.get("successful_run") is not True or
                    not (manifest.get("publication", {}).get("status") in {"published", "superseded"}
                         and (manifest.get("publication", {}).get("original_publication_verified") is True
                              or manifest.get("publication", {}).get("status") == "published"))):
                raise BundleValidationError("Default cohorts require a healthy successful published source run")
        original = read_json(Path(bundle) / "relevance/incumbent-decision.json")
        original_model = original.get("model")
        if not isinstance(original_model, str) or not original_model:
            raise BundleValidationError("Captured incumbent model identity is unavailable")
        if route["model"] and original_model and route["model"].rsplit("/", 1)[-1].lower() != original_model.rsplit("/", 1)[-1].lower():
            raise BundleValidationError("Control model must match the captured incumbent model")
        result.update(manifest=manifest, original_model=original_model,
                      bundle_sha256=manifest["bundle_sha256"], integrity_verified=True)
    return result


def validate_output_root(path, bundle=None) -> Path:
    raw = Path(path).absolute()
    if any(parent.is_symlink() for parent in (raw, *raw.parents)):
        raise BundleValidationError("Output path may not traverse symlinks")
    target = raw.resolve()
    forbidden = [ROOT / "web", ROOT / "config", ROOT / ".git"]
    if bundle is not None:
        forbidden.append(Path(bundle).resolve())
    if target == ROOT or ROOT.is_relative_to(target):
        raise BundleValidationError("Output path must be a dedicated experiment directory")
    for other in forbidden:
        if target == other or target.is_relative_to(other) or other.is_relative_to(target):
            raise BundleValidationError("Output path overlaps a protected directory")
    return target


def model_child_environment(environ=None) -> dict[str, str]:
    """Only runtime and explicitly named model credentials enter replay workers."""
    source = os.environ if environ is None else environ
    permitted = {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR",
                 "TYPESAFE_API_KEY", "RDSEC_API_KEY", "SHADOW_INCUMBENT_API_KEY",
                 "SHADOW_INCUMBENT_MODEL", "SHADOW_INCUMBENT_BASE_URL"}
    result = {key: source[key] for key in permitted if key in source}
    result.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York",
                  NEWS_SHADOW_SKIP_DOTENV="1", LLM_TRUST_ENV_PROXY="false")
    return result
