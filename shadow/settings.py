"""Versioned experiment policy and credential-free execution identity."""
from __future__ import annotations

import math
import re
import subprocess
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import BundleValidationError, hash_file, read_json, sha256_json

POLICY_SCHEMA = "news-shadow-policy/v1"
JUDGE_MODEL = "deepseek-v4.1-flash"
JUDGE_VERSION = "news-editorial-judge/v3"
RDSEC_BASE = "https://api.rdsec.trendmicro.com/prod/aiendpoint/v1"


def load_policy(path: str | Path) -> dict:
    policy = read_json(path)
    if not isinstance(policy, dict) or policy.get("schema_version") != POLICY_SCHEMA:
        raise BundleValidationError("Unsupported shadow policy")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", policy.get("version", "")):
        raise BundleValidationError("Invalid policy version")
    if policy.get("model") != "jev-1.13.0":
        raise BundleValidationError("Policy must pin the reviewed Jev revision")
    thresholds = [policy.get(key) for key in ("reject_max", "keep_min", "sufficiency_min")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds):
        raise BundleValidationError("Policy thresholds must be finite probabilities")
    if thresholds[0] >= thresholds[1]:
        raise BundleValidationError("Reject threshold must be below keep threshold")
    if type(policy.get("frozen")) is not bool:
        raise BundleValidationError("Policy frozen status must be explicit")
    for key, maximum in (("chunk_size", 64), ("concurrency", 4), ("max_attempts", 3)):
        value = policy.get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise BundleValidationError(f"Invalid policy {key}")
    timeout = policy.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 120:
        raise BundleValidationError("Invalid candidate timeout")
    if policy.get("fallback") != "retain":
        raise BundleValidationError("Only explicit retained-superset fallback is qualified")
    return policy


def safe_api_base(value: str, *, allowed_hosts: set[str]) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or parsed.hostname not in allowed_hosts or
            parsed.username or parsed.password or parsed.query or parsed.fragment or
            parsed.port not in (None, 443)):
        raise BundleValidationError("Unapproved or credential-bearing model endpoint")
    return value.rstrip("/")


def code_identity(root: str | Path) -> dict:
    root = Path(root)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True)
    source_files = {}
    for folder in ("shadow", "agents", "generators", "scripts/shadow"):
        for path in sorted((root / folder).rglob("*.py")):
            if path.is_symlink():
                raise BundleValidationError("Executable source may not be symlinked")
            source_files[path.relative_to(root).as_posix()] = hash_file(path)
    for filename in ("requirements.txt", "run_pipeline.py"):
        source_files[filename] = hash_file(root / filename)
    dependencies = {}
    for package in ("httpx", "requests", "aiohttp", "anthropic", "pydantic", "PyYAML", "python-dotenv"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    runtime = {"python": platform.python_version(), "dependencies": dependencies}
    return {"git_sha": result.stdout.strip(), "source_sha256": sha256_json({"files": source_files, "runtime": runtime}),
            "files": source_files, "runtime": runtime}


def experiment_identity(manifest: dict, code: dict, policy: dict, *, mode: str,
                        judge_enabled: bool = True, repeat_control: bool = False,
                        runtime: dict | None = None) -> tuple[str, dict]:
    identity = {
        "schema_version": "news-shadow-experiment/v1",
        "source": manifest.get("source"), "bundle_sha256": manifest["bundle_sha256"],
        "replay_code": {key: code[key] for key in ("git_sha", "source_sha256")},
        "mode": mode, "policy_sha256": sha256_json(policy),
        "candidate_model": policy["model"], "judge_model": JUDGE_MODEL if judge_enabled else None,
        "judge_version": JUDGE_VERSION if judge_enabled else None,
        "repeat_control": repeat_control, "runtime": runtime or {},
    }
    return sha256_json(identity), identity
