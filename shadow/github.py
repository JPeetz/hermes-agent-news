"""Read-only GitHub acquisition through the owner's existing ``gh`` identity.

Only the production repository and Actions/commit/content GET endpoints are used.
Downloaded source is data: no checkout, import, archive extraction or execution.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
from datetime import date, datetime, timedelta, timezone

SOURCE_REPOSITORY = "flyryan/ai-news-aggregator"
WORKFLOW_PATH = ".github/workflows/daily-pipeline.yml"
MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class AcquisitionError(ValueError):
    pass


def validate_repository(repository: str) -> str:
    if repository != SOURCE_REPOSITORY:
        raise AcquisitionError("Only the configured production source repository is accepted")
    return repository


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
        raise AcquisitionError(f"{name} must be a positive integer")
    return int(value)


def parse_date(value: str) -> date:
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise AcquisitionError("Dates must use YYYY-MM-DD")
    return parsed


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


class GitHubSource:
    def __init__(self, repository: str = SOURCE_REPOSITORY):
        self.repository = validate_repository(repository)

    def _gh(self, args: list[str], limit: int = MAX_DOWNLOAD_BYTES) -> bytes:
        # Never include gh stderr in errors: authentication/provider diagnostics
        # can contain sensitive connection details. No shell interpolation.
        result = subprocess.run(["gh", *args], capture_output=True, timeout=180, check=False)
        if result.returncode:
            raise AcquisitionError("Read-only GitHub request failed; check existing gh authentication/access")
        if len(result.stdout) > limit:
            raise AcquisitionError("GitHub response exceeds the acquisition size limit")
        return result.stdout

    def api(self, suffix: str, *, raw: bool = False):
        payload = self._gh(["api", "--method", "GET", f"repos/{self.repository}/{suffix}"])
        return payload if raw else json.loads(payload)

    def run(self, run_id: int, attempt: int | None = None) -> dict:
        run_id = positive_int(run_id, "run_id")
        suffix = f"actions/runs/{run_id}"
        if attempt is not None:
            suffix += f"/attempts/{positive_int(attempt, 'attempt')}"
        run = self.api(suffix)
        if run.get("path", "").split("@")[0] != WORKFLOW_PATH:
            raise AcquisitionError("Run is not the production Daily Pipeline workflow")
        if run.get("repository", {}).get("full_name") != self.repository:
            raise AcquisitionError("Run repository identity mismatch")
        if run.get("id") != run_id or (attempt is not None and run.get("run_attempt") != attempt):
            raise AcquisitionError("Run/attempt identity mismatch")
        return run

    def artifacts(self, run_id: int) -> list[dict]:
        rows = []
        for page in range(1, 1001):
            batch = self.api(f"actions/runs/{positive_int(run_id, 'run_id')}/artifacts?per_page=100&page={page}")
            rows.extend(batch.get("artifacts", []))
            if len(rows) >= batch.get("total_count", 0):
                return rows
        raise AcquisitionError("Artifact pagination limit exceeded")

    def attempt_artifacts(self, run: dict) -> list[dict]:
        """Artifact listing is run-scoped; bind its timestamp to attempt bounds."""
        run_id, attempt = run["id"], run["run_attempt"]
        latest = self.run(run_id)
        start = _timestamp(run["run_started_at"])
        # updated_at can reflect later API changes; next attempt start is the
        # authoritative upper boundary when a later attempt exists.
        end = (_timestamp(self.run(run_id, attempt + 1)["run_started_at"])
               if latest["run_attempt"] > attempt else _timestamp(run["updated_at"]) + timedelta(seconds=1))
        return [row for row in self.artifacts(run_id)
                if start <= _timestamp(row["created_at"]) < end]

    def diagnostics(self, run: dict) -> tuple[bytes, dict]:
        matches = [r for r in self.attempt_artifacts(run) if r.get("name") == "pipeline-diagnostics"]
        if len(matches) != 1:
            raise AcquisitionError("Missing or ambiguous diagnostics artifact for this attempt")
        artifact = matches[0]
        if artifact.get("expired"):
            raise AcquisitionError("Diagnostics artifact has expired")
        if artifact.get("size_in_bytes", MAX_DOWNLOAD_BYTES + 1) > MAX_DOWNLOAD_BYTES:
            raise AcquisitionError("Diagnostics artifact exceeds acquisition size limit")
        payload = self.api(f"actions/artifacts/{positive_int(artifact['id'], 'artifact id')}/zip", raw=True)
        digest = artifact.get("digest")
        if digest and digest != "sha256:" + hashlib.sha256(payload).hexdigest():
            raise AcquisitionError("Downloaded artifact digest mismatch")
        return payload, artifact

    def logs(self, run_id: int, attempt: int) -> str:
        raw = self._gh(["run", "view", str(positive_int(run_id, "run_id")),
                        "--repo", self.repository, "--attempt", str(positive_int(attempt, "attempt")),
                        "--log"]).decode("utf-8", errors="replace")
        # gh can label all historical lines UNKNOWN STEP. Bind those lines to
        # the API's job/step time windows before accepting checkout/push proof.
        jobs = self.api(f"actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100").get("jobs", [])
        windows = []
        for job in jobs:
            for step in job.get("steps", []):
                if step.get("name") in ("Check out flyryan main", "Push generated commit") and step.get("started_at") and step.get("completed_at"):
                    windows.append((job["name"], step["name"], _timestamp(step["started_at"]),
                                    _timestamp(step["completed_at"]) + timedelta(seconds=1)))
        lines = []
        for line in raw.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3 and parts[1] == "UNKNOWN STEP":
                stamp = parts[2].split(" ", 1)[0]
                try:
                    observed = _timestamp(stamp)
                except ValueError:
                    continue
                names = [name for job, name, start, end in windows
                         if job == parts[0] and start <= observed < end]
                if len(names) == 1:
                    parts[1] = names[0]
                    line = "\t".join(parts)
            lines.append(line)
        return "\n".join(lines)

    def resolve_commit(self, ref: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{7,40}|main", ref):
            raise AcquisitionError("Invalid commit reference")
        sha = self.api(f"commits/{ref}")["sha"]
        if not SHA_RE.fullmatch(sha):
            raise AcquisitionError("Invalid resolved commit SHA")
        return sha

    def content(self, sha: str, path: str) -> bytes:
        if not SHA_RE.fullmatch(sha) or not re.fullmatch(r"web/data/\d{4}-\d{2}-\d{2}/(summary|news|research|social|reddit)\.json", path):
            raise AcquisitionError("Output lookup outside allowlisted report files")
        row = self.api(f"contents/{path}?ref={sha}")
        if row.get("type") != "file" or row.get("size", MAX_DOWNLOAD_BYTES + 1) > MAX_DOWNLOAD_BYTES:
            raise AcquisitionError("Output is not a bounded GitHub content file")
        blob_sha = row.get("sha", "")
        if not SHA_RE.fullmatch(blob_sha):
            raise AcquisitionError("Invalid Git blob SHA")
        if row.get("encoding") == "none":
            row = self.api(f"git/blobs/{blob_sha}")
        if row.get("encoding") != "base64" or row.get("sha") != blob_sha:
            raise AcquisitionError("Unsupported GitHub blob encoding")
        payload = base64.b64decode(row["content"], validate=False)
        expected = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if expected != row.get("sha"):
            raise AcquisitionError("Git blob hash mismatch")
        return payload

    def output_files(self, sha: str, report_date: str) -> dict[str, bytes]:
        parse_date(report_date)
        return {f"{name}.json": self.content(sha, f"web/data/{report_date}/{name}.json")
                for name in ("summary", "news", "research", "social", "reddit")}

    def publication(self, output_sha: str, report_date: str) -> tuple[dict, dict[str, bytes]]:
        sha = self.resolve_commit(output_sha)
        files = self.output_files(sha, report_date)
        current_sha = self.resolve_commit("main")
        current = self.output_files(current_sha, report_date)
        matches = all(files[name] == current[name] for name in files)
        status, revert_sha = ("published" if matches else "superseded"), None
        if not matches:
            # Explicit revert evidence wins; a different current file alone
            # proves replacement only, never a revert. Paginate the comparison
            # because retention-window imports can be hundreds of commits old.
            for page in range(1, 1001):
                comparison = self.api(f"compare/{sha}...{current_sha}?per_page=100&page={page}")
                commits = comparison.get("commits", [])
                for commit in commits:
                    message = commit.get("commit", {}).get("message", "")
                    if re.search(rf"This reverts commit {re.escape(sha)}\b", message):
                        status, revert_sha = "reverted", commit["sha"]
                if len(commits) < 100 or page * 100 >= comparison.get("total_commits", 0):
                    break
            else:
                raise AcquisitionError("Publication reconciliation exceeds commit pagination limit")
        return {"status": status, "output_commit": sha, "current_commit": current_sha,
                "revert_commit": revert_sha, "verification": "github_git_blob_hash",
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}, files

    def inventory(self, from_date: str, to_date: str) -> list[dict]:
        first, last = parse_date(from_date), parse_date(to_date)
        if last < first or (last - first).days > 89:
            raise AcquisitionError("Inventory requires an ordered range of at most 90 dates")
        # Creation dates are discovery windows, not report-date claims. Include
        # a one-day overlap; callers must inspect the artifact's actual date.
        query = f"{(first - timedelta(days=1)).isoformat()}..{(last + timedelta(days=1)).isoformat()}"
        rows = []
        for page in range(1, 1001):
            result = self.api(f"actions/workflows/daily-pipeline.yml/runs?per_page=100&page={page}&created={query}")
            runs = result.get("workflow_runs", [])
            for run in runs:
                for attempt in range(1, run.get("run_attempt", 1) + 1):
                    row = self.run(run["id"], attempt)
                    artifacts = self.attempt_artifacts(row) if row.get("status") == "completed" else []
                    rows.append({"id": row["id"], "attempt": attempt,
                                 "created_at": row.get("created_at"), "started_at": row.get("run_started_at"),
                                 "event_sha": row.get("head_sha"), "status": row.get("status"),
                                 "conclusion": row.get("conclusion"), "event": row.get("event"),
                                 "artifacts": [{k: a.get(k) for k in ("id", "name", "size_in_bytes", "expired", "digest", "created_at", "expires_at")} for a in artifacts],
                                 "classification": "pending" if row.get("status") != "completed" else
                                 ("diagnostics_available" if any(a["name"] == "pipeline-diagnostics" and not a.get("expired") for a in artifacts) else "no_usable_diagnostics"),
                                 "report_date": None, "filter_replay": None, "publication_status": "unverified"})
            if len(runs) < 100:
                return rows
        raise AcquisitionError("Run pagination limit exceeded")


def lineage_from_logs(logs: str) -> dict:
    """Read only known checkout/push step output, never article/LLM log prose."""
    execution, pushed = [], []
    waiting_for_sha = False
    for line in logs.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        _, step, body = parts
        body = re.sub(r"^\d{4}-\d\d-\d\dT\S+\s+", "", body).strip()
        if step == "Check out flyryan main":
            if waiting_for_sha and SHA_RE.fullmatch(body.strip("'\"")):
                execution.append(body.strip("'\""))
                waiting_for_sha = False
            if re.search(r"\bgit log -1 --format=%H\b", body):
                waiting_for_sha = True
        if step == "Push generated commit":
            match = re.search(r"\b[0-9a-f]{7,40}\.\.([0-9a-f]{7,40})\s+HEAD -> main\b", body)
            if match:
                pushed.append(match.group(1))
    if len(set(execution)) > 1 or len(set(pushed)) > 1:
        raise AcquisitionError("Ambiguous checkout or publication lineage")
    return {"execution_sha": execution[-1] if execution else None,
            "output_commit": pushed[-1] if pushed else None,
            "evidence": "trusted_workflow_step_logs", "logs_sha256": hashlib.sha256(logs.encode()).hexdigest()}
