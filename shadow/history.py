"""Offline reconstruction of bounded news-filter evidence from legacy diagnostics.

Archives remain inert. Missing/ambiguous evidence produces an unavailable filter
capability, never an inferred rejection. Publication and collection health remain
separate qualifications from a technically replayable filter.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

from .contracts import canonical_bytes, sha256_json, seal_bundle, write_json, hash_file
from .github import SOURCE_REPOSITORY, WORKFLOW_PATH, SHA_RE, positive_int, parse_date, validate_repository
from .rendering import (keyword_filter, relevance_records, render_filter_input, KEYWORD_SHA256,
                        RENDERER_SHA256, FILTER_PROMPT, filter_system_prompt, build_fenced_user_message)

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 768 * 1024 * 1024
CATEGORIES = ("news", "research", "social", "reddit")
ALLOWED_MEMBER = re.compile(
    r"(?:data/)?(?:checkpoints/\d{4}-\d{2}-\d{2}/(?:gathering|analysis|topics|summary|hero)\.json|"
    r"processed/(?:orchestrator_result|cost_report)_\d{4}-\d{2}-\d{2}\.json|llm_metrics\.jsonl)$")


class HistoricalEvidenceError(ValueError):
    pass


def _read_diagnostics_archive(payload: bytes) -> dict[str, bytes]:
    """Read allowlisted regular members without extracting any archive paths."""
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise HistoricalEvidenceError("Diagnostics archive exceeds size limit")
    accepted, names, total = {}, set(), 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) > 10000:
            raise HistoricalEvidenceError("Too many archive members")
        for entry in archive.infolist():
            name = entry.filename
            path = PurePosixPath(name)
            if "\\" in name or path.is_absolute() or ".." in path.parts or "\x00" in name:
                raise HistoricalEvidenceError("Unsafe archive member path")
            normalized = str(path)
            if normalized in names:
                raise HistoricalEvidenceError("Duplicate archive member")
            names.add(normalized)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise HistoricalEvidenceError("Non-regular archive member")
            total += entry.file_size
            if entry.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise HistoricalEvidenceError("Expanded archive exceeds size limits")
            if entry.flag_bits & 1:
                raise HistoricalEvidenceError("Encrypted archive member")
            if entry.is_dir() or not ALLOWED_MEMBER.fullmatch(normalized):
                continue
            data = archive.read(entry)
            if len(data) != entry.file_size:
                raise HistoricalEvidenceError("Incomplete archive member")
            key = normalized.removeprefix("data/")
            if key in accepted:
                raise HistoricalEvidenceError("Ambiguous archive path aliases")
            accepted[key] = data
    return accepted


def read_diagnostics_archive(payload: bytes) -> dict[str, bytes]:
    try:
        return _read_diagnostics_archive(payload)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError) as exc:
        raise HistoricalEvidenceError("Invalid or unsupported diagnostics ZIP archive") from exc


def _load(payload: bytes):
    def reject_constant(value):
        raise HistoricalEvidenceError("Non-finite JSON number")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalEvidenceError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def _validate_items(items: object) -> list[dict]:
    if not isinstance(items, list):
        raise HistoricalEvidenceError("Gathered news must be an ordered list")
    ids = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise HistoricalEvidenceError("Missing source ID")
        if any(not isinstance(item.get(key, ""), str) for key in ("title", "content", "source")):
            raise HistoricalEvidenceError("Invalid filter evidence field")
        ids.append(item["id"])
    if len(ids) != len(set(ids)):
        raise HistoricalEvidenceError("Duplicate gathered source IDs")
    return items


def reconstruct_text(span: dict) -> str:
    if span.get("outcome") != "ok" or span.get("stop_reason") not in ("end_turn", "stop"):
        raise HistoricalEvidenceError("Filter attempt did not terminate normally")
    if span.get("truncated") is not False or span.get("dropped_deltas") != 0:
        raise HistoricalEvidenceError("Filter response capture is incomplete")
    deltas = span.get("deltas") or {}
    kinds, texts, times = (deltas.get(k) for k in ("kind", "text", "t"))
    if not all(isinstance(v, list) for v in (kinds, texts, times)) or len({len(kinds), len(texts), len(times)}) != 1:
        raise HistoricalEvidenceError("Malformed filter delta arrays")
    if any(type(kind) is not int or kind not in (0, 1) for kind in kinds) or any(not isinstance(t, str) for t in texts):
        raise HistoricalEvidenceError("Invalid filter delta schema")
    text = "".join(value for kind, value in zip(kinds, texts) if kind == 1)
    if type(span.get("text_chars")) is not int or len(text) != span["text_chars"]:
        raise HistoricalEvidenceError("Final filter text length mismatch")
    return text


def map_selected_ids(raw_ids: object, full_ids: list[str]) -> list[str]:
    if not isinstance(raw_ids, list) or any(not isinstance(v, str) or not v for v in raw_ids):
        raise HistoricalEvidenceError("Filter response requires a string ID list")
    if len(raw_ids) != len(set(raw_ids)):
        raise HistoricalEvidenceError("Duplicate returned filter IDs")
    display_ids = [value[:16] for value in full_ids]
    if len(display_ids) != len(set(display_ids)):
        raise HistoricalEvidenceError("Ambiguous displayed source IDs")
    selected = []
    for returned in raw_ids:
        if returned in display_ids:
            matches = [full_ids[display_ids.index(returned)]]
        else:
            # Match the incumbent's unusual eight-character prefix rule, but
            # reject ambiguity instead of silently selecting its first match.
            matches = [full for short, full in zip(display_ids, full_ids)
                       if short.startswith(returned[:8]) or returned.startswith(short[:8])]
        if len(matches) != 1:
            raise HistoricalEvidenceError("Unknown or ambiguous returned source ID")
        if matches[0] in selected:
            raise HistoricalEvidenceError("Multiple returned aliases for the same source ID")
        selected.append(matches[0])
    return selected


def _parse_answer(text: str) -> dict:
    # Older GLM responses put introductory prose around their JSON code fence.
    # Only JSON objects in the final text channel can supply IDs; prose and all
    # thinking deltas are never interpreted. Multiple answer objects are unsafe.
    if len(text) > 2 * 1024 * 1024:
        raise HistoricalEvidenceError("Filter response exceeds reconstruction size limit")
    candidates = []
    decoder = json.JSONDecoder()
    for index, match in enumerate(re.finditer(r"\{", text)):
        if index >= 1024:
            raise HistoricalEvidenceError("Too many JSON objects in filter response")
        try:
            value, length = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "ai_article_ids" in value:
            candidates.append(_load(text[match.start():match.start() + length].encode()))
    if len(candidates) != 1:
        raise HistoricalEvidenceError("Missing or ambiguous authoritative filter JSON answer")
    return candidates[0]


def validate_prompt(span: dict, items: list[dict]) -> tuple[str, str]:
    prompt = span.get("prompt") or {}
    system, recorded = prompt.get("system"), prompt.get("messages")
    if not isinstance(system, str) or not isinstance(recorded, str) or span.get("prompt_truncated") is not False:
        raise HistoricalEvidenceError("Filter request capture is incomplete")
    if span.get("prompt_chars") != len(system) + len(recorded):
        raise HistoricalEvidenceError("Filter request length mismatch")
    if not recorded.startswith("[USER]\n"):
        raise HistoricalEvidenceError("Unrecognized historical filter message structure")
    user = recorded[len("[USER]\n"):]
    nonces = re.findall(r'<source_data nonce="([a-f0-9]{16})">', system)
    if len(nonces) != 1:
        raise HistoricalEvidenceError("Unrecognized historical request fence")
    nonce = nonces[0]
    if user != build_fenced_user_message(render_filter_input(items), nonce):
        raise HistoricalEvidenceError("Historical keyword selection/order or evidence rendering mismatch")
    expected = [filter_system_prompt(items[0]["id"][:16], nonce, trailing_newline=value) for value in (False, True)]
    if system not in expected:
        raise HistoricalEvidenceError("Unknown historical filter rubric/security renderer")
    return system, user


def _attempt_record(span: dict) -> dict:
    context = span.get("context") or {}
    return {"call_id": span.get("id"), "request_id": span.get("request_id"),
            "outcome": span.get("outcome"), "stop_reason": span.get("stop_reason"),
            "provider_id": span.get("provider_id"), "model": span.get("provider_model"),
            "route_attempt": context.get("attempt"), "fallback_from": context.get("fallback_from"),
            "retry_reason": context.get("retry_reason"), "start_ms": span.get("start_ms"),
            "end_ms": span.get("end_ms"), "input_tokens": span.get("input_tokens"),
            "output_tokens": span.get("output_tokens"), "cache_read_tokens": span.get("cache_read_tokens"),
            "cache_creation_tokens": span.get("cache_creation_tokens"),
            "usage_partial": span.get("outcome") != "ok", "cost_usd": None,
            "prompt_sha256": sha256_json(span.get("prompt"))}


def recover_filter(gathered: list[dict], spans: list[dict]) -> tuple[dict, dict]:
    items = keyword_filter(_validate_items(gathered))
    calls = [s for s in spans if s.get("caller") == "news_analyzer.filter"]
    if not items:
        if calls:
            raise HistoricalEvidenceError("Filter calls exist despite empty keyword input")
        records = relevance_records(items)
        return {"schema_version": "news-relevance-input/v1", "records": records, "ordered_ids": [],
                "input_sha256": sha256_json(records), "system_prompt": "", "user_message": ""}, {
                    "schema_version": "news-incumbent-decision/v1", "decisions": [], "attempts": [],
                    "model": None, "input_sha256": sha256_json(records), "final_call_id": None,
                    "raw_selected_ids": [], "selected_ids": []}
    if not calls:
        raise HistoricalEvidenceError("No captured semantic-filter call")
    if len({c.get("id") for c in calls}) != len(calls):
        raise HistoricalEvidenceError("Ambiguous duplicate filter call identities")
    # Timeline order matters on routed retries (September 15 c026). Never parse
    # failed partial JSON; a later failed call invalidates an earlier success.
    if any(not isinstance(c.get("start_ms"), (int, float)) for c in calls):
        raise HistoricalEvidenceError("Missing filter attempt timeline")
    calls = sorted(calls, key=lambda c: (c["start_ms"], c.get("id", "")))
    selected_span = calls[-1]
    text = reconstruct_text(selected_span)
    system, user = validate_prompt(selected_span, items)
    # Retries must refer to the same complete prompt; missing failed-attempt
    # capture is preserved, but an explicitly different prompt is ambiguous.
    for call in calls[:-1]:
        if call.get("prompt") and call["prompt"] != selected_span.get("prompt"):
            raise HistoricalEvidenceError("Filter attempt prompts do not establish one retry chain")
    raw_ids = _parse_answer(text)["ai_article_ids"]
    full_ids = [item["id"] for item in items]
    selected = set(map_selected_ids(raw_ids, full_ids))
    records = relevance_records(items)
    digest = sha256_json(records)
    inputs = {"schema_version": "news-relevance-input/v1", "records": records, "ordered_ids": full_ids,
              "input_sha256": digest, "system_prompt": system, "user_message": user}
    decisions = {"schema_version": "news-incumbent-decision/v1", "input_sha256": digest,
                 "decisions": [{"id": item_id, "decision": "keep" if item_id in selected else "reject",
                                "effective_keep": item_id in selected} for item_id in full_ids],
                 "attempts": [_attempt_record(call) for call in calls],
                 "model": selected_span.get("provider_model"), "final_call_id": selected_span["id"],
                 "raw_selected_ids": raw_ids, "selected_ids": [v for v in full_ids if v in selected],
                 "response_text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return inputs, decisions


def source_health(gathering: dict, result: dict) -> dict:
    statuses = gathering.get("collection_status") or {}
    normalized = {key: value.get("status") if isinstance(value, dict) else value for key, value in statuses.items()}
    healthy = all(normalized.get(cat) == "success" for cat in CATEGORIES)
    healthy = healthy and all(value == "success" for value in normalized.values())
    degraded_steps = [{"source": source, "step": step.get("name"), "status": step.get("status")}
                      for source, value in statuses.items() if isinstance(value, dict)
                      for step in value.get("steps", []) if isinstance(step, dict)
                      and step.get("status") in ("partial", "failed")]
    reddit_empty = not bool((gathering.get("categories") or {}).get("reddit"))
    healthy = healthy and not reddit_empty and not degraded_steps
    healthy = healthy and not result.get("degradations")
    return {"healthy": healthy, "collection_status": statuses,
            "degradations": result.get("degradations", []), "degraded_steps": degraded_steps,
            "reddit_empty": reddit_empty,
            "missing_statuses": [cat for cat in CATEGORIES if cat not in normalized]}


def _coverage(report_date: str, result: dict) -> dict:
    report_day = parse_date(report_date)
    day = report_day - timedelta(days=1)
    zone = ZoneInfo("America/New_York")
    expected_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=zone)
    expected_end = datetime.combine(report_day, datetime.min.time()).replace(tzinfo=zone) - timedelta(microseconds=1)
    for key, expected in (("coverage_start", expected_start), ("coverage_end", expected_end)):
        if result.get(key):
            recorded = datetime.fromisoformat(result[key])
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=zone)
            if recorded != expected:
                raise HistoricalEvidenceError("Coverage dates disagree with the report date")
    return {"timezone": "America/New_York", "start": expected_start.isoformat(), "end": expected_end.isoformat(),
            "basis": "recorded" if result.get("coverage_start") and result.get("coverage_end") else "report_date_contract"}


def _usage_rows(rows: list[dict]) -> list[dict]:
    allowed = {"timestamp", "caller", "thinking_level", "input_tokens", "output_tokens", "cache_creation_tokens",
               "cache_read_tokens", "model", "provider_id", "analysis_profile", "adaptive_effort", "duration_seconds",
               "partial", "cost", "cost_usd", "total_cost", "input_cost", "output_cost", "cache_write_cost", "cache_hit_cost"}
    return [{key: value for key, value in row.items() if key in allowed} for row in rows
            if isinstance(row, dict) and row.get("caller") == "news_analyzer.filter"]


def _validate_publication(result: dict, publication: dict, files: dict[str, bytes], metadata: dict) -> dict:
    receipt = dict(publication)
    if not files:
        return {"status": "unverified", "reason": "No commit-bound output files supplied"}
    commit = receipt.get("output_commit")
    if not isinstance(commit, str) or not SHA_RE.fullmatch(commit):
        raise HistoricalEvidenceError("Publication needs a full output commit SHA")
    if not metadata.get("output_commit") or not commit.startswith(metadata["output_commit"]):
        raise HistoricalEvidenceError("Publication commit is not bound to source run lineage")
    expected_files = {f"{name}.json" for name in ("summary", *CATEGORIES)}
    if set(files) != expected_files or set(receipt.get("files", {})) != expected_files:
        raise HistoricalEvidenceError("Publication requires all five report files and hashes")
    for name, payload in files.items():
        if hashlib.sha256(payload).hexdigest() != receipt["files"][name]:
            raise HistoricalEvidenceError("Publication file digest mismatch")
    summary = _load(files["summary.json"])
    # The trusted pushed commit binds exact output bytes. Corroborate with
    # structured run output; a same-date current publication is insufficient.
    for key in ("date", "generated_at", "coverage_start", "coverage_end", "executive_summary", "total_items_collected", "total_items_analyzed"):
        if not result.get(key) and key in ("date", "generated_at"):
            raise HistoricalEvidenceError("Missing result publication identity")
        if summary.get(key) != result.get(key):
            raise HistoricalEvidenceError("Published summary belongs to a different run/result")
    for category in CATEGORIES:
        output = _load(files[f"{category}.json"])
        report = result.get("category_reports", {}).get(category, {})
        if output.get("date") != result["date"] or output.get("category") != category:
            raise HistoricalEvidenceError("Published category/date identity mismatch")
        if [item.get("id") for item in output.get("items", [])] != [item.get("id") for item in report.get("all_items", [])]:
            raise HistoricalEvidenceError("Published output item order/IDs disagree with source result")
        if output.get("category_summary") != report.get("category_summary"):
            raise HistoricalEvidenceError("Published category summary differs from source result")
    if receipt.get("verification") not in ("github_git_blob_hash", "local_git_blob_hash"):
        receipt["status"] = "unverified"
        receipt["reason"] = "File hashes supplied without Git object verification"
    receipt["original_publication_verified"] = receipt.get("verification") in ("github_git_blob_hash", "local_git_blob_hash")
    receipt["run_id"] = metadata["run_id"]
    receipt["run_attempt"] = metadata["run_attempt"]
    return receipt


def import_legacy_bundle(payload: bytes, metadata: dict, destination: Path,
                         publication: dict | None = None, output_files: dict[str, bytes] | None = None) -> dict:
    validate_repository(metadata.get("repository", SOURCE_REPOSITORY))
    run_id = positive_int(metadata.get("run_id", metadata.get("id")), "run_id")
    attempt = positive_int(metadata.get("run_attempt", metadata.get("attempt")), "run_attempt")
    metadata = {**metadata, "run_id": run_id, "run_attempt": attempt}
    if metadata.get("workflow_path", WORKFLOW_PATH) != WORKFLOW_PATH:
        raise HistoricalEvidenceError("Incorrect source workflow")
    archive_hash = hashlib.sha256(payload).hexdigest()
    artifact = metadata.get("artifact") or {}
    if artifact.get("digest") and artifact["digest"] != "sha256:" + archive_hash:
        raise HistoricalEvidenceError("Diagnostics archive digest mismatch")
    members = read_diagnostics_archive(payload)
    dates = {name.split("/")[1] for name in members if name.startswith("checkpoints/")}
    requested_date = metadata.get("report_date")
    if requested_date:
        parse_date(requested_date)
        if requested_date not in dates:
            raise HistoricalEvidenceError("Requested report date has no checkpoints")
        report_date = requested_date
    elif len(dates) == 1:
        report_date = dates.pop()
    else:
        raise HistoricalEvidenceError("Missing or ambiguous report date; provide report_date metadata")
    result = _load(members.get(f"processed/orchestrator_result_{report_date}.json", b"{}"))
    if result.get("date", report_date) != report_date:
        raise HistoricalEvidenceError("Result/checkpoint date mismatch")
    gathering = _load(members.get(f"checkpoints/{report_date}/gathering.json", b"{}"))
    reasons = [{"reason": "legacy_dependencies_unavailable", "key": key} for key in
               ("context/grounding.txt", "context/model-releases-before.yaml", "context/effective-config.json",
                "context/history-manifest.json", "evidence/index.json", "checkpoints/analysis_pre_continuity.json")]
    checkpoint_candidates = []
    for rank, phase in enumerate(("gathering", "analysis", "topics", "summary", "hero")):
        data = members.get(f"checkpoints/{report_date}/{phase}.json")
        if data:
            checkpoint = _load(data)
            replay = checkpoint.get("_replay") or {}
            checkpoint_candidates.append((len(replay.get("spans", [])), rank, replay))
    replay = max(checkpoint_candidates, key=lambda entry: entry[:2])[2] if checkpoint_candidates else {}
    gathered = (gathering.get("categories") or {}).get("news")
    inputs, decision = None, None
    try:
        inputs, decision = recover_filter(gathered, replay.get("spans", []))
    except (HistoricalEvidenceError, ValueError, TypeError, KeyError) as error:
        reasons.append({"reason": "filter_evidence_invalid", "key": "relevance/input.json", "detail": str(error)})
    execution_sha = metadata.get("execution_sha")
    if not isinstance(execution_sha, str) or not SHA_RE.fullmatch(execution_sha):
        reasons.append({"reason": "execution_sha_unverified", "key": "source.execution_sha"})
        execution_sha = None
    health = source_health(gathering, result)
    coverage = _coverage(report_date, result)
    publication_receipt = _validate_publication(result, publication or {}, output_files or {}, metadata)
    source = {"repository": SOURCE_REPOSITORY, "workflow_path": WORKFLOW_PATH, "run_id": run_id,
              "run_attempt": attempt, "event_sha": metadata.get("event_sha", metadata.get("head_sha")),
              "execution_sha": execution_sha, "execution_sha_evidence": metadata.get("execution_sha_evidence", "explicit_metadata"),
              "output_commit": publication_receipt.get("output_commit"), "conclusion": metadata.get("conclusion"),
              "archive_sha256": archive_hash, "logs_sha256": metadata.get("logs_sha256"),
              "artifact": {key: artifact[key] for key in ("id", "name", "digest", "created_at", "expires_at", "expired") if key in artifact}}
    destination = Path(destination)
    if destination.exists():
        raise HistoricalEvidenceError("Bundle destination already exists; immutable imports require a new path")
    if any(parent.is_symlink() for parent in [destination, *destination.parents]):
        raise HistoricalEvidenceError("Bundle output must not traverse symlinks")
    destination.mkdir(parents=True)
    for category, items in (gathering.get("categories") or {}).items():
        if category in CATEGORIES:
            write_json(destination / "gathered" / f"{category}.json", items)
    write_json(destination / "diagnostics/health.json", health)
    write_json(destination / "diagnostics/source-result.json", {key: result.get(key) for key in
               ("date", "generated_at", "coverage_start", "coverage_end", "phase_status", "degradations")})
    write_json(destination / "diagnostics/filter-attempts.json", [
        _attempt_record(call) for call in replay.get("spans", []) if call.get("caller") == "news_analyzer.filter"])
    usage = _usage_rows(replay.get("cost_calls", []))
    cost_report = members.get(f"processed/cost_report_{report_date}.json")
    cost_summary = {"cost_usd": None, "basis": "unavailable", "per_attempt_cost_usd": None}
    if cost_report:
        report_cost = _load(cost_report)
        cost_rows = _usage_rows(report_cost.get("calls", []))
        if len(cost_rows) >= len(usage):
            usage = cost_rows
        recorded_cost = (report_cost.get("cost_by_component") or report_cost.get("cost_by_caller") or {}).get("news_analyzer.filter")
        if isinstance(recorded_cost, (int, float)) and not isinstance(recorded_cost, bool) and recorded_cost >= 0:
            cost_summary = {"cost_usd": recorded_cost, "basis": "production_tracker_estimate",
                            "per_attempt_cost_usd": None, "includes_recorded_failed_attempts": True,
                            "unreported_failed_attempts": sum(
                                call.get("outcome") != "ok" and
                                (call.get("input_tokens") is None or call.get("output_tokens") is None)
                                for call in replay.get("spans", []) if call.get("caller") == "news_analyzer.filter")}
    write_json(destination / "diagnostics/filter-cost.json", cost_summary)
    (destination / "diagnostics/usage.jsonl").write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in usage))
    if inputs is not None and decision is not None:
        write_json(destination / "relevance/input.json", inputs)
        write_json(destination / "relevance/incumbent-decision.json", decision)
    for name, content in (output_files or {}).items():
        target = destination / "original" / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(content)
    input_manifest = {"schema_version": "news-shadow-input-manifest/v1", "source": source,
                      "report_date": report_date, "coverage": coverage,
                      "input_sha256": inputs["input_sha256"] if inputs else None,
                      "archive_sha256": archive_hash}
    write_json(destination / "input-manifest.json", input_manifest)
    publication_receipt["input_manifest_sha256"] = hash_file(destination / "input-manifest.json")
    write_json(destination / "publication.json", publication_receipt)
    filter_capable = inputs is not None and execution_sha is not None
    manifest = {"schema_version": "news-shadow-bundle/v1", "source": source, "report_date": report_date,
                "coverage": coverage, "capabilities": {"filter_replay": filter_capable, "pipeline_replay": False},
                "capability": "filter_replay" if filter_capable else ("output_review" if output_files else "unavailable"),
                "missing_dependencies": reasons, "versions": {"keyword": KEYWORD_SHA256, "renderer": RENDERER_SHA256,
                "rubric": sha256_json(FILTER_PROMPT) if inputs else None},
                "source_health": {"healthy": health["healthy"]},
                "eligibility": {"healthy": health["healthy"], "successful_run": metadata.get("conclusion") == "success",
                                "published": publication_receipt.get("original_publication_verified") is True and
                                             publication_receipt.get("status") in ("published", "superseded")},
                "publication": {"status": publication_receipt["status"],
                                "original_publication_verified": publication_receipt.get("original_publication_verified", False),
                                "receipt_sha256": hash_file(destination / "publication.json")}}
    return seal_bundle(destination, manifest)
