"""Small, dependency-free integrity contracts for captured experiment data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_SCHEMA = "news-shadow-bundle/v1"
INPUT_SCHEMA = "news-relevance-input/v1"
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_BUNDLE_FILES = 4096


class BundleValidationError(ValueError):
    """Evidence cannot support the requested comparison."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleValidationError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> Any:
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise BundleValidationError("JSON file is a symlink or exceeds its size limit")
    with path.open("rb") as handle:
        body = handle.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise BundleValidationError("JSON file exceeds its size limit")
    try:
        return json.loads(body, object_pairs_hook=_unique_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              BundleValidationError("Nonfinite JSON number")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Invalid JSON evidence") from exc


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    if path.is_symlink():
        raise BundleValidationError("Refusing to replace a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".shadow-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bundle_path(root: str | Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise BundleValidationError("Invalid bundle path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(p in (".", "..") for p in pure.parts):
        raise BundleValidationError("Bundle path must be a normalized relative path")
    if str(pure) != relative or "\x00" in relative:
        raise BundleValidationError("Noncanonical bundle path")
    root = Path(root).resolve()
    path = root.joinpath(*pure.parts)
    current = root
    for component in pure.parts:
        current = current / component
        if current.is_symlink():
            raise BundleValidationError("Bundle contains a symlink")
    if not path.resolve().is_relative_to(root):
        raise BundleValidationError("Bundle path escapes root")
    return path


def validate_records(records: Any) -> list[dict]:
    if not isinstance(records, list):
        raise BundleValidationError("Relevance records must be a list")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"id", "title", "source", "snippet"}:
            raise BundleValidationError("Unexpected relevance evidence fields")
        if not all(isinstance(value, str) for value in record.values()):
            raise BundleValidationError("Relevance evidence values must be strings")
        article_id = record["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", article_id) or article_id in seen:
            raise BundleValidationError("Missing, duplicate or malformed article ID")
        if len(record["title"]) > 303 or len(record["snippet"]) > 306:
            raise BundleValidationError("Article evidence exceeds the incumbent bound")
        if len(record["source"]) > 4096:
            raise BundleValidationError("Article source exceeds the safety bound")
        seen.add(article_id)
    return records


def validate_input(value: dict) -> dict:
    if not isinstance(value, dict):
        raise BundleValidationError("Relevance input is not an object")
    records = validate_records(value.get("records"))
    ids = [record["id"] for record in records]
    if value.get("ordered_ids", ids) != ids:
        raise BundleValidationError("Input ordering does not match records")
    if value.get("input_sha256") != sha256_json(records):
        raise BundleValidationError("Relevance input hash mismatch")
    for key in ("system_prompt", "user_message"):
        if not isinstance(value.get(key), str) or (records and not value[key].strip()):
            raise BundleValidationError(f"Missing frozen {key}")
    return value


def validate_decisions(records: list[dict], decisions: Any) -> list[dict]:
    validate_records(records)
    if not isinstance(decisions, list) or len(decisions) != len(records):
        raise BundleValidationError("Decision coverage is incomplete")
    by_id = {}
    for row in decisions:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row.get("id") in by_id:
            raise BundleValidationError("Duplicate or malformed decision")
        if row.get("decision") not in ("keep", "reject", "abstain"):
            raise BundleValidationError("Invalid semantic decision")
        if type(row.get("effective_keep")) is not bool:
            raise BundleValidationError("Decision effective_keep must be boolean")
        if row["effective_keep"] != (row["decision"] != "reject"):
            raise BundleValidationError("Decision violates keep/abstain policy")
        probabilities = row.get("probabilities")
        if probabilities is not None and not isinstance(probabilities, dict):
            raise BundleValidationError("Probabilities must be an object or null")
        for probability in (probabilities or {}).values():
            if probability is not None and (isinstance(probability, bool) or
                    not isinstance(probability, (int, float)) or
                    not math.isfinite(probability) or not 0 <= probability <= 1):
                raise BundleValidationError("Invalid probability")
        by_id[row["id"]] = row
    if set(by_id) != {record["id"] for record in records}:
        raise BundleValidationError("Decision IDs do not match input")
    return [by_id[record["id"]] for record in records]


def _inventory(root: Path) -> list[dict]:
    files = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BundleValidationError("Bundle contains a symlink")
        if not path.is_file() or path == root / "manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        bundle_path(root, relative)
        size = path.stat().st_size
        total += size
        if len(files) >= MAX_BUNDLE_FILES or total > MAX_BUNDLE_BYTES:
            raise BundleValidationError("Bundle exceeds file/byte limit")
        files.append({"path": relative, "sha256": hash_file(path), "bytes": size})
    return files


def seal_bundle(root: str | Path, manifest: dict) -> dict:
    root = Path(root)
    if root.is_symlink():
        raise BundleValidationError("Bundle root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise BundleValidationError("Bundle already sealed; choose a new output directory")
    result = dict(manifest)
    result["schema_version"] = BUNDLE_SCHEMA
    result["files"] = _inventory(root)
    result.pop("bundle_sha256", None)
    result["bundle_sha256"] = sha256_json(result)
    write_json(root / "manifest.json", result)
    return result


def verify_bundle(root: str | Path, *, capability: str | None = "filter_replay",
                  require_healthy: bool = False, require_published: bool = False) -> dict:
    root = Path(root)
    if root.is_symlink():
        raise BundleValidationError("Bundle root may not be a symlink")
    manifest = read_json(bundle_path(root, "manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleValidationError("Unsupported bundle schema")
    unhashed = {key: val for key, val in manifest.items() if key != "bundle_sha256"}
    if manifest.get("bundle_sha256") != sha256_json(unhashed):
        raise BundleValidationError("Manifest hash mismatch")
    if manifest.get("files") != _inventory(root):
        raise BundleValidationError("Bundle file inventory/hash mismatch")
    try:
        if date.fromisoformat(manifest["report_date"]).isoformat() != manifest["report_date"]:
            raise ValueError()
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleValidationError("Invalid report date") from exc
    if capability and (manifest.get("capabilities") or {}).get(capability) is not True:
        raise BundleValidationError(f"Bundle does not support {capability}")
    if require_healthy and (manifest.get("eligibility") or {}).get("healthy") is not True:
        raise BundleValidationError("Bundle is not a verified healthy production run")
    if require_published and (manifest.get("publication") or {}).get("status") != "published":
        raise BundleValidationError("Bundle is not bound to a verified publication")
    if (manifest.get("capabilities") or {}).get("filter_replay") is True:
        inputs = validate_input(read_json(bundle_path(root, "relevance/input.json")))
        incumbent = read_json(bundle_path(root, "relevance/incumbent-decision.json"))
        if incumbent.get("input_sha256") != inputs["input_sha256"]:
            raise BundleValidationError("Incumbent decision input hash mismatch")
        validate_decisions(inputs["records"], incumbent.get("decisions"))
    return manifest
