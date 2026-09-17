#!/usr/bin/env python3
"""Read-only run/attempt discovery, or an offline index of imported manifests.

Discovery does not declare a successful scheduled no-op a published report. Each
run attempt remains pending/unqualified until diagnostics and publication proof
are imported. Every invocation rescans the selected window; no creation cursor
can strand late/retried runs. Persist this file as a discovery snapshot only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from shadow.contracts import read_json, write_json, verify_bundle
from shadow.github import GitHubSource, SOURCE_REPOSITORY, parse_date


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=SOURCE_REPOSITORY, choices=[SOURCE_REPOSITORY])
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--bundles", type=Path, help="Offline directory containing imported bundle directories")
    offline.add_argument("--metadata", type=Path, help="Offline prior inventory/metadata; does not upgrade its evidence")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        first, last = parse_date(args.from_date), parse_date(args.to_date)
        if last < first or (last - first).days > 89:
            raise ValueError("Choose an ordered date range of at most 90 dates")
        rows = []
        if args.bundles:
            for path in sorted(args.bundles.glob("*/manifest.json")):
                manifest = verify_bundle(path.parent, capability=None)
                if not first <= parse_date(manifest["report_date"]) <= last:
                    continue
                decision_path = path.parent / "relevance/incumbent-decision.json"
                decision = read_json(decision_path) if decision_path.exists() else {}
                inputs = read_json(path.parent / "relevance/input.json") if decision else {}
                gathered_path = path.parent / "gathered/news.json"
                rows.append({"run_id": manifest["source"]["run_id"], "run_attempt": manifest["source"]["run_attempt"],
                             "report_date": manifest["report_date"], "bundle_sha256": manifest["bundle_sha256"],
                             "capabilities": manifest["capabilities"], "eligibility": manifest.get("eligibility"),
                             "publication": manifest.get("publication"), "final_call_id": decision.get("final_call_id"),
                             "model": decision.get("model"), "gathered_news": len(read_json(gathered_path)) if gathered_path.exists() else None,
                             "input_count": len(inputs.get("records", [])) if decision else None,
                             "kept": sum(d["effective_keep"] for d in decision.get("decisions", [])) if decision else None,
                             "rejected": sum(not d["effective_keep"] for d in decision.get("decisions", [])) if decision else None})
            kind = "verified_local_bundles"
        elif args.metadata:
            value = read_json(args.metadata)
            source_rows = value.get("metadata_inventory", value.get("runs", [])) if isinstance(value, dict) else value
            for row in source_rows:
                created = str(row.get("created_at", ""))[:10]
                if created and first <= parse_date(created) <= last:
                    rows.append({**row, "filter_replay": None, "publication_status": "unverified"})
            kind = "offline_metadata_only"
        else:
            rows = GitHubSource(args.source_repo).inventory(args.from_date, args.to_date)
            kind = "github_metadata_only"
        result = {"schema_version": "news-shadow-inventory/v1", "source_repo": args.source_repo,
                  "from_date": args.from_date, "to_date": args.to_date, "kind": kind,
                  "observed_at": datetime.now(timezone.utc).isoformat(),
                  "date_semantics": "bundle report date" if args.bundles else "run creation discovery window; report date requires import",
                  "runs": rows}
        write_json(args.out, result)
        print(json.dumps({"inventory": str(args.out.resolve()), "entries": len(rows), "kind": kind}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"Inventory unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
