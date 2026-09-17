"""Deterministic, ground-truth-free metrics for shadow comparisons.

All functions in this module are pure transformations over JSON-like values.
They preserve ``None`` for undefined denominators and never turn a missing
judge label, unknown usage value, or absent decision into a zero.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else numerator / denominator


def _ids_from_records(records: Sequence[Mapping[str, Any]] | Sequence[str] | None) -> list[str]:
    if records is None:
        return []
    ids: list[str] = []
    for record in records:
        if isinstance(record, str):
            article_id = record
        elif isinstance(record, Mapping):
            article_id = record.get("id")
        else:
            continue
        if isinstance(article_id, str) and article_id not in ids:
            ids.append(article_id)
    return ids


def _decision_rows(decisions: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> Sequence[Any]:
    if isinstance(decisions, Mapping) and isinstance(decisions.get("decisions"), list):
        return decisions["decisions"]
    return decisions or []


def _decision_map(decisions: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    result: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in _decision_rows(decisions):
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            continue
        article_id = row["id"]
        if article_id in result:
            duplicates.append(article_id)
        else:
            result[article_id] = row
    return result, sorted(set(duplicates))


def _is_kept(row: Mapping[str, Any]) -> bool:
    return row.get("effective_keep") is True


def _semantic_decision(row: Mapping[str, Any]) -> str:
    decision = row.get("decision")
    return decision if decision in {"keep", "reject", "abstain"} else "abstain"


def summarize_branch(
    decisions: Sequence[Mapping[str, Any]] | None,
    input_ids: Sequence[str],
) -> dict[str, Any]:
    """Return coverage, selection, abstention, error, and fallback counts."""

    expected = set(input_ids)
    rows, duplicates = _decision_map(decisions)
    present = expected & set(rows)
    missing = sorted(expected - set(rows))
    extras = sorted(set(rows) - expected)
    kept = {article_id for article_id in present if _is_kept(rows[article_id])}
    rejected = {
        article_id
        for article_id in present
        if _semantic_decision(rows[article_id]) == "reject" and not _is_kept(rows[article_id])
    }
    abstained = {
        article_id for article_id in present if _semantic_decision(rows[article_id]) == "abstain"
    }
    fallback = {
        article_id for article_id in present if rows[article_id].get("fallback_reason") is not None
    }
    errors = {
        article_id
        for article_id in present
        if rows[article_id].get("fallback_reason") not in (None, "", "insufficient_evidence")
    }
    count = len(input_ids)
    # An empty decision collection means this branch produced no usable
    # response.  Its rates are therefore undefined, even when the input
    # population is known: reporting 0% would turn an unavailable branch into
    # an observed outcome.  Once at least one decision exists, use the full
    # input population so partial responses retain measurable coverage and
    # missing IDs remain visible in the counts above.
    rate_denominator = count if rows else 0
    return {
        "input_count": count,
        "decision_count": len(present),
        "missing_count": len(missing),
        "missing_ids": missing,
        "extra_count": len(extras),
        "extra_ids": extras,
        "duplicate_ids": duplicates,
        "coverage_count": len(present),
        "coverage_rate": _rate(len(present), rate_denominator),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "abstention_count": len(abstained),
        "fallback_count": len(fallback),
        "error_count": len(errors),
        "keep_rate": _rate(len(kept), rate_denominator),
        "abstention_rate": _rate(len(abstained), rate_denominator),
        "fallback_rate": _rate(len(fallback), rate_denominator),
        "error_rate": _rate(len(errors), rate_denominator),
        "kept_ids": sorted(kept),
        "rejected_ids": sorted(rejected),
        "abstained_ids": sorted(abstained),
        "fallback_ids": sorted(fallback),
        "error_ids": sorted(errors),
    }


def set_differences(
    left_decisions: Sequence[Mapping[str, Any]] | None,
    right_decisions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Compare effective kept sets with stable sorted IDs."""

    left, _ = _decision_map(left_decisions)
    right, _ = _decision_map(right_decisions)
    left_kept = {article_id for article_id, row in left.items() if _is_kept(row)}
    right_kept = {article_id for article_id, row in right.items() if _is_kept(row)}
    added = sorted(right_kept - left_kept)
    removed = sorted(left_kept - right_kept)
    return {
        "left_kept_count": len(left_kept),
        "right_kept_count": len(right_kept),
        "intersection_count": len(left_kept & right_kept),
        "added_count": len(added),
        "removed_count": len(removed),
        "added_ids": added,
        "removed_ids": removed,
        "jaccard": _rate(len(left_kept & right_kept), len(left_kept | right_kept)),
    }


def _paired_agreement(
    left_decisions: Sequence[Mapping[str, Any]] | None,
    right_decisions: Sequence[Mapping[str, Any]] | None,
    input_ids: Sequence[str],
) -> dict[str, Any]:
    left, _ = _decision_map(left_decisions)
    right, _ = _decision_map(right_decisions)
    comparable = [article_id for article_id in input_ids if article_id in left and article_id in right]
    changed = [article_id for article_id in comparable if _is_kept(left[article_id]) != _is_kept(right[article_id])]
    return {
        "comparable_count": len(comparable),
        "changed_count": len(changed),
        "unchanged_count": len(comparable) - len(changed),
        "change_rate": _rate(len(changed), len(comparable)),
        "changed_ids": sorted(changed),
    }


def rank_comparison(
    left_decisions: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    right_decisions: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare explicit integer ranks when branches recorded them.

    A missing rank is unavailable evidence, so the overlap and rank deltas are
    null instead of being inferred from JSON/list order.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    left, _ = _decision_map(left_decisions)
    right, _ = _decision_map(right_decisions)
    left_ranked = {
        article_id: row.get("rank")
        for article_id, row in left.items()
        if isinstance(row.get("rank"), int) and not isinstance(row.get("rank"), bool) and row.get("rank") > 0
    }
    right_ranked = {
        article_id: row.get("rank")
        for article_id, row in right.items()
        if isinstance(row.get("rank"), int) and not isinstance(row.get("rank"), bool) and row.get("rank") > 0
    }
    if not left_ranked or not right_ranked:
        return {"top_k": top_k, "available": False, "top_k_overlap": None, "rank_changes": None}
    left_top = {article_id for article_id, rank in left_ranked.items() if rank <= top_k}
    right_top = {article_id for article_id, rank in right_ranked.items() if rank <= top_k}
    common = set(left_ranked) & set(right_ranked)
    changes = {
        article_id: right_ranked[article_id] - left_ranked[article_id]
        for article_id in sorted(common)
        if right_ranked[article_id] != left_ranked[article_id]
    }
    return {
        "top_k": top_k,
        "available": True,
        "left_ranked_count": len(left_ranked),
        "right_ranked_count": len(right_ranked),
        "top_k_overlap_count": len(left_top & right_top),
        "top_k_overlap": _rate(len(left_top & right_top), len(left_top | right_top)),
        "rank_changes": changes,
    }


def judge_estimated_quality(
    decisions: Sequence[Mapping[str, Any]] | None,
    adjudications: Sequence[Mapping[str, Any]] | None,
    input_ids: Sequence[str],
) -> dict[str, Any]:
    """Estimate selection quality from judge labels, explicitly not truth."""

    decision_map, _ = _decision_map(decisions)
    labels: dict[str, Mapping[str, Any]] = {}
    for row in adjudications or []:
        if isinstance(row, Mapping) and isinstance(row.get("article_id"), str):
            labels[row["article_id"]] = row
    judged_ids = [article_id for article_id in input_ids if article_id in labels]
    reliable = {
        article_id
        for article_id in judged_ids
        if labels[article_id].get("evidence_sufficiency") not in {"insufficient", "unsupported"}
    }
    relevant = {
        article_id
        for article_id in judged_ids
        if article_id in reliable and labels[article_id].get("relevance") == "relevant"
    }
    irrelevant = {
        article_id
        for article_id in judged_ids
        if article_id in reliable and labels[article_id].get("relevance") == "irrelevant"
    }
    insufficient = {
        article_id
        for article_id in judged_ids
        if article_id not in reliable or labels[article_id].get("relevance") == "insufficient_evidence"
    }
    kept = {article_id for article_id in judged_ids if article_id in decision_map and _is_kept(decision_map[article_id])}
    true_positive = len(kept & relevant)
    false_positive = len(kept & irrelevant)
    false_negative = len(relevant - kept)
    return {
        "basis": "judge_estimated_not_ground_truth",
        "judged_count": len(judged_ids),
        "coverage_rate": _rate(len(judged_ids), len(input_ids)),
        "relevant_count": len(relevant),
        "irrelevant_count": len(irrelevant),
        "insufficient_evidence_count": len(insufficient),
        "kept_judged_count": len(kept),
        "true_positive_estimate": true_positive,
        "false_positive_estimate": false_positive,
        "false_negative_estimate": false_negative,
        "precision_estimate": _rate(true_positive, true_positive + false_positive),
        "recall_estimate": _rate(true_positive, len(relevant)),
        "irrelevant_keep_rate_estimate": _rate(false_positive, len(irrelevant)),
        "critical_kept_count": sum(
            1 for article_id in kept if labels[article_id].get("critical_story") is True
        ),
        "critical_lost_count": sum(
            1 for article_id in relevant - kept if labels[article_id].get("critical_story") is True
        ),
    }


def summarize_requests(requests: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Aggregate token/cost accounting while preserving unknowns."""

    rows = [row for row in (requests or []) if isinstance(row, Mapping)]
    input_known = [row["input_tokens"] for row in rows if isinstance(row.get("input_tokens"), int)]
    output_known = [row["output_tokens"] for row in rows if isinstance(row.get("output_tokens"), int)]
    total_known = [row["total_tokens"] for row in rows if isinstance(row.get("total_tokens"), int)]
    cost_values = [
        float(row["cost_usd"])
        for row in rows
        if isinstance(row.get("cost_usd"), (int, float))
        and not isinstance(row.get("cost_usd"), bool)
        and math.isfinite(float(row["cost_usd"]))
    ]
    elapsed_values = [
        float(row["elapsed_ms"])
        for row in rows
        if isinstance(row.get("elapsed_ms"), (int, float))
        and not isinstance(row.get("elapsed_ms"), bool)
        and math.isfinite(float(row["elapsed_ms"]))
    ]
    unknown_usage = sum(
        1 for row in rows if row.get("usage_known") is not True or row.get("input_tokens") is None or row.get("output_tokens") is None
    )
    unknown_cost = sum(
        1 for row in rows if row.get("cost_known") is not True or row.get("cost_usd") is None
    )
    if not rows:
        cost_status = "undefined"
        known_cost: float | None = None
    elif unknown_cost == 0:
        cost_status = "known"
        known_cost = sum(cost_values)
    elif cost_values:
        cost_status = "partially_known"
        known_cost = sum(cost_values)
    else:
        cost_status = "unknown"
        known_cost = None
    return {
        "request_count": len(rows),
        "successful_request_count": sum(1 for row in rows if row.get("status") == "success"),
        "failed_request_count": sum(1 for row in rows if row.get("status") in {"error", "failed"}),
        "input_tokens_known": sum(input_known) if input_known else None,
        "output_tokens_known": sum(output_known) if output_known else None,
        "total_tokens_known": sum(total_known) if total_known else None,
        "unknown_usage_request_count": unknown_usage,
        "elapsed_ms_known": sum(elapsed_values) if elapsed_values else None,
        "cost_status": cost_status,
        "cost_usd_known": known_cost,
        "unknown_cost_request_count": unknown_cost,
    }


def summarize_output_comparisons(comparisons: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Count mapped judge outcomes without treating them as labelled truth."""

    rows = [row for row in (comparisons or []) if isinstance(row, Mapping)]
    dimension_counts: dict[str, dict[str, int]] = {dimension: {} for dimension in (
        "important_story_coverage",
        "irrelevant_inclusions",
        "safety_policy_omissions",
        "supported_claims",
        "duplicates",
        "ranking_usefulness",
        "summary_quality",
    )}
    overall: dict[str, int] = {}
    critical_findings: list[dict[str, Any]] = []
    repeat_selected = 0
    repeat_consistent = 0
    repeat_inconsistent = 0
    repeat_unavailable = 0
    for row in rows:
        comparison = row.get("comparison", row)
        if not isinstance(comparison, Mapping):
            continue
        dimensions = comparison.get("dimensions")
        if isinstance(dimensions, Mapping):
            for key, value in dimensions.items():
                if key in dimension_counts and isinstance(value, str):
                    dimension_counts[key][value] = dimension_counts[key].get(value, 0) + 1
        value = comparison.get("overall")
        if isinstance(value, str):
            overall[value] = overall.get(value, 0) + 1
        for finding in comparison.get("findings", []) if isinstance(comparison.get("findings"), list) else []:
            if isinstance(finding, Mapping) and finding.get("severity") == "critical":
                critical_findings.append(dict(finding))
        repeat = row.get("repeat")
        if isinstance(repeat, Mapping) and repeat.get("selected") is True:
            repeat_selected += 1
            if repeat.get("consistent") is True:
                repeat_consistent += 1
            elif repeat.get("consistent") is False:
                repeat_inconsistent += 1
            else:
                repeat_unavailable += 1
    return {
        "comparison_count": len(rows),
        "dimensions": dimension_counts,
        "overall": overall,
        "critical_findings": critical_findings,
        "repeat_selected_count": repeat_selected,
        "repeat_consistent_count": repeat_consistent,
        "repeat_inconsistent_count": repeat_inconsistent,
        "repeat_unavailable_count": repeat_unavailable,
        "repeat_consistency_rate": _rate(repeat_consistent, repeat_selected),
        "basis": "judge_estimated_not_ground_truth",
    }


# The full-pipeline branch produces JSON files rather than filter decision
# rows.  Keep this small extractor here so the comparison remains deterministic
# and does not depend on a live pipeline import.  It intentionally uses the
# order in which article objects occur in the frozen output as their fallback
# rank.  An explicit positive ``rank`` is retained when a producer supplies
# one, but scores, timestamps, and dictionary key order never become an
# inferred ranking.
_PIPELINE_CATEGORIES = frozenset({"news", "research", "social", "reddit"})
_PIPELINE_ITEM_CONTAINERS = frozenset(
    {"items", "articles", "stories", "results", "representative_items", "example_items"}
)
_PIPELINE_ID_KEYS = ("id", "article_id", "item_id")
_PIPELINE_CATEGORY_KEYS = ("category", "source_category")
_PIPELINE_ARTICLE_FIELDS = frozenset(
    {"title", "headline", "source", "source_type", "url", "content", "summary", "snippet", "published"}
)


def _pipeline_category(value: Any, path: Sequence[str]) -> str | None:
    if isinstance(value, str) and value.strip():
        candidate = value.strip().lower()
        if candidate in _PIPELINE_CATEGORIES:
            return candidate
    for component in reversed(path):
        candidate = str(component).lower()
        if candidate.endswith(".json"):
            candidate = candidate[:-5]
        if candidate in _PIPELINE_CATEGORIES:
            return candidate
    return None


def _pipeline_item_id(value: Mapping[str, Any]) -> str | None:
    for key in _PIPELINE_ID_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _looks_like_pipeline_item(value: Mapping[str, Any], path: Sequence[str]) -> bool:
    article_id = _pipeline_item_id(value)
    if article_id is None:
        return False
    keys = {str(key) for key in value}
    if len(keys & _PIPELINE_ARTICLE_FIELDS) >= 2:
        return True
    return bool(path and str(path[-1]).lower() in _PIPELINE_ITEM_CONTAINERS)


def _pipeline_items(output: Any) -> list[dict[str, Any]]:
    """Extract article-like rows while preserving output list order.

    Published output has several nested representations of the same article
    (for example a summary's representative item and a category file's full
    item).  The first occurrence wins; later duplicates are ignored so counts,
    ranks, and category deltas are stable for the same JSON artifact.
    """

    found: dict[str, dict[str, Any]] = {}
    duplicate_count = 0

    def visit(value: Any, path: tuple[str, ...], inherited_category: str | None) -> None:
        nonlocal duplicate_count
        if isinstance(value, Mapping):
            explicit_category = next(
                (value.get(key) for key in _PIPELINE_CATEGORY_KEYS if key in value), None
            )
            category = _pipeline_category(explicit_category, path) or inherited_category or _pipeline_category(None, path)
            if _looks_like_pipeline_item(value, path):
                article_id = _pipeline_item_id(value)
                assert article_id is not None
                if article_id not in found:
                    explicit_rank = value.get("rank")
                    if not (isinstance(explicit_rank, int) and not isinstance(explicit_rank, bool) and explicit_rank > 0):
                        explicit_rank = None
                    found[article_id] = {
                        "id": article_id,
                        "category": category or "unknown",
                        "explicit_rank": explicit_rank,
                    }
                else:
                    duplicate_count += 1
            # Mapping order is not part of an output ranking.  Known container
            # lists carry ranking; sorting keys here makes traversal stable for
            # semantically equivalent mappings created by different writers.
            for key in sorted(value, key=str):
                visit(value[key], path + (str(key),), category or _pipeline_category(None, path + (str(key),)))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, path, inherited_category)

    visit(output, (), None)
    rows = list(found.values())
    # Keep the count available to ``pipeline_output_view`` without exposing a
    # mutable module-level accumulator.  The value is copied onto the list's
    # first row only when needed; an empty output naturally has no duplicates.
    if rows:
        rows[0]["_duplicate_count"] = duplicate_count
    for ordinal, row in enumerate(rows, start=1):
        row["rank"] = row["explicit_rank"] if row["explicit_rank"] is not None else ordinal
        row.pop("explicit_rank", None)
    return rows


def pipeline_output_view(output: Any, *, top_k: int = 10) -> dict[str, Any]:
    """Build a deterministic article/rank/category view of one pipeline output."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rows = _pipeline_items(output)
    duplicate_count = int(rows[0].get("_duplicate_count", 0)) if rows else 0
    for row in rows:
        row.pop("_duplicate_count", None)
    ordered_ids = [row["id"] for row in rows]
    ranks = {row["id"]: row["rank"] for row in rows}
    rank_ordered_ids = [
        row["id"] for _, row in sorted(enumerate(rows), key=lambda item: (item[1]["rank"], item[0]))
    ]
    category_counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "available": True,
        "item_count": len(rows),
        "ordered_ids": ordered_ids,
        "ranks": {key: ranks[key] for key in sorted(ranks)},
        "category_counts": {key: category_counts[key] for key in sorted(category_counts)},
        "top_k": top_k,
        "top_k_ids": rank_ordered_ids[:top_k],
        "duplicate_ids_collapsed": duplicate_count,
    }


def _unavailable_pipeline_pair(left_name: str, right_name: str, top_k: int) -> dict[str, Any]:
    return {
        "available": False,
        "left": left_name,
        "right": right_name,
        "top_k": top_k,
        "reason": "one_or_both_outputs_unavailable",
        "added_ids": None,
        "removed_ids": None,
        "survivor_top_k": None,
        "rank_changed": None,
        "rank_changes": None,
        "category_counts": None,
    }


def _pipeline_pair(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    left_name: str,
    right_name: str,
    top_k: int,
) -> dict[str, Any]:
    if left is None or right is None:
        return _unavailable_pipeline_pair(left_name, right_name, top_k)
    left_ids = set(left["ordered_ids"])
    right_ids = set(right["ordered_ids"])
    added = sorted(right_ids - left_ids)
    removed = sorted(left_ids - right_ids)
    left_top = list(left["top_k_ids"])
    right_top = list(right["top_k_ids"])
    top_intersection = sorted(set(left_top) & set(right_top))
    left_ranks = left["ranks"]
    right_ranks = right["ranks"]
    common = sorted(left_ids & right_ids)
    rank_changes = {
        article_id: right_ranks[article_id] - left_ranks[article_id]
        for article_id in common
        if right_ranks[article_id] != left_ranks[article_id]
    }
    left_categories = dict(left["category_counts"])
    right_categories = dict(right["category_counts"])
    categories = sorted(set(left_categories) | set(right_categories))
    category_delta = {
        category: right_categories.get(category, 0) - left_categories.get(category, 0)
        for category in categories
    }
    return {
        "available": True,
        "left": left_name,
        "right": right_name,
        "left_count": len(left_ids),
        "right_count": len(right_ids),
        "intersection_count": len(left_ids & right_ids),
        "added_count": len(added),
        "removed_count": len(removed),
        "added_ids": added,
        "removed_ids": removed,
        "survivor_top_k": {
            "k": top_k,
            "left_ids": left_top,
            "right_ids": right_top,
            "intersection_ids": top_intersection,
            "intersection_count": len(top_intersection),
            "overlap": _rate(len(top_intersection), len(set(left_top) | set(right_top))),
        },
        "rank_changed": bool(rank_changes),
        "rank_changed_count": len(rank_changes),
        "rank_unchanged_count": len(common) - len(rank_changes),
        "rank_change_rate": _rate(len(rank_changes), len(common)),
        "rank_changes": rank_changes,
        "category_counts": {
            "left": {key: left_categories[key] for key in sorted(left_categories)},
            "right": {key: right_categories[key] for key in sorted(right_categories)},
            "delta": category_delta,
        },
    }


def compare_pipeline_views(
    original_output: Any,
    control_output: Any,
    candidate_output: Any,
    repeated_control_output: Any | None = None,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare original/control/candidate pipeline outputs independently.

    ``repeated_control_output`` is optional; when omitted its comparison is
    explicitly unavailable rather than silently treated as equal.  The
    function accepts a single output JSON object, a list of article objects,
    or a mapping containing several output files.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    names_and_values = (
        ("original", original_output),
        ("control", control_output),
        ("candidate", candidate_output),
        ("repeated_control", repeated_control_output),
    )
    views: dict[str, dict[str, Any] | None] = {
        name: None if value is None else pipeline_output_view(value, top_k=top_k)
        for name, value in names_and_values
    }
    return {
        "schema_version": "news-shadow-pipeline-metrics/v1",
        "top_k": top_k,
        "views": views,
        "comparisons": {
            "original_vs_control": _pipeline_pair(
                views["original"], views["control"], left_name="original", right_name="control", top_k=top_k
            ),
            "control_vs_candidate": _pipeline_pair(
                views["control"], views["candidate"], left_name="control", right_name="candidate", top_k=top_k
            ),
            "control_vs_repeated_control": _pipeline_pair(
                views["control"], views["repeated_control"],
                left_name="control", right_name="repeated_control", top_k=top_k,
            ),
        },
    }


# Keep descriptive aliases for callers that already use "outputs" or
# "metrics" terminology.  All aliases are the same pure implementation.
compare_pipeline_outputs = compare_pipeline_views
pipeline_output_metrics = compare_pipeline_views


def compare_decisions(
    records: Sequence[Mapping[str, Any]] | Sequence[str],
    original_decisions: Sequence[Mapping[str, Any]] | None,
    control_decisions: Sequence[Mapping[str, Any]] | None,
    candidate_decisions: Sequence[Mapping[str, Any]] | None,
    *,
    judge_adjudications: Sequence[Mapping[str, Any]] | None = None,
    original_requests: Sequence[Mapping[str, Any]] | None = None,
    control_requests: Sequence[Mapping[str, Any]] | None = None,
    candidate_requests: Sequence[Mapping[str, Any]] | None = None,
    judge_requests: Sequence[Mapping[str, Any]] | None = None,
    output_comparisons: Sequence[Mapping[str, Any]] | None = None,
    budget_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a complete deterministic filter comparison report."""

    input_ids = _ids_from_records(records)
    branches = {
        "original": summarize_branch(original_decisions, input_ids),
        "control": summarize_branch(control_decisions, input_ids),
        "candidate": summarize_branch(candidate_decisions, input_ids),
    }
    return {
        "schema_version": "news-shadow-metrics/v1",
        "quality_basis": "judge_estimated_not_ground_truth",
        "population": {
            "input_count": len(input_ids),
            "input_ids": sorted(input_ids),
        },
        "branches": branches,
        "set_differences": {
            "candidate_vs_control": set_differences(control_decisions, candidate_decisions),
            "control_vs_original": set_differences(original_decisions, control_decisions),
            "candidate_vs_original": set_differences(original_decisions, candidate_decisions),
        },
        "paired": {
            "candidate_vs_control": _paired_agreement(control_decisions, candidate_decisions, input_ids),
            "control_vs_original": _paired_agreement(original_decisions, control_decisions, input_ids),
            "candidate_vs_original": _paired_agreement(original_decisions, candidate_decisions, input_ids),
        },
        "ranking": {
            "candidate_vs_control": rank_comparison(control_decisions, candidate_decisions),
            "control_vs_original": rank_comparison(original_decisions, control_decisions),
            "candidate_vs_original": rank_comparison(original_decisions, candidate_decisions),
        },
        "judge_estimated_quality": {
            "control": judge_estimated_quality(control_decisions, judge_adjudications, input_ids),
            "candidate": judge_estimated_quality(candidate_decisions, judge_adjudications, input_ids),
        },
        "usage": {
            "original": summarize_requests(original_requests),
            "control": summarize_requests(control_requests),
            "candidate": summarize_requests(candidate_requests),
            "judge": summarize_requests(judge_requests),
            "budget": dict(budget_snapshot) if isinstance(budget_snapshot, Mapping) else {},
        },
        "output_comparisons": summarize_output_comparisons(output_comparisons),
    }


compute_metrics = compare_decisions


__all__ = [
    "compare_decisions",
    "compute_metrics",
    "compare_pipeline_outputs",
    "compare_pipeline_views",
    "judge_estimated_quality",
    "pipeline_output_metrics",
    "pipeline_output_view",
    "rank_comparison",
    "set_differences",
    "summarize_branch",
    "summarize_output_comparisons",
    "summarize_requests",
]
