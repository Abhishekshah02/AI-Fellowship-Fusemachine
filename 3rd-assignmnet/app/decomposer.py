"""Task 2: turn a natural-language question into a structured plan.

Uses the LLM with a JSON-mode prompt; falls back to a minimal rule-based
plan if the LLM fails or returns invalid JSON, so the rest of the
pipeline never crashes on this step.
"""

from typing import Any

from .llm_client import MissingAPIKeyError, chat_json
from .logger import log_event
from .prompts import load
from .schema_info import SCHEMA_DESCRIPTION, TABLE_COLUMNS

_EMPTY_PLAN: dict[str, Any] = {
    "intent": "",
    "tables": [],
    "columns": [],
    "filters": [],
    "joins": [],
    "aggregations": [],
    "group_by": [],
}


def decompose(question: str) -> dict[str, Any]:
    """Best-effort structured plan. Always returns a dict with the expected keys."""
    template = load("decompose")
    prompt = template.format(schema=SCHEMA_DESCRIPTION, question=question)
    try:
        plan = chat_json(
            system="You break SQL questions into JSON plans.",
            user=prompt,
            purpose="decompose",
        )
    except MissingAPIKeyError:
        log_event("decompose_fallback", reason="no_api_key", question=question)
        return _rule_based(question)
    except Exception as exc:  # noqa: BLE001 - any LLM error -> fallback
        log_event("decompose_fallback", reason=str(exc), question=question)
        return _rule_based(question)

    # Normalize: make sure every expected key exists, default to [].
    normalized = {**_EMPTY_PLAN, **{k: plan.get(k, _EMPTY_PLAN[k]) for k in _EMPTY_PLAN}}
    log_event("decomposed", question=question, plan=normalized)
    return normalized


def _rule_based(question: str) -> dict[str, Any]:
    """Cheap keyword-based decomposition used when the LLM is unavailable.
    Just identifies likely tables and aggregation intent — enough to log
    something useful and let the SQL generator try."""
    q = question.lower()
    tables = [t for t in TABLE_COLUMNS if t in q or t.rstrip("s") in q]

    aggs = []
    for word, agg in [
        ("count", "COUNT(*)"), ("total", "SUM"), ("sum", "SUM"),
        ("average", "AVG"), ("avg", "AVG"), ("max", "MAX"), ("min", "MIN"),
        ("number of", "COUNT(*)"),
    ]:
        if word in q:
            aggs.append(agg)

    intent = "Aggregate" if aggs else "Retrieve"
    return {
        **_EMPTY_PLAN,
        "intent": f"{intent} from {', '.join(tables) or '?'}",
        "tables": tables,
        "aggregations": list(dict.fromkeys(aggs)),
    }
