"""Tools for the dispute-investigation agent.

Split into two kinds, because the loop treats them differently:

  * investigative -- gather evidence, the loop continues
  * terminal      -- resolve / ask the customer / escalate, the loop stops

Each tool returns (payload, note). `payload` is the full evidence and is stored
in the case file's evidence store; `note` is a one-line structured digest that
stays in the model's context after the payload is cleared. Writing the digest
here rather than asking the model for one keeps it deterministic and free.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from . import rag
from .config import settings
from .schemas import INTENTS
from .tools import create_support_ticket, lookup_order

# --------------------------------------------------------------------------- #
# Fault injection (W16 additional requirement 3). Off unless explicitly asked.
# --------------------------------------------------------------------------- #
FAULTS: set[str] = set()


class ToolUnavailable(RuntimeError):
    """Raised by an injected fault, surfaced to the model as a tool error."""


@lru_cache
def _payments() -> dict[str, list[dict]]:
    path = settings().payments_path
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Investigative tools
# --------------------------------------------------------------------------- #
def inspect_order(order_id: str) -> tuple[dict, str]:
    order = lookup_order(order_id=order_id)
    if "error" in order:
        return order, f"inspect_order({order_id}) -> NOT FOUND"
    refund = order.get("refund")
    note = (
        f"inspect_order({order_id}) -> status={order['status']}, "
        f"placed={order['placed_at']}, total=${order['total_usd']}, "
        f"payment={order['payment_method']}, "
        f"refund={refund['status'] + ' $' + str(refund['amount_usd']) if refund else 'none recorded'}"
    )
    return order, note


def inspect_payments(order_id: str) -> tuple[dict, str]:
    """The ledger of authorisations, captures and refunds for one order."""
    if "payments_unavailable" in FAULTS:
        raise ToolUnavailable("payment ledger service is unreachable (injected fault)")

    events = _payments().get(order_id.strip().upper())
    if not events:
        return (
            {"error": f"no payment ledger for {order_id}"},
            f"inspect_payments({order_id}) -> NO LEDGER",
        )
    captures = [e for e in events if e["event"] == "CAPTURE"]
    refunds = [e for e in events if e["event"] == "REFUND"]
    note = (
        f"inspect_payments({order_id}) -> {len(events)} events "
        f"[{', '.join(e['event'] for e in events)}]; "
        f"{len(captures)} capture(s) totalling ${sum(e['amount_usd'] for e in captures):.2f}, "
        f"{len(refunds)} refund(s) totalling ${sum(e['amount_usd'] for e in refunds):.2f}"
    )
    return {"order_id": order_id, "events": events}, note


def search_policy(query: str, k: int = 3) -> tuple[dict, str]:
    hits = rag.search(query, k=k)
    ids = [h["id"] for h in hits]
    # Cap each chunk: the agent needs the rule, not the whole document, and full
    # chunks are the single largest thing that would sit in the loop's context.
    hits = [{**h, "text": h["text"][:700]} for h in hits]
    return (
        {"query": query, "results": hits},
        f"search_policy({query!r}) -> {len(hits)} chunks: {', '.join(ids) or 'none'}",
    )


# --------------------------------------------------------------------------- #
# Terminal tools -- calling one of these ends the investigation
# --------------------------------------------------------------------------- #
TERMINAL = {"resolve_case", "request_information", "escalate_case"}


def resolve_case(answer: str, findings: list[str], citations: list[str]) -> tuple[dict, str]:
    return (
        {"answer": answer, "findings": findings, "citations": citations},
        f"resolve_case -> {len(findings)} finding(s), {len(citations)} citation(s)",
    )


def request_information(question: str, reason: str = "") -> tuple[dict, str]:
    return (
        {"question": question, "reason": reason},
        f"request_information -> asked customer: {question}",
    )


def escalate_case(summary: str, priority: str = "high", category: str = "PAYMENT",
                  order_id: str | None = None) -> tuple[dict, str]:
    ticket = create_support_ticket(category, summary, priority, order_id)
    return ticket, f"escalate_case -> {ticket['ticket_id']} ({priority})"


IMPL = {
    "inspect_order": inspect_order,
    "inspect_payments": inspect_payments,
    "search_policy": search_policy,
    "resolve_case": resolve_case,
    "request_information": request_information,
    "escalate_case": escalate_case,
}

SCHEMAS: list[dict] = [
    {
        "name": "inspect_order",
        "description": "Fetch one order: status, dates, totals, payment method and any recorded refund.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. SA-10231"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_payments",
        "description": (
            "The payment ledger for one order: authorisations, captures, refunds and return "
            "scans with dates and references. Use it whenever money is disputed -- an order "
            "record alone cannot tell you whether a charge was taken twice or a refund issued."
        ),
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policy",
        "description": "Semantic search over ShopAssist support policy. Use it to find the rule that applies to what you found.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve_case",
        "description": (
            "END the investigation with an answer for the customer. Only call this when the "
            "evidence you gathered actually explains their complaint. Every finding must be "
            "traceable to a tool result or a policy chunk you retrieved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Reply to show the customer."},
                "findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The factual claims your answer rests on, one per line.",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence ids: policy chunk ids, order ids, payment references.",
                },
            },
            "required": ["answer", "findings", "citations"],
            "additionalProperties": False,
        },
    },
    {
        "name": "request_information",
        "description": (
            "END the investigation by asking the customer for something only they can supply "
            "(an order id, which charge they mean). Do not use it for anything a tool can answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_case",
        "description": (
            "END the investigation by opening a ticket for a human. Use it when policy requires "
            "escalation, or when the evidence you need is unavailable -- never guess instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                "category": {"type": "string", "enum": INTENTS},
                "order_id": {"type": "string"},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


def openai_schemas() -> list[dict]:
    return [{"type": "function", "function": s} for s in SCHEMAS]


def run(name: str, args: dict) -> tuple[Any, str, bool]:
    """Execute a tool. Returns (payload, note, ok). Never raises: an unavailable
    tool has to reach the model as a result it can reason about, otherwise the
    agent cannot tell 'no evidence' from 'evidence says no'."""
    fn = IMPL.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}, f"{name} -> UNKNOWN TOOL", False
    try:
        payload, note = fn(**(args or {}))
        return payload, note, "error" not in (payload if isinstance(payload, dict) else {})
    except ToolUnavailable as exc:
        return {"error": str(exc), "retryable": False}, f"{name} -> UNAVAILABLE: {exc}", False
    except TypeError as exc:
        return {"error": f"bad arguments: {exc}"}, f"{name} -> BAD ARGS: {exc}", False
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, never raised
        return {"error": f"{type(exc).__name__}: {exc}"}, f"{name} -> ERROR: {exc}", False
