"""External tools the assistant can call.

TOOLS is the Anthropic tool-definition list; dispatch() executes one by name and
always returns a JSON string, never raises -- a tool failure has to come back to
the model as a readable result, not blow up the request.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from . import rag
from .config import settings
from .schemas import INTENTS

TOOLS = [
    {
        "name": "lookup_order",
        "description": (
            "Fetch one order by its ShopAssist order id (format SA-#####), or all "
            "orders for a customer email. Returns status, tracking, items, refund state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "e.g. SA-10231"},
                "email": {"type": "string", "description": "customer email"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Semantic search over ShopAssist support policy documents. Use it when the "
            "context already provided does not answer the question."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "required": ["query", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_support_ticket",
        "description": (
            "Escalate to a human agent. Only call this when the escalation rules in the "
            "knowledge base are met. Returns the ticket id to give the customer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": INTENTS},
                "summary": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                "order_id": {"type": "string"},
            },
            "required": ["category", "summary", "priority"],
            "additionalProperties": False,
        },
    },
]


@lru_cache
def _orders() -> list[dict]:
    return json.loads(settings().orders_path.read_text(encoding="utf-8"))


def lookup_order(order_id: str | None = None, email: str | None = None) -> dict:
    if order_id:
        hit = next(
            (o for o in _orders() if o["order_id"].upper() == order_id.strip().upper()), None
        )
        return hit or {"error": f"No order {order_id}. Ask the customer to re-check the id."}
    if email:
        found = [o for o in _orders() if o["email"].lower() == email.strip().lower()]
        return {"orders": found} if found else {"error": f"No orders for {email}."}
    return {"error": "Provide either order_id or email."}


def search_knowledge_base(query: str, k: int = 4) -> dict:
    return {"results": rag.search(query, k=k)}


def create_support_ticket(
    category: str, summary: str, priority: str = "normal", order_id: str | None = None
) -> dict:
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "category": category,
        "summary": summary,
        "priority": priority,
        "order_id": order_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = settings().tickets_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ticket) + "\n")
    return ticket


_IMPL = {
    "lookup_order": lookup_order,
    "search_knowledge_base": search_knowledge_base,
    "create_support_ticket": create_support_ticket,
}


async def dispatch(name: str, args: dict) -> str:
    """Run a tool off the event loop and JSON-encode whatever comes back."""
    fn = _IMPL.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool {name}"})
    clean = {k: v for k, v in (args or {}).items() if v is not None}
    try:
        result = await asyncio.to_thread(lambda: fn(**clean))
    except Exception as exc:  # surfaced to the model, which can retry or apologise
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, default=str)


def openai_tools() -> list[dict]:
    """The same three tools in OpenAI function-calling shape, for any
    OpenAI-compatible endpoint (Groq, OpenRouter, Gemini, vLLM, Ollama)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOLS
    ]
