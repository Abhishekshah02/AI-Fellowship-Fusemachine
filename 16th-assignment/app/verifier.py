"""Verification sub-agent.

It is a separate agent rather than another turn of the investigator on purpose:
an agent asked to check its own reasoning tends to ratify it (the
self-verification paradox), and by this point the investigator's context is a
narrative it has already committed to.

The verifier gets a deliberately narrow context -- the complaint, the claims, and
the raw evidence pulled fresh from the case file's evidence store. It never sees
the investigator's trajectory, so it judges whether the claims are supported, not
whether the story sounds coherent.

It uses response_format and no tools, which is also the shape providers accept
(tools + structured output together are rejected by several of them).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import openai as openai_sdk

from .config import settings
from .llm import _retrying, fix_mojibake, openai_client
from .tokens import TokenLedger

log = logging.getLogger("shopassist.verifier")

SYSTEM_PROMPT = """\
You are a claims auditor for ShopAssist support. You are given a customer complaint,
a draft reply, the factual claims it rests on, and the raw evidence that was gathered.

Decide one thing: is every claim supported by the evidence shown?

What the assistant can actually do: look up orders, read the payment ledger, search
policy, and open a support ticket. It CANNOT issue or initiate a refund, cancel or
change an order, move a delivery, or contact a carrier. Any draft claiming to have
done one of those is describing something that did not happen.

Reject the draft if any of these is true:
- a claim states a fact the evidence does not contain (dates, amounts, statuses,
  reference numbers, refund states)
- the draft says it has performed, started or arranged an action the assistant
  cannot perform -- "I have issued the refund", "I have cancelled it"
- the draft gives a timeframe that does not match the policy text in the evidence
  (quote the policy window, do not round it or invent one)
- the evidence shows something material the draft ignores, such as a duplicate
  charge, a failed authorisation, or a missing refund
- the draft answers a different question than the one the customer asked

Do not reject for tone, brevity or style. Do not invent evidence. If the evidence
supports the claims, pass it.

Reply with JSON only: {"passed": true|false, "problems": ["..."], "confidence": 0.0-1.0}
`problems` must be specific and actionable, and empty when passed is true."""

SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["passed", "problems", "confidence"],
    "additionalProperties": False,
}


MAX_EVIDENCE_ITEMS = 4


def _render_evidence(evidence: dict[str, Any]) -> str:
    out = []
    for key, payload in list(evidence.items())[-MAX_EVIDENCE_ITEMS:]:
        if payload is None:
            continue
        out.append(f"--- {key} ---\n{json.dumps(payload, indent=2, default=str)[:1200]}")
    return "\n\n".join(out) or "(no evidence was gathered)"


async def verify(
    *,
    complaint: str,
    answer: str,
    findings: list[str],
    evidence: dict[str, Any],
    ledger: TokenLedger,
) -> dict:
    """Returns {passed, problems, confidence}. A verifier that cannot run must not
    silently wave the answer through, so a failure is reported as a problem."""
    s = settings()
    client = openai_client()
    user = (
        f"CUSTOMER COMPLAINT:\n{complaint}\n\n"
        f"DRAFT REPLY:\n{answer}\n\n"
        f"CLAIMS THE REPLY RESTS ON:\n"
        + ("\n".join(f"- {f}" for f in findings) or "(none stated)")
        + f"\n\nRAW EVIDENCE GATHERED:\n{_render_evidence(evidence)}"
    )

    async def call(structured: bool):
        params = {
            "model": s.verifier_model or s.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            # Reasoning models burn the budget before emitting content; too small
            # a cap returns an empty message that JSON mode then rejects.
            "max_tokens": 1500,
            "temperature": 0.0,  # auditing is not a creative task
            "top_p": 1.0,
            # Honoured by gpt-oss on Groq, ignored elsewhere: this is an audit,
            # not a puzzle, and long reasoning here just eats the rate limit.
            "extra_body": {"reasoning_effort": "low"},
        }
        if structured:
            params["response_format"] = {"type": "json_object"}
        async for attempt in _retrying():
            with attempt:
                return await client.chat.completions.create(**params)

    try:
        try:
            response = await call(structured=True)
        except openai_sdk.BadRequestError as exc:
            log.warning("verifier JSON mode refused (%s), retrying unstructured", exc)
            response = await call(structured=False)
        ledger.record("verifier", response)
        raw = fix_mojibake(response.choices[0].message.content or "")
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return {
            "passed": bool(data.get("passed", False)),
            "problems": [str(p) for p in data.get("problems", [])][:5],
            "confidence": float(data.get("confidence", 0.0)),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("verifier unavailable: %s", exc)
        return {
            "passed": False,
            "problems": [f"verification could not run ({type(exc).__name__}); answer not confirmed"],
            "confidence": 0.0,
            "error": str(exc),
        }
