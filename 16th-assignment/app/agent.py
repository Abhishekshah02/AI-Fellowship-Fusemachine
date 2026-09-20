"""Dispute-investigation agent (W16).

Why this cannot be a fixed pipeline: the next lookup depends on what the last
one returned -- a refund complaint where the ledger shows a REFUND event is a
"when will it land" answer, the same complaint where the ledger shows two
CAPTUREs is a duplicate-charge case needing a different policy and a different
outcome, and neither branch is knowable before the ledger is read.

Loop shape (single investigator + a verifier sub-agent):

    while steps < MAX_STEPS:
        model picks one tool
        investigative tool -> record note, clear the previous raw payload, continue
        terminal tool      -> verifier checks the claims against cited evidence
                              pass -> done
                              fail -> feed the problems back, continue (bounded)

Context engineering -- clearing tool results, carried by structured notes:
only the most recent KEEP_RAW payloads stay verbatim in the prompt. Older ones
are replaced by the one-line digest each tool returned, and the full payload
stays in `CaseFile.evidence` where the verifier reads it. See README section (a).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import openai as openai_sdk

from . import agent_tools, verifier
from .config import settings
from .llm import _retrying, openai_client
from .tokens import TokenLedger

log = logging.getLogger("shopassist.agent")

KEEP_RAW = 1  # how many recent tool payloads stay verbatim in the prompt

SYSTEM_PROMPT = """\
You are the dispute investigator for ShopAssist, an online store. A customer has
raised a complaint about money, delivery or an order. Your job is to find out what
actually happened before anyone answers them.

How to work:
- Take ONE action at a time and look at what it returns before choosing the next.
- An order record alone never settles a money dispute. If a charge or refund is in
  question, read the payment ledger.
- When the evidence tells you what happened, find the policy rule that applies to
  it before you answer.
- Ground every claim in something a tool returned. If you cannot, do not claim it.
- If a tool is unavailable or the evidence is missing, escalate. Never fill the gap
  with a guess -- a confident wrong answer about someone's money is the worst
  outcome available to you.

What you can and cannot do. You can look things up and open a ticket. You CANNOT
issue or start a refund, cancel or change an order, reroute a delivery, or contact
the payments team or a carrier. Never tell a customer you have done one of those.
Say what the policy says will happen and who will do it -- "our payments team will
refund the duplicate charge" is true; "I have refunded it" is not. Quote timeframes
exactly as the policy chunk states them; do not round or estimate them.

Ending the investigation -- you must finish by calling exactly one of:
- resolve_case ....... the evidence explains the complaint and you can answer it
- request_information  only the customer can supply what is missing
- escalate_case ...... policy requires a human, or the evidence cannot be obtained

CASE FILE below is your own record of what you have already done. Payloads from
earlier steps have been cleared; their findings are in the notes. Do not repeat a
lookup that is already noted."""


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    note: str
    ok: bool
    payload: Any = None


@dataclass
class CaseFile:
    """Append-only record of the investigation. This is what survives clearing."""

    complaint: str
    steps: list[Step] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    revisions: list[str] = field(default_factory=list)

    def add(self, step: Step) -> None:
        self.steps.append(step)
        self.evidence[f"step{step.n}:{step.tool}"] = step.payload

    def render(self, remaining: int | None = None) -> str:
        lines = [f"COMPLAINT: {self.complaint}", "", "STEPS TAKEN:"]
        if not self.steps:
            lines.append("  (none yet -- this is your first action)")
        for s in self.steps:
            lines.append(f"  {s.n}. {s.note}")
        if remaining is not None:
            lines += ["", f"STEPS REMAINING: {remaining}"]
            if remaining <= 1:
                lines.append(
                    "  This is your LAST action. You must call resolve_case, "
                    "request_information or escalate_case now -- escalate if the "
                    "evidence you have does not settle the complaint."
                )
            elif remaining <= 2:
                lines.append("  Wrap up: gather at most one more thing, then conclude.")
        if self.revisions:
            lines += ["", "VERIFIER REJECTED YOUR PREVIOUS ANSWER:"]
            lines += [f"  - {p}" for p in self.revisions]
            lines.append(
                "  Re-answer with the evidence you ALREADY have: these are almost always "
                "wording problems, not missing facts. Call resolve_case again with the "
                "claims corrected. Only look something up if a problem above says the "
                "evidence itself is missing."
            )
        return "\n".join(lines)

    def cited_evidence(self, citations: list[str]) -> dict[str, Any]:
        """Everything the verifier needs, pulled fresh from the store rather than
        from the investigator's (cleared) context."""
        return dict(self.evidence)


@dataclass
class AgentResult:
    outcome: str                     # resolved | needs_input | escalated | exhausted | failed
    answer: str
    findings: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    iterations: int = 0
    verifier_verdicts: list[dict] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    latency_ms: int = 0
    context_saved_chars: int = 0
    error: str | None = None


def _messages(case: CaseFile, remaining: int | None = None) -> list[dict[str, Any]]:
    """Rebuild the prompt from scratch each iteration.

    This is where tool results are cleared: rather than appending an ever-growing
    transcript of assistant/tool turns, the prompt is the case file plus the last
    KEEP_RAW payloads. Context stays roughly flat as the investigation deepens.
    """
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case.render(remaining)},
    ]
    for step in case.steps[-KEEP_RAW:]:
        if step.payload is not None:
            msgs.append(
                {
                    "role": "user",
                    "content": f"FULL RESULT OF STEP {step.n} ({step.tool}):\n"
                    + json.dumps(step.payload, indent=2, default=str)[:2000],
                }
            )
    msgs.append(
        {"role": "user", "content": "Choose your next action by calling exactly one tool."}
    )
    return msgs


MALFORMED = "__malformed_tool_call__"


async def _choose_action(
    case: CaseFile, ledger: TokenLedger, remaining: int | None = None
) -> tuple[str, dict] | None:
    """One model turn: pick the next tool. No response_format here -- several
    providers reject tools and structured output in the same request."""
    s = settings()
    client = openai_client()
    try:
        async for attempt in _retrying():
            with attempt:
                response = await client.chat.completions.create(
                    model=s.openai_model,
                    messages=_messages(case, remaining),
                    tools=agent_tools.openai_schemas(),
                    tool_choice="auto",
                    max_tokens=1200,
                    temperature=s.temperature,
                    top_p=s.top_p,
                )
    except openai_sdk.BadRequestError as exc:
        # Usually a tool call whose arguments ran past the token ceiling and got
        # cut mid-JSON. That is recoverable: tell the model, let it try again.
        if "tool_use_failed" not in str(exc):
            raise
        log.warning("malformed tool call, asking the model to retry shorter")
        return MALFORMED, {}
    ledger.record("investigator", response)
    choice = response.choices[0].message
    calls = choice.tool_calls or []
    if not calls:
        return None
    call = calls[0]  # one action at a time: the loop reacts to each result
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return call.function.name, args


async def investigate(complaint: str, *, verify: bool = True) -> AgentResult:
    """Run the agentic loop. Bounded by AGENT_MAX_STEPS and AGENT_MAX_REVISIONS."""
    s = settings()
    started = time.perf_counter()
    case = CaseFile(complaint=complaint)
    ledger = TokenLedger()
    verdicts: list[dict] = []
    revisions = 0
    malformed = 0
    seen: dict[tuple[str, str], int] = {}   # (tool, args) -> step that ran it
    repeats = 0
    raw_chars = 0  # what the prompt would have carried without clearing

    result = AgentResult(outcome="exhausted", answer="")

    for step_n in range(1, s.agent_max_steps + 1):
        try:
            action = await _choose_action(case, ledger, s.agent_max_steps - step_n + 1)
        except Exception as exc:  # provider down mid-investigation
            log.warning("agent step %s failed: %s", step_n, exc)
            result.outcome, result.error = "failed", str(exc)
            result.answer = (
                "I could not complete the investigation because the AI service was "
                "unavailable. A human agent will pick this up."
            )
            break

        if action is None:
            log.info("model returned no tool call at step %s, stopping", step_n)
            result.outcome = "exhausted"
            break

        name, args = action
        if name is MALFORMED or name == MALFORMED:
            malformed += 1
            if malformed > 2:
                log.warning("giving up after repeated malformed tool calls")
                break
            case.revisions = [
                "Your last tool call was cut off before it was valid JSON. Keep the "
                "answer under 120 words and call the tool again."
            ]
            metrics_note = "malformed tool call -- asked the model to retry shorter"
            case.add(Step(n=step_n, tool="(malformed)", args={}, note=metrics_note, ok=False))
            continue

        # An identical lookup returns an identical result. Re-running it burns a
        # step and a request to learn nothing, and it is how a loop stalls.
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if name not in agent_tools.TERMINAL and key in seen:
            repeats += 1
            earlier = seen[key]
            log.warning("step %s repeats step %s (%s)", step_n, earlier, name)
            metrics_note = (
                f"{name} REPEATED -- you already ran this at step {earlier} and the "
                f"result has not changed. Do NOT run it again. Either conclude with "
                f"what you have, or take a genuinely different action."
            )
            case.add(Step(n=step_n, tool=name, args=args, note=metrics_note, ok=False))
            if repeats >= 2:
                log.warning("no progress after repeated actions, ending the investigation")
                break
            continue
        seen[key] = step_n

        payload, note, ok = agent_tools.run(name, args)
        raw_chars += len(json.dumps(payload, default=str))
        case.add(Step(n=step_n, tool=name, args=args, note=note, ok=ok, payload=payload))
        log.info("step %s: %s", step_n, note)

        if name not in agent_tools.TERMINAL:
            continue

        # ---- terminal action: verify before it reaches the customer ---------
        if name == "resolve_case" and verify:
            verdict = await verifier.verify(
                complaint=complaint,
                answer=payload.get("answer", ""),
                findings=payload.get("findings", []),
                evidence=case.cited_evidence(payload.get("citations", [])),
                ledger=ledger,
            )
            verdicts.append(verdict)
            if verdict.get("error"):
                # The auditor is down. We hold an unverified answer about someone's
                # money, so hand it to a human rather than ship it or retry blindly.
                escalation, _, _ = agent_tools.run(
                    "escalate_case",
                    {
                        "summary": f"Answer could not be verified (verifier unavailable: "
                        f"{verdict['error'][:150]}). Complaint: {complaint[:200]}",
                        "priority": "high",
                        "category": "CONTACT",
                    },
                )
                result.outcome = "escalated"
                result.answer = (
                    "I found what looks like the explanation, but I could not get it "
                    "double-checked before replying, so a human agent will confirm it with "
                    f"you (ticket {escalation.get('ticket_id', 'n/a')})."
                )
                result.findings = payload.get("findings", [])
                result.citations = payload.get("citations", [])
                result.iterations = step_n
                break
            if not verdict["passed"] and revisions < s.agent_max_revisions:
                revisions += 1
                case.revisions = verdict["problems"]
                case.steps.pop()  # the rejected answer is not evidence
                log.info("verifier rejected the answer, revision %s", revisions)
                continue
            if not verdict["passed"]:
                # Out of revisions with an answer we cannot stand behind.
                escalation, note, _ = agent_tools.run(
                    "escalate_case",
                    {
                        "summary": f"Investigation could not be verified: "
                        f"{'; '.join(verdict['problems'])[:300]}",
                        "priority": "high",
                        "category": "CONTACT",
                    },
                )
                result.outcome = "escalated"
                result.answer = (
                    "I could not confirm an answer I am confident in, so I have passed this "
                    f"to a human agent (ticket {escalation.get('ticket_id', 'n/a')})."
                )
                result.iterations = step_n
                break

        if name == "resolve_case":
            result.outcome = "resolved"
            result.answer = payload.get("answer", "")
            result.findings = payload.get("findings", [])
            result.citations = payload.get("citations", [])
        elif name == "request_information":
            result.outcome = "needs_input"
            result.answer = payload.get("question", "")
        else:
            result.outcome = "escalated"
            result.answer = (
                "I have opened a ticket for a human agent to take this on "
                f"(ticket {payload.get('ticket_id', 'n/a')})."
            )
        result.iterations = step_n
        break

    if result.outcome == "exhausted" and not result.answer:
        # Step budget spent without a conclusion: say so, do not invent one.
        escalation, _, _ = agent_tools.run(
            "escalate_case",
            {
                "summary": f"Investigation ended without a conclusion (step limit "
                f"{s.agent_max_steps} or no further progress). "
                f"Complaint: {complaint[:200]}",
                "priority": "high",
                "category": "CONTACT",
            },
        )
        result.answer = (
            "I could not get to the bottom of this within my investigation limit, so I have "
            f"handed it to a human agent (ticket {escalation.get('ticket_id', 'n/a')})."
        )
        result.iterations = len(case.steps)

    result.steps = [
        {"n": s_.n, "tool": s_.tool, "args": s_.args, "note": s_.note, "ok": s_.ok}
        for s_ in case.steps
    ]
    result.verifier_verdicts = verdicts
    result.tokens = ledger.summary()
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    # Cleared payloads never re-entered the prompt on later iterations.
    kept = sum(
        len(json.dumps(s_.payload, default=str)) for s_ in case.steps[-KEEP_RAW:] if s_.payload
    )
    result.context_saved_chars = max(0, raw_chars - kept)
    return result
