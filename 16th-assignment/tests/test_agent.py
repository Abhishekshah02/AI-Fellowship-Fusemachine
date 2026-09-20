"""Offline tests for the W16 agentic loop. No API key, no network: the provider
is stubbed, so these assert the loop's control flow rather than model quality."""
from __future__ import annotations

import json

import pytest

from app import agent, agent_tools, verifier
from app.agent import CaseFile, Step
from app.tokens import TokenLedger
from evaluation.harness import validate_args


# --------------------------------------------------------------------------- #
# Stub provider
# --------------------------------------------------------------------------- #
class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, name, args):
        self.id, self.type = f"call_{name}", "function"
        self.function = _Fn(name, json.dumps(args))


class _Msg:
    def __init__(self, tool_calls=None, content=None):
        self.tool_calls, self.content, self.role = tool_calls, content, "assistant"


class _Usage:
    prompt_tokens, completion_tokens = 100, 20


class _Completion:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]
        self.usage = _Usage()


class _Script:
    """Replays a fixed sequence of model turns, recording what it was sent."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    async def create(self, **params):
        self.seen.append(params)
        turn = self.turns.pop(0) if self.turns else ("resolve_case", {
            "answer": "fallback", "findings": [], "citations": []})
        if isinstance(turn, str):          # a plain JSON reply (the verifier)
            return _Completion(_Msg(content=turn))
        name, args = turn
        return _Completion(_Msg(tool_calls=[_ToolCall(name, args)]))


def _client(turns):
    """One stub instance, reused across iterations -- the loop calls
    openai_client() every turn, so a factory would restart the script each time."""
    return type("Client", (), {"chat": type("Chat", (), {"completions": _Script(turns)})()})()


def _stub(monkeypatch, module, turns):
    client = _client(turns)
    monkeypatch.setattr(module, "openai_client", lambda: client)
    return client


RESOLVE = ("resolve_case", {
    "answer": "Two captures totalling $379.00 were taken; our payments team will refund one.",
    "findings": ["SA-10231 shows two CAPTURE events"],
    "citations": ["payments_and_invoices.md#4"],
})
PASS = json.dumps({"passed": True, "problems": [], "confidence": 0.95})
FAIL = json.dumps({"passed": False, "problems": ["claim X is unsupported"], "confidence": 0.1})


# --------------------------------------------------------------------------- #
# Case file / context clearing
# --------------------------------------------------------------------------- #
def test_case_file_keeps_notes_and_clears_payloads():
    case = CaseFile(complaint="charged twice")
    for n in range(1, 4):
        case.add(Step(n=n, tool="inspect_payments", args={}, note=f"note {n}", ok=True,
                      payload={"big": "x" * 5000}))

    rendered = case.render()
    assert "note 1" in rendered and "note 3" in rendered      # notes survive
    assert "xxxxx" not in rendered                            # payloads do not

    msgs = agent._messages(case)
    raw = "".join(m["content"] for m in msgs)
    # only the most recent payload is sent verbatim
    assert raw.count("big") == agent.KEEP_RAW
    assert len(case.evidence) == 3                            # all of it still retrievable


def test_verifier_rejection_is_shown_back_to_the_investigator():
    case = CaseFile(complaint="c")
    case.revisions = ["you claimed a refund was issued; the ledger shows none"]
    assert "ledger shows none" in case.render()


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_payment_ledger_note_summarises_without_the_payload():
    payload, note = agent_tools.inspect_payments("SA-10231")
    assert len(payload["events"]) == 3
    assert "2 capture(s) totalling $379.00" in note
    assert len(note) < 200                                    # the note is the cheap part


def test_tools_never_raise_and_report_failure_to_the_model():
    payload, note, ok = agent_tools.run("inspect_order", {"order_id": "SA-99999"})
    assert not ok and "error" in payload
    payload, note, ok = agent_tools.run("nonexistent_tool", {})
    assert not ok and "unknown tool" in payload["error"]
    payload, note, ok = agent_tools.run("inspect_order", {"wrong_arg": 1})
    assert not ok and "bad arguments" in payload["error"]


def test_injected_fault_surfaces_as_an_unavailable_tool():
    agent_tools.FAULTS.add("payments_unavailable")
    try:
        payload, note, ok = agent_tools.run("inspect_payments", {"order_id": "SA-10231"})
        assert not ok and "unreachable" in payload["error"] and "UNAVAILABLE" in note
    finally:
        agent_tools.FAULTS.discard("payments_unavailable")
    # and it recovers once the fault is cleared
    _, _, ok = agent_tools.run("inspect_payments", {"order_id": "SA-10231"})
    assert ok


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_loop_runs_multiple_iterations_and_stops_on_a_terminal_action(monkeypatch):
    turns = [
        ("inspect_order", {"order_id": "SA-10231"}),
        ("inspect_payments", {"order_id": "SA-10231"}),
        ("search_policy", {"query": "duplicate charge"}),
        RESOLVE,
        PASS,
    ]
    _stub(monkeypatch, agent, turns)
    _stub(monkeypatch, verifier, [PASS])

    result = await agent.investigate("I was charged twice for SA-10231.")

    assert result.outcome == "resolved"
    assert result.iterations == 4                             # more than one iteration
    assert [s["tool"] for s in result.steps] == [
        "inspect_order", "inspect_payments", "search_policy", "resolve_case"
    ]
    assert result.verifier_verdicts[0]["passed"]
    assert result.tokens["total_tokens"] > 0
    assert result.context_saved_chars > 0                     # clearing actually happened


@pytest.mark.asyncio
async def test_failed_verification_sends_the_agent_back_then_escalates(monkeypatch):
    _stub(monkeypatch, agent, [RESOLVE] * 6)
    _stub(monkeypatch, verifier, [FAIL] * 6)

    result = await agent.investigate("refund missing")

    # rejected, revised, rejected again, then handed to a human rather than sent
    assert result.outcome == "escalated"
    assert len(result.verifier_verdicts) == 3                 # 1 + AGENT_MAX_REVISIONS
    assert all(not v["passed"] for v in result.verifier_verdicts)
    assert "human" in result.answer.lower()


@pytest.mark.asyncio
async def test_unverifiable_answer_is_escalated_not_shipped(monkeypatch):
    """A verifier that cannot run must not count as approval."""
    _stub(monkeypatch, agent, [RESOLVE])

    async def broken(**_kwargs):
        return {"passed": False, "problems": ["verifier down"], "confidence": 0.0,
                "error": "ConnectionError"}

    monkeypatch.setattr(verifier, "verify", broken)
    result = await agent.investigate("refund missing")

    assert result.outcome == "escalated"
    assert "could not get it double-checked" in result.answer


@pytest.mark.asyncio
async def test_step_budget_is_a_hard_stop(monkeypatch):
    """An agent that never terminates must be cut off and escalated, not looped."""
    from app.config import settings

    monkeypatch.setattr(settings(), "agent_max_steps", 3)
    _stub(monkeypatch, agent, [("inspect_order", {"order_id": "SA-10231"})] * 10)
    result = await agent.investigate("goes nowhere")

    assert result.iterations == 3
    assert len(result.steps) == 3
    assert "human agent" in result.answer.lower()


@pytest.mark.asyncio
async def test_request_information_ends_the_loop_in_one_step(monkeypatch):
    _stub(monkeypatch, agent, [("request_information", {"question": "Which order id is this about?"})])
    result = await agent.investigate("you overcharged me")

    assert result.outcome == "needs_input" and result.iterations == 1
    assert "order id" in result.answer.lower()


# --------------------------------------------------------------------------- #
# Verifier + accounting
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_verifier_sees_evidence_but_never_the_trajectory(monkeypatch):
    script = _Script([PASS])
    monkeypatch.setattr(
        verifier, "openai_client",
        lambda: type("C", (), {"chat": type("Ch", (), {"completions": script})()})(),
    )
    verdict = await verifier.verify(
        complaint="charged twice",
        answer="two captures were taken",
        findings=["SA-10231 has two CAPTURE events"],
        evidence={"step2:inspect_payments": {"events": [{"event": "CAPTURE"}]}},
        ledger=TokenLedger(),
    )
    assert verdict["passed"]
    sent = "".join(m["content"] for m in script.seen[0]["messages"])
    assert "CAPTURE" in sent                    # the raw evidence is there
    assert "inspect_order(" not in sent         # the trajectory notes are not
    assert "tools" not in script.seen[0]        # audits with no tools available


def test_token_ledger_splits_by_role():
    ledger = TokenLedger()
    ledger.record("investigator", _Completion(_Msg()))
    ledger.record("investigator", _Completion(_Msg()))
    ledger.record("verifier", _Completion(_Msg()))
    summary = ledger.summary()

    assert summary["total_tokens"] == 360
    assert summary["by_role"]["investigator"]["calls"] == 2
    assert summary["by_role"]["verifier"]["total_tokens"] == 120


def test_ledger_survives_a_response_with_no_usage_block():
    ledger = TokenLedger()
    ledger.record("investigator", type("R", (), {"usage": None})())
    assert ledger.calls == 1 and ledger.total == 0


# --------------------------------------------------------------------------- #
# Harness scoring
# --------------------------------------------------------------------------- #
def test_harness_validates_tool_arguments_against_the_real_schemas():
    assert validate_args("inspect_order", {"order_id": "SA-1"}) == []
    assert "missing required" in validate_args("inspect_order", {})[0]
    assert "unexpected" in validate_args("inspect_order", {"order_id": "x", "junk": 1})[0]
    assert "should be integer" in validate_args("search_policy", {"query": "x", "k": "3"})[0]
    assert "not one of" in validate_args("escalate_case", {"summary": "s", "priority": "nope"})[0]
    assert "unknown tool" in validate_args("teleport", {})[0]


def test_harness_failure_taxonomy():
    from evaluation.harness import classify

    case = {"max_steps": 5, "expect_outcomes": ["resolved"], "require_tools": ["inspect_payments"]}
    ok = {"all_ok": True, "outcome": True, "required_tools": True, "grounded": True}

    class R:
        outcome, error, steps = "resolved", None, []

    assert classify(case, R(), ok) == (None, None)

    R.outcome = "exhausted"
    assert classify(case, R(), ok)[0] == "hard"

    # answer produced, but the required evidence was never gathered
    R.outcome = "resolved"
    bad = {"all_ok": False, "outcome": False, "required_tools": False, "grounded": False}
    assert classify(case, R(), bad)[0] == "cascading_soft"

    # answer produced from good evidence, but the final claim is wrong
    bad2 = {"all_ok": False, "outcome": True, "required_tools": True, "grounded": False}
    R.steps = [{"tool": "inspect_payments", "ok": True}]
    assert classify(case, R(), bad2)[0] == "soft"


@pytest.mark.asyncio
async def test_repeated_identical_lookups_are_blocked_not_re_executed(monkeypatch):
    """A model that asks the same question twice gets told, not obeyed: an
    identical call returns an identical result and only burns the budget."""
    same = ("inspect_payments", {"order_id": "SA-10231"})
    _stub(monkeypatch, agent, [same, same, same, same])
    _stub(monkeypatch, verifier, [PASS])

    result = await agent.investigate("charged twice")

    notes = [s["note"] for s in result.steps]
    assert "REPEATED" in notes[1]                     # second attempt was refused
    # "exhausted" rather than "escalated" on purpose: the agent never reached a
    # conclusion, and the failure log should not confuse that with choosing to
    # escalate. The customer still gets a ticket rather than silence.
    assert result.outcome == "exhausted"
    assert "human agent" in result.answer.lower()
    assert result.iterations <= 4                     # and it stopped quickly


# --------------------------------------------------------------------------- #
# HTTP surface
# --------------------------------------------------------------------------- #
def test_investigate_endpoint_returns_the_whole_trajectory(monkeypatch):
    """The endpoint must expose the trajectory and the token cost, not just the
    answer -- that is what makes the loop inspectable from the UI."""
    from fastapi.testclient import TestClient

    from app import main
    from app.agent import AgentResult

    async def fake(complaint, *, verify=True):
        return AgentResult(
            outcome="resolved",
            answer="two captures were taken",
            findings=["SA-10231 shows two CAPTURE events"],
            citations=["payments_and_invoices.md#4"],
            steps=[{"n": 1, "tool": "inspect_payments", "args": {}, "note": "n", "ok": True}],
            iterations=1,
            verifier_verdicts=[{"passed": True, "problems": [], "confidence": 0.9}],
            tokens={"total_tokens": 1234, "by_role": {}},
            latency_ms=42,
            context_saved_chars=512,
        )

    monkeypatch.setattr(main.agent, "investigate", fake)

    with TestClient(main.app) as client:
        r = client.post("/investigate", json={"complaint": "charged twice for SA-10231"})

    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "resolved"
    assert body["steps"][0]["tool"] == "inspect_payments"
    assert body["tokens"]["total_tokens"] == 1234
    assert body["verifier_verdicts"][0]["passed"] is True
    assert r.headers["X-Agent-Iterations"] == "1"


def test_investigate_rejects_an_empty_complaint():
    from fastapi.testclient import TestClient

    from app import main

    with TestClient(main.app) as client:
        assert client.post("/investigate", json={"complaint": ""}).status_code == 422
