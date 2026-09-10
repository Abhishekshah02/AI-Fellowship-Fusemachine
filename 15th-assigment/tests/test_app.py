"""Offline tests: no API key, no network, no vector DB required except for the
one ingestion test. Run with: pytest -q"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app import llm, rag, tools
from app.reliability import Metrics, RateLimiter, cache_key
from app.schemas import AssistantAnswer, json_schema


def test_json_schema_is_closed_and_complete():
    schema = json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "REFUND" in schema["properties"]["intent"]["enum"]
    assert "$defs" not in schema


def test_chunking_splits_on_headings_and_windows_long_sections():
    text = "# A\n" + "a" * 50 + "\n# B\n" + "b" * 2500
    chunks = rag.chunk(text, size=900, overlap=150)
    assert chunks[0].startswith("# A")
    assert len(chunks) >= 4                       # B was windowed
    assert all(len(c) <= 900 for c in chunks)
    # overlap really overlaps, so a fact on a boundary is not lost
    assert chunks[2][-100:] in chunks[1] + chunks[2]


def test_lookup_order_finds_by_id_and_by_email():
    assert tools.lookup_order(order_id="sa-10231")["status"] == "SHIPPED"
    assert len(tools.lookup_order(email="ava.patel@example.com")["orders"]) == 2
    assert "error" in tools.lookup_order(order_id="SA-99999")


def test_dispatch_never_raises_and_always_returns_json():
    out = asyncio.run(tools.dispatch("lookup_order", {"order_id": None}))
    assert "error" in json.loads(out)
    assert "unknown tool" in json.loads(asyncio.run(tools.dispatch("nope", {})))["error"]


def test_create_ticket_writes_a_record(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings(), "tickets_path", tmp_path / "t.jsonl")
    ticket = tools.create_support_ticket("REFUND", "duplicate charge", "urgent", "SA-10231")
    assert ticket["ticket_id"].startswith("TCK-")
    assert json.loads((tmp_path / "t.jsonl").read_text())["priority"] == "urgent"


def test_rate_limiter_allows_burst_then_blocks():
    limiter = RateLimiter(per_minute=3)
    assert asyncio.run(limiter.take()) and asyncio.run(limiter.take())
    assert asyncio.run(limiter.take())
    assert asyncio.run(limiter.take()) is False   # bucket empty
    assert limiter.retry_after() >= 1


def test_cache_key_ignores_case_and_padding_but_not_history():
    assert cache_key(" Where is my ORDER? ", []) == cache_key("where is my order?", [])
    assert cache_key("hi", []) != cache_key("hi", [{"role": "user", "content": "x"}])


def test_metrics_percentiles():
    m = Metrics()
    for v in range(1, 101):
        m.observe(float(v))
    m.bump("requests", 3)
    snap = m.snapshot()
    assert snap["requests"] == 3 and 45 <= snap["latency_ms_p50"] <= 55 and snap["samples"] == 100


def test_parse_answer_recovers_json_wrapped_in_prose():
    payload = {
        "answer": "Refunds take 5-7 business days.",
        "intent": "REFUND",
        "confidence": 0.9,
        "citations": ["refunds_and_returns.md#2"],
        "actions": [],
        "escalate": False,
    }
    wrapped = f"Sure! Here you go:\n```json\n{json.dumps(payload)}\n```"
    assert llm.parse_answer(wrapped).intent == "REFUND"
    assert llm.parse_answer("not json at all") is None


def test_sampling_capability_matrix():
    assert llm.supports_sampling("claude-opus-5") is False
    assert llm.supports_sampling("claude-sonnet-5") is False
    assert llm.supports_sampling("claude-haiku-4-5") is True
    assert llm.supports_sampling("mistralai/Mistral-7B-Instruct-v0.3") is True


def test_degraded_tier_returns_valid_answer_and_escalates():
    hits = [{"id": "refunds_and_returns.md#2", "text": "Refunds take 5-7 business days."}]
    answer = llm._degraded(hits, "REFUND")
    assert isinstance(answer, AssistantAnswer)
    assert answer.escalate and answer.citations == ["refunds_and_returns.md#2"]


@pytest.mark.asyncio
async def test_generate_falls_through_to_degraded_when_no_provider(monkeypatch):
    """No API key and no local server: the request must still return an answer."""
    monkeypatch.setattr(llm, "_openai", _boom)
    answer, provider, _, reason = await llm.generate("hi", [], [], "", None)
    assert provider == "degraded" and reason and isinstance(answer, AssistantAnswer)


async def _boom(*_args, **_kwargs):
    raise ConnectionError("no OpenAI-compatible endpoint running")


def test_onnx_router_when_the_artifact_is_present():
    from app import router

    if not router.available():
        pytest.skip("no models/router/model.onnx -- run scripts/train_router.py")
    assert router.classify("I want a refund for my order")[0] == "REFUND"
    assert router.classify("zzz qqq") is None  # below the confidence floor


# --------------------------------------------------------------------------- #
# The tool-calling loop, driven by a stub client (no API key, no network).
# --------------------------------------------------------------------------- #
class _Block(dict):
    """Content block that answers to attribute access like the SDK's models do."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Reply:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = None


class _StubMessages:
    """First call asks for a tool, second call returns the final JSON."""

    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **params):
        self.calls.append(params)
        if len(self.calls) == 1:
            return _Reply(
                [_Block(type="tool_use", id="tu_1", name="lookup_order",
                        input={"order_id": "SA-10231"})],
                stop_reason="tool_use",
            )
        payload = {
            "answer": "It shipped with DHL, tracking JD0002210934.",
            "intent": "ORDER",
            "confidence": 0.95,
            "citations": ["shipping_and_delivery.md#1"],
            "actions": [],
            "escalate": False,
        }
        return _Reply([_Block(type="text", text=json.dumps(payload))])


class _StubClient:
    def __init__(self):
        self.messages = _StubMessages()


@pytest.mark.asyncio
async def test_tool_loop_executes_the_tool_and_returns_schema_valid_json(monkeypatch):
    from app.config import settings

    stub = _StubClient()
    monkeypatch.setattr(llm, "anthropic_client", lambda: stub)
    monkeypatch.setattr(settings(), "provider", "claude")

    answer, provider, model, reason = await llm.generate(
        "Where is order SA-10231?", [], [], "context here", "ORDER"
    )

    assert provider == "claude" and reason is None
    assert answer.actions == ["lookup_order"]          # the tool really ran
    assert answer.intent == "ORDER"

    first, second = stub.messages.calls
    # request shape: constrained JSON output + tools declared
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert {t["name"] for t in first["tools"]} == {
        "lookup_order", "search_knowledge_base", "create_support_ticket"
    }
    # opus-5 must not be sent temperature/top_p
    assert "temperature" not in first and first["output_config"]["effort"] == "medium"
    # the tool result was fed back as one user message carrying real order data
    tool_result = second["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1"
    assert "JD0002210934" in tool_result["content"]


@pytest.mark.asyncio
async def test_sampling_params_are_sent_to_models_that_accept_them(monkeypatch):
    from app.config import settings

    stub = _StubClient()
    monkeypatch.setattr(llm, "anthropic_client", lambda: stub)
    monkeypatch.setattr(settings(), "provider", "claude")
    monkeypatch.setattr(settings(), "primary_model", "claude-haiku-4-5")

    await llm.generate("hi", [], [], "", None)
    first = stub.messages.calls[0]
    assert first["temperature"] == settings().temperature
    assert first["top_p"] == settings().top_p
    assert "effort" not in first["output_config"]


# --------------------------------------------------------------------------- #
# The OpenAI-compatible tool loop (Groq / OpenRouter / Gemini / vLLM), stubbed.
# --------------------------------------------------------------------------- #
class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, id_, name, arguments):
        self.id, self.type, self.function = id_, "function", _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls, self.role = content, tool_calls, "assistant"

    def model_dump(self, **_kw):
        return {"role": "assistant", "content": self.content, "tool_calls": self.tool_calls}


class _Completion:
    def __init__(self, message):
        self.choices = [type("C", (), {"message": message})()]


class _StubCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **params):
        self.calls.append(params)
        if len(self.calls) == 1:
            return _Completion(
                _Msg(tool_calls=[_ToolCall("call_1", "lookup_order",
                                           '{"order_id": "SA-10099"}')])
            )
        return _Completion(_Msg(content=json.dumps({
            "answer": "Your refund was scanned on 5 Sep and is still pending.",
            "intent": "REFUND",
            "confidence": 0.88,
            "citations": ["refunds_and_returns.md#3"],
            "actions": [],
            "escalate": True,
        })))


class _StubOpenAI:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _StubCompletions()})()


@pytest.mark.asyncio
async def test_openai_compatible_provider_runs_tools_and_returns_valid_json(monkeypatch):
    from app.config import settings

    stub = _StubOpenAI()
    monkeypatch.setattr(llm, "openai_client", lambda: stub)
    monkeypatch.setattr(settings(), "provider", "openai")

    answer, provider, model, _reason = await llm.generate(
        "Where is my refund for SA-10099?", [], [], "context", "REFUND"
    )

    assert provider == "openai"                    # free model is primary now
    assert answer.actions == ["lookup_order"] and answer.escalate

    calls = stub.chat.completions.calls
    first, first_followup, final = calls[0], calls[1], calls[-1]

    # phase 1 offers tools and never a response_format (providers reject both)
    assert {t["function"]["name"] for t in first["tools"]} == {
        "lookup_order", "search_knowledge_base", "create_support_ticket"
    }
    assert "response_format" not in first
    # open models take the sampling knobs the assignment asks us to tune
    assert first["temperature"] == settings().temperature
    assert first["top_p"] == settings().top_p

    # phase 2 asks for the schema and withdraws the tools
    assert "tools" not in final
    assert final["response_format"]["type"] in ("json_schema", "json_object")

    # phase 1 fed the result back as a tool-role message...
    tool_msgs = [m for m in first_followup["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "call_1"
    # ...and phase 2 restated it as plain text on a transcript with no tool calls,
    # which is what keeps json_schema usable on providers like Groq
    assert not any(m.get("role") == "tool" for m in final["messages"])
    carried = final["messages"][-1]["content"]
    assert carried.startswith("TOOL RESULTS") and "lookup_order" in carried
    assert "REFUNDED" in carried or "PENDING" in carried


@pytest.mark.asyncio
async def test_response_format_steps_down_when_the_provider_rejects_it(monkeypatch):
    """A provider that does not support json_schema must not break the request."""
    import openai as openai_sdk
    from app.config import settings

    monkeypatch.setattr(llm, "_format_index", 0)
    rejected: list[str] = []

    class _PickyCompletions(_StubCompletions):
        async def create(self, **params):
            fmt = (params.get("response_format") or {}).get("type")
            if fmt == "json_schema":
                rejected.append(fmt)
                raise openai_sdk.BadRequestError(
                    "unsupported response_format",
                    response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                    body=None,
                )
            return _Completion(_Msg(content=json.dumps({
                "answer": "ok", "intent": "CONTACT", "confidence": 0.5,
                "citations": [], "actions": [], "escalate": False,
            })))

    stub = _StubOpenAI()
    stub.chat.completions = _PickyCompletions()
    monkeypatch.setattr(llm, "openai_client", lambda: stub)
    monkeypatch.setattr(settings(), "provider", "openai")

    answer, provider, _model, _reason = await llm.generate("hi", [], [], "", None)
    assert rejected == ["json_schema"]          # tried the strongest format first
    assert provider == "openai" and answer.intent == "CONTACT"


@pytest.mark.asyncio
async def test_blank_openai_key_degrades_instead_of_crashing(monkeypatch):
    """A user who has not pasted a key yet gets a real answer, not a stack trace."""
    from app.config import settings

    monkeypatch.setattr(settings(), "provider", "openai")
    monkeypatch.setattr(settings(), "openai_api_key", "")
    monkeypatch.setattr(settings(), "anthropic_api_key", "")

    hits = [{"id": "refunds_and_returns.md#3", "text": "Refunds take 5-7 business days."}]
    answer, provider, _model, reason = await llm.generate("hi", [], hits, "", None)
    assert provider == "degraded" and answer.escalate
    assert "OPENAI_API_KEY" in reason and "ANTHROPIC_API_KEY" in reason


def test_mojibake_from_charsetless_providers_is_repaired():
    """Groq sends application/json with no charset; en-dashes arrive doubly
    encoded. The customer must not see 'a EUR' soup in the answer."""
    broken = "Refunds appear 5\u00e2\u20ac\u201c7 business days after the scan."
    assert llm.fix_mojibake(broken) == "Refunds appear 5\u20137 business days after the scan."
    # clean text, and text with unrelated accents, must pass through untouched
    for ok in ("Refunds take 5-7 days.", "Café conversion rate", "naïve"):
        assert llm.fix_mojibake(ok) == ok


def test_retry_policy_covers_both_sdks():
    """A provider 429 must be retryable whichever SDK raised it."""
    import anthropic
    import openai as openai_sdk

    def resp(code):
        return httpx.Response(code, request=httpx.Request("POST", "http://x"))

    assert llm._transient(openai_sdk.RateLimitError("429", response=resp(429), body=None))
    assert llm._transient(anthropic.APIConnectionError(request=httpx.Request("POST", "http://x")))
    assert llm._transient(openai_sdk.InternalServerError("500", response=resp(500), body=None))
    # a bad request will never fix itself -- retrying it just wastes the quota
    assert not llm._transient(openai_sdk.BadRequestError("400", response=resp(400), body=None))
    assert not llm._transient(ValueError("schema mismatch"))
