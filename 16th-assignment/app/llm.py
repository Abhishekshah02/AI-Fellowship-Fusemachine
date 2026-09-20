"""Provider layer. Retries, fallback and both tool-calling loops live here.

Two full providers, either of which can be primary (PROVIDER=claude|openai);
whichever is not primary becomes the fallback, so the ladder is always:

  1. primary   -- tool calling + structured JSON output
  2. fallback  -- the other provider, same features
  3. degraded  -- neither reachable: return the retrieved policy text and escalate

  claude -> Anthropic API
  openai -> any OpenAI-compatible endpoint: Groq, OpenRouter, Google Gemini's
            compat endpoint, Cerebras, or a local vLLM / Ollama server
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic
import openai as openai_sdk
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from . import tools as toolkit
from .config import settings
from .reliability import metrics
from .schemas import AssistantAnswer, json_schema

log = logging.getLogger("shopassist.llm")

SYSTEM_PROMPT = """\
You are ShopAssist AI, the front-line customer support assistant for the ShopAssist
online store. You answer shoppers directly, in a warm and concise voice.

Rules you must not break:
1. Ground every factual claim about policy in the CONTEXT block or in a
   search_knowledge_base result. If neither covers the question, say so plainly and
   escalate instead of guessing. Never invent dates, amounts or policy.
2. Cite the chunk ids you actually used (e.g. "refunds_and_returns.md#2") in
   `citations`. No citation means you did not use the knowledge base.
3. Use tools rather than asking the customer for information you can look up.
   Look up an order whenever an order id or an email address is given.
4. Escalate (set `escalate` true and call create_support_ticket) when the
   escalation rules in the knowledge base apply -- disputed money, lost parcels,
   account lockouts, an angry customer, or anything the knowledge base cannot answer.
5. Never request or repeat full card numbers, passwords, or another customer's data.
6. Keep `answer` under 120 words unless the customer asked for step-by-step help."""

# Kept separate from SYSTEM_PROMPT on purpose: an instruction to emit JSON while
# tools are offered pushes some models (Groq's gpt-oss among them) to answer by
# "calling" a tool literally named `json`, which the provider then rejects.
JSON_INSTRUCTION = (
    "\n\nReply with the JSON object described by the output schema and nothing else: "
    "exactly the keys " + ", ".join(json_schema()["properties"]) + ". "
    "No prose, no markdown fence, no tool calls."
)

# Models that reject temperature/top_p (the thinking-era models tune depth with
# output_config.effort instead). Sampling knobs are applied to everything else.
_NO_SAMPLING = (
    "claude-opus-5",
    "claude-opus-4-",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-",
    "claude-mythos-",
)


def supports_sampling(model: str) -> bool:
    return not model.startswith(_NO_SAMPLING)


# Both SDKs raise their own exception classes, and the retry policy has to know
# about both -- otherwise a provider's 429 sails straight past the retries.
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    openai_sdk.APIConnectionError,
    openai_sdk.APITimeoutError,
    openai_sdk.RateLimitError,
    TimeoutError,
    ConnectionError,
)


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    if isinstance(exc, (anthropic.APIStatusError, openai_sdk.APIStatusError)):
        return exc.status_code >= 500  # 4xx other than 429 will not fix itself
    return False


_RETRY_AFTER = re.compile(r"try again in ([0-9.]+)s", re.I)


def retry_after_seconds(exc: BaseException) -> float:
    """How long the provider asked us to wait, from the header or the message.

    Rate limits are the one retryable error that carries its own answer; ignoring
    it means backing off too little (and failing) or too much (and stalling).
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    raw = header.get("retry-after") or header.get("Retry-After")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    match = _RETRY_AFTER.search(str(exc))
    return float(match.group(1)) if match else 0.0


def _wait(retry_state) -> float:
    s = settings()
    backoff = wait_exponential_jitter(initial=s.retry_base_delay, max=s.retry_max_delay)
    delay = backoff(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        # +1s of slack: waiting exactly the stated window tends to land on it again.
        delay = max(delay, min(retry_after_seconds(exc) + 1.0, s.retry_max_delay))
    return delay


def _retrying() -> AsyncRetrying:
    s = settings()
    return AsyncRetrying(
        stop=stop_after_attempt(s.max_retries),
        wait=_wait,
        retry=retry_if_exception(_transient),
        reraise=True,
    )


def _user_block(message: str, context: str, router_hint: str | None) -> str:
    hint = (
        f"\nA fine-tuned intent classifier routed this message to: {router_hint}. "
        "Treat it as a strong prior; override it only if the message clearly says otherwise."
        if router_hint
        else ""
    )
    return (
        f"CONTEXT (retrieved knowledge-base chunks, most relevant first):\n"
        f"{context or '(none)'}\n{hint}\n\nCUSTOMER MESSAGE:\n{message}"
    )


# Tell-tales of UTF-8 bytes decoded as cp1252 ("5–7" arriving as "5â€"7").
_MOJIBAKE = ("â€", "Ã©", "Ã¨", "Â ", "Â»", "â„")


def fix_mojibake(text: str) -> str:
    """Some providers (Groq among them) send `application/json` with no charset.
    The HTTP client then guesses cp1252 and every non-ASCII character arrives
    doubly encoded. Round-trip it back when the damage is visible."""
    if not any(m in text for m in _MOJIBAKE):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text  # not this kind of damage; leave it alone


def parse_answer(text: str) -> AssistantAnswer | None:
    """Schema-constrained decoding makes this a formality; smaller open models
    are less obedient, so pull the first JSON object out and validate it."""
    text = fix_mojibake(text or "")
    for candidate in (text, *re.findall(r"\{.*\}", text, re.S)):
        try:
            return AssistantAnswer.model_validate(json.loads(candidate))
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Tier 1: Claude
# --------------------------------------------------------------------------- #
_anthropic_client: anthropic.AsyncAnthropic | None = None


def anthropic_client() -> anthropic.AsyncAnthropic | None:
    global _anthropic_client
    s = settings()
    if not s.anthropic_api_key:
        return None
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic(
            api_key=s.anthropic_api_key,
            timeout=s.request_timeout,
            max_retries=0,  # tenacity owns retries so the policy lives in one place
        )
    return _anthropic_client


async def _claude(
    message: str, history: list[dict[str, str]], context: str, router_hint: str | None
) -> tuple[AssistantAnswer, list[str]]:
    s = settings()
    client = anthropic_client()
    if client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    params: dict[str, Any] = {
        "model": s.primary_model,
        "max_tokens": s.max_tokens,
        # Cache the frozen system prompt + tool list: later turns skip re-reading it.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT + JSON_INSTRUCTION,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": toolkit.TOOLS,
        # Constrained decoding: the response is guaranteed to match the schema.
        "output_config": {"format": {"type": "json_schema", "schema": json_schema()}},
    }
    if supports_sampling(s.primary_model):
        params["temperature"] = s.temperature
        params["top_p"] = s.top_p
    else:
        # The models that reject temperature/top_p tune depth with effort instead.
        params["output_config"]["effort"] = s.effort

    messages: list[dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    messages.append({"role": "user", "content": _user_block(message, context, router_hint)})

    called: list[str] = []
    for _ in range(s.max_tool_iterations):
        async for attempt in _retrying():
            with attempt:
                response = await client.messages.create(**params, messages=messages)

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"refused: {getattr(response.stop_details, 'category', 'unknown')}"
            )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in tool_uses:
            called.append(block.name)
            metrics.bump(f"tool_{block.name}")
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": await toolkit.dispatch(block.name, dict(block.input)),
                }
            )
        # all results for one assistant turn go back in a single user message
        messages.append({"role": "user", "content": results})
    else:
        log.warning("tool loop hit max_tool_iterations")

    text = next((b.text for b in response.content if b.type == "text"), "")
    parsed = parse_answer(text)
    if parsed is None:
        raise ValueError(f"model returned non-schema output: {text[:200]!r}")
    return parsed, called


# --------------------------------------------------------------------------- #
# Tier 2: any OpenAI-compatible endpoint -- Groq, OpenRouter, Google Gemini's
# compat endpoint, Cerebras, or a local vLLM / Ollama server. Full tool calling,
# so this tier is a complete provider, not a stripped-down spare.
#
# Two phases per turn, because providers reject `tools` together with
# `response_format` ("json mode cannot be combined with tool/function calling"):
#
#   phase 1  tools, no response_format  -- let the model gather what it needs
#   phase 2  response_format, no tools  -- make it answer inside the schema
#
# Phase 2 restates the tool results as plain text on a fresh transcript instead
# of replaying tool-call messages: a history containing tool calls keeps some
# providers in tool-calling mode, where they answer by "calling" a synthetic
# `json` tool that is not in request.tools and the call 400s.
# --------------------------------------------------------------------------- #
_openai_client = None

# Providers also disagree about which structured-output mode they support:
# json_schema constrains decoding and is strongest, json_object is near-universal,
# some accept neither. Step down the ladder the first time a provider rejects a
# format and remember it, rather than hard-coding a guess per provider.
_RESPONSE_FORMATS: list[dict[str, Any] | None] = [
    {
        "type": "json_schema",
        "json_schema": {"name": "assistant_answer", "strict": True, "schema": json_schema()},
    },
    {"type": "json_object"},
    None,
]
_format_index = 0


def openai_client():
    global _openai_client
    s = settings()
    if not s.openai_api_key:
        # Blank means unconfigured; local servers that need no auth take "EMPTY".
        raise RuntimeError("OPENAI_API_KEY is not set")
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI(
            base_url=s.openai_base_url,
            api_key=s.openai_api_key,
            timeout=s.request_timeout,
            max_retries=0,
        )
    return _openai_client


async def _openai(
    message: str, history: list[dict[str, str]], context: str, router_hint: str | None
) -> tuple[AssistantAnswer, list[str]]:
    global _format_index

    s = settings()
    client = openai_client()
    base: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    base += [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    base.append({"role": "user", "content": _user_block(message, context, router_hint)})

    async def call(messages: list[dict[str, Any]], *, use_tools: bool):
        """One request. When asking for structured output, step down the format
        ladder if this provider rejects what we asked for."""
        global _format_index
        while True:
            params: dict[str, Any] = {
                "model": s.openai_model,
                "messages": messages,
                "max_tokens": s.max_tokens,
                "temperature": s.temperature,
                "top_p": s.top_p,
            }
            if use_tools:
                params["tools"] = toolkit.openai_tools()
            elif (fmt := _RESPONSE_FORMATS[_format_index]) is not None:
                params["response_format"] = fmt
            try:
                return await client.chat.completions.create(**params)
            except openai_sdk.BadRequestError:
                if use_tools or _format_index >= len(_RESPONSE_FORMATS) - 1:
                    raise
                _format_index += 1
                log.warning(
                    "provider rejected response_format, stepping down to %s",
                    _RESPONSE_FORMATS[_format_index],
                )

    # ---- phase 1: let the model call tools ---------------------------------
    work = list(base)
    called: list[str] = []
    notes: list[str] = []
    for _ in range(s.max_tool_iterations):
        try:
            async for attempt in _retrying():
                with attempt:
                    response = await call(work, use_tools=True)
        except openai_sdk.BadRequestError as exc:
            # Some models answer by inventing a tool the provider then rejects.
            # Losing the tool phase is survivable; losing the answer is not.
            log.warning("tool phase unusable on this provider, skipping it: %s", exc)
            metrics.bump("tool_phase_skipped")
            break

        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []
        if not tool_calls:
            break

        work.append(choice.model_dump(exclude_none=True))
        for tc in tool_calls:
            called.append(tc.function.name)
            metrics.bump(f"tool_{tc.function.name}")
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}  # a malformed call still gets a result, never a crash
            result = await toolkit.dispatch(tc.function.name, args)
            work.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            notes.append(f"{tc.function.name}({json.dumps(args)}) -> {result}")
    else:
        log.warning("tool loop hit max_tool_iterations")

    # ---- phase 2: clean transcript, constrained to the schema ---------------
    final = [{"role": "system", "content": SYSTEM_PROMPT + JSON_INSTRUCTION}] + base[1:]
    if notes:
        final.append(
            {
                "role": "user",
                "content": "TOOL RESULTS (already fetched for you, use them):\n"
                + "\n".join(notes),
            }
        )
    async for attempt in _retrying():
        with attempt:
            response = await call(final, use_tools=False)

    content = response.choices[0].message.content or ""
    parsed = parse_answer(content)
    if parsed is None:
        raise ValueError(f"model returned non-schema output: {content[:200]!r}")
    return parsed, called

# --------------------------------------------------------------------------- #
# Tier 3: no model reachable
# --------------------------------------------------------------------------- #
def _degraded(hits: list[dict], router_hint: str | None) -> AssistantAnswer:
    excerpt = hits[0]["text"][:400] if hits else ""
    return AssistantAnswer(
        answer=(
            "I can't reach my AI service right now, so I can't answer properly. "
            "Here is the policy text closest to your question; a human agent will "
            "follow up.\n\n" + excerpt
        ).strip(),
        intent=router_hint or "CONTACT",  # type: ignore[arg-type]
        confidence=0.0,
        citations=[h["id"] for h in hits[:2]],
        actions=[],
        escalate=True,
    )


async def generate(
    message: str,
    history: list[dict[str, str]],
    hits: list[dict],
    context: str,
    router_hint: str | None,
) -> tuple[AssistantAnswer, str, str, str | None]:
    """Try each provider in turn, degrade rather than fail.

    Returns (answer, provider, model, degraded_reason).
    """
    s = settings()
    tiers = [("claude", _claude, s.primary_model), ("openai", _openai, s.openai_model)]
    if s.provider == "openai":
        tiers.reverse()

    failures: list[str] = []
    for name, fn, model in tiers:
        try:
            answer, called = await fn(message, history, context, router_hint)
            answer.actions = called
            metrics.bump(f"provider_{name}")
            return answer, name, model, "; ".join(failures) or None
        except Exception as exc:
            log.warning("provider %s failed: %s", name, exc)
            metrics.bump(f"{name}_failures")
            failures.append(f"{name}: {exc}")

    metrics.bump("provider_degraded")
    return _degraded(hits, router_hint), "degraded", "none", "; ".join(failures)
