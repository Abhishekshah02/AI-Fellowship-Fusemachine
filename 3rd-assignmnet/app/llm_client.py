"""Thin wrapper around the Google Generative AI (Gemini) SDK.

The rest of the codebase only sees `chat()` and `chat_json()`. Provider
specifics (JSON mode, token-usage shape, fence-stripping) live here.
"""

import json
from typing import Any

import google.generativeai as genai

from .config import GEMINI_MODEL, GOOGLE_API_KEY
from .logger import log_event


class MissingAPIKeyError(RuntimeError):
    pass


_configured = False


def _ensure_configured() -> None:
    global _configured
    if not GOOGLE_API_KEY:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/app/apikey and add it to .env."
        )
    if not _configured:
        genai.configure(api_key=GOOGLE_API_KEY)
        _configured = True


def _make_model(system: str, json_mode: bool, temperature: float):
    generation_config: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system,
        generation_config=generation_config,
    )


def chat(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
    purpose: str = "chat",
) -> str:
    _ensure_configured()
    model = _make_model(system=system, json_mode=json_mode, temperature=temperature)
    response = model.generate_content(user)

    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    log_event(
        "llm_call",
        purpose=purpose,
        model=GEMINI_MODEL,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        completion_tokens=getattr(usage, "candidates_token_count", None),
    )
    return text


def chat_json(system: str, user: str, *, purpose: str = "chat_json") -> dict:
    raw = chat(system, user, json_mode=True, purpose=purpose)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {raw!r}") from exc
