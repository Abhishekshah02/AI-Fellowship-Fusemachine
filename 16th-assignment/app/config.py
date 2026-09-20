"""Single source of configuration. Everything is overridable by env var / .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which provider is tried first. The other one becomes the fallback, so
    # either order gives the same three-tier degradation ladder.
    #   claude -> Anthropic API        openai -> any OpenAI-compatible endpoint
    provider: Literal["claude", "openai"] = "claude"

    # --- Claude ---------------------------------------------------------------
    anthropic_api_key: str = ""
    primary_model: str = "claude-opus-5"

    # --- OpenAI-compatible endpoint ------------------------------------------
    # One client, many backends: Groq, OpenRouter, Google Gemini's compat
    # endpoint, Cerebras, or a local vLLM / Ollama server. See .env.example.
    openai_base_url: str = "http://localhost:8000/v1"
    openai_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    openai_api_key: str = "EMPTY"  # local servers ignore it, the client needs one

    # --- Generation -----------------------------------------------------------
    max_tokens: int = 2048
    # Sampling knobs. Applied to any model that accepts them (every
    # OpenAI-compatible model, plus Haiku 4.5). Claude Opus 5 / Sonnet 5 reject
    # temperature/top_p with a 400 and take output_config.effort instead --
    # see llm.py:supports_sampling.
    temperature: float = 0.2
    top_p: float = 0.9
    effort: str = "medium"  # low | medium | high | xhigh | max

    # --- RAG -----------------------------------------------------------------
    kb_dir: Path = ROOT / "data" / "kb"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection: str = "shopassist_kb"
    chunk_size: int = 900       # characters
    chunk_overlap: int = 150
    top_k: int = 4

    # --- Intent router (ONNX) -----------------------------------------------
    router_path: Path = ROOT / "models" / "router"
    router_min_confidence: float = 0.35  # below this the router stays quiet

    # --- Agent (W16) ----------------------------------------------------------
    agent_max_steps: int = 10       # hard stop: the loop can never run forever
    agent_max_revisions: int = 2    # how many times the verifier may send it back
    verifier_model: str = ""        # blank = same model as the investigator

    # --- Tools ---------------------------------------------------------------
    orders_path: Path = ROOT / "data" / "orders.json"
    payments_path: Path = ROOT / "data" / "payments.json"
    tickets_path: Path = ROOT / "data" / "tickets.jsonl"

    # --- Reliability / performance ------------------------------------------
    rate_limit_per_minute: int = 30
    max_retries: int = 4
    retry_base_delay: float = 0.5
    retry_max_delay: float = 25.0   # provider TPM windows need >8s to clear
    request_timeout: float = 60.0
    cache_ttl_seconds: int = 900
    cache_max_entries: int = 512
    max_tool_iterations: int = 4


@lru_cache
def settings() -> Settings:
    return Settings()
