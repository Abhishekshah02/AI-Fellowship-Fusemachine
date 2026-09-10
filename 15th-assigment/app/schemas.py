"""Request/response contracts. AssistantAnswer doubles as the JSON schema the
model is constrained to emit, so the API can never return prose where the client
expects JSON."""
from typing import Any, Literal

from pydantic import BaseModel, Field

INTENTS = [
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]


class AssistantAnswer(BaseModel):
    """The structured payload the LLM must produce."""

    answer: str = Field(description="Reply to show the customer, plain text.")
    intent: Literal[tuple(INTENTS)] = Field(  # type: ignore[valid-type]
        description="Support queue this message belongs to."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer, 0-1.")
    citations: list[str] = Field(
        default_factory=list,
        description="Source ids of knowledge-base chunks used, e.g. 'refunds_and_returns.md#2'.",
    )
    actions: list[str] = Field(
        default_factory=list, description="Tools called while answering."
    )
    escalate: bool = Field(description="True when a human agent must follow up.")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(
        default_factory=list, description="Prior turns: [{role: user|assistant, content: ...}]"
    )
    session_id: str | None = None


class ChatResponse(AssistantAnswer):
    model: str
    provider: Literal["claude", "openai", "degraded"]
    cached: bool = False
    latency_ms: int = 0
    router_intent: str | None = None
    router_confidence: float | None = None
    degraded_reason: str | None = None


def json_schema() -> dict[str, Any]:
    """Anthropic/vLLM-compatible JSON schema: closed object, every field required."""
    schema = AssistantAnswer.model_json_schema()
    schema.pop("$defs", None)
    schema["additionalProperties"] = False
    schema["required"] = list(schema["properties"])
    for prop in schema["properties"].values():
        prop.pop("default", None)
    return schema
