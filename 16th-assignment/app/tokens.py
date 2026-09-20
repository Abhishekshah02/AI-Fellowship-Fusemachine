"""Per-run token accounting.

The evaluation harness has to report what each query cost, and the multi-agent
design has to be comparable against a single-agent baseline, so every provider
call is recorded against the role that made it (investigator / verifier /
single_agent) rather than into one global counter.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoleUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class TokenLedger:
    """One ledger per agent run. Cheap enough to always be on."""

    by_role: dict[str, RoleUsage] = field(default_factory=dict)

    def record(self, role: str, response) -> None:
        """Pull usage off a provider response. OpenAI and Anthropic name the
        fields differently and either may omit them, so read defensively -- a
        missing usage block must not break the run."""
        usage = getattr(response, "usage", None)
        entry = self.by_role.setdefault(role, RoleUsage())
        entry.calls += 1
        if usage is None:
            return
        entry.prompt_tokens += int(
            getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        )
        entry.completion_tokens += int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
        )

    @property
    def total(self) -> int:
        return sum(u.total for u in self.by_role.values())

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.by_role.values())

    def summary(self) -> dict:
        return {
            "total_tokens": self.total,
            "total_calls": self.calls,
            "by_role": {
                role: {
                    "calls": u.calls,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total,
                }
                for role, u in sorted(self.by_role.items())
            },
        }
