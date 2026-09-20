"""Evaluation harness for the dispute-investigation agent. Written from scratch:
no pytest, no eval framework, no LLM judge -- the checks are deterministic so a
result means the same thing on every run.

    python -m evaluation.harness                    # full run + report
    python -m evaluation.harness --cases dup_charge # one case
    python -m evaluation.harness --no-baseline      # skip the single-agent run
    python -m evaluation.harness --pace 45          # seconds between runs

Measures, per the assignment:
  * task completion rate   -- did the agent reach an acceptable terminal state
                              with the evidence its answer needed
  * tool-call correctness  -- every call names a real tool and its arguments
                              validate against that tool's schema
  * trajectory length      -- iterations vs. what the case reasonably needs
  * token cost per query   -- multi-agent, and a single-agent baseline to price
                              the coordination overhead
  * failure log            -- hard / soft / cascading-soft classification

Writes evaluation/results.md and evaluation/results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import agent, agent_tools, rag  # noqa: E402
from app.config import settings  # noqa: E402
from evaluation.cases import CASES, FAULT_CASE  # noqa: E402

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Tool-argument validation -- a small JSON-Schema subset, written here so the
# harness has no dependency on the app's own validation path.
# --------------------------------------------------------------------------- #
_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(tool: str, args: dict) -> list[str]:
    """Returns a list of problems; empty means the call was well-formed."""
    schema = next((s for s in agent_tools.SCHEMAS if s["name"] == tool), None)
    if schema is None:
        return [f"unknown tool {tool!r}"]

    spec = schema["parameters"]
    props, problems = spec.get("properties", {}), []

    for name in spec.get("required", []):
        if name not in args:
            problems.append(f"{tool}: missing required argument {name!r}")

    for name, value in (args or {}).items():
        if name not in props:
            problems.append(f"{tool}: unexpected argument {name!r}")
            continue
        rule = props[name]
        expected = _TYPES.get(rule.get("type"))
        if expected and not isinstance(value, expected):
            problems.append(
                f"{tool}: {name!r} should be {rule['type']}, got {type(value).__name__}"
            )
        if "enum" in rule and value not in rule["enum"]:
            problems.append(f"{tool}: {name!r}={value!r} is not one of {rule['enum']}")
        if rule.get("type") == "integer" and isinstance(value, int):
            if "minimum" in rule and value < rule["minimum"]:
                problems.append(f"{tool}: {name!r}={value} below minimum {rule['minimum']}")
            if "maximum" in rule and value > rule["maximum"]:
                problems.append(f"{tool}: {name!r}={value} above maximum {rule['maximum']}")
    return problems


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class CaseResult:
    id: str
    mode: str
    outcome: str
    completed: bool
    iterations: int
    max_steps: int
    trajectory: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    arg_problems: list[str] = field(default_factory=list)
    tool_calls: int = 0
    valid_tool_calls: int = 0
    tokens_total: int = 0
    tokens_by_role: dict = field(default_factory=dict)
    latency_ms: int = 0
    context_saved_chars: int = 0
    verifier_rejections: int = 0
    failure_type: str | None = None
    failure_reason: str | None = None
    answer: str = ""


PROVIDER_EXHAUSTED = "provider_unavailable"


def _is_provider_limit(error: str | None) -> bool:
    """Distinguish "the provider refused to serve us" from "the agent failed"."""
    text = (error or "").lower()
    return "rate limit" in text or "429" in text or "quota" in text


def classify(case: dict, result, checks: dict) -> tuple[str | None, str | None]:
    """The failure taxonomy.

    provider_unavailable -- not an agent failure at all: the provider refused the
                      request (rate limit or quota). Reported separately and
                      excluded from the completion rate, because it measures the
                      account's budget rather than the agent's behaviour.
    hard           -- no usable output at all: the run errored, or the step
                      budget ran out without a conclusion
    soft           -- a usable answer that is wrong: unacceptable terminal state,
                      an unsupported claim, or a missing piece of evidence
    cascading soft -- a soft failure downstream of an earlier broken step: a tool
                      call failed or was skipped, and the final answer is wrong
                      because of it
    """
    if result.outcome == "failed" and _is_provider_limit(result.error):
        return PROVIDER_EXHAUSTED, f"provider refused the request: {(result.error or '')[:160]}"

    if result.outcome in {"failed", "exhausted"}:
        return "hard", result.error or f"no conclusion within {case['max_steps']} steps"

    if checks["all_ok"]:
        return None, None

    reasons = [k for k, ok in checks.items() if k != "all_ok" and not ok]
    upstream_broken = (not checks["required_tools"]) or any(
        not s["ok"] for s in result.steps if s["tool"] not in agent_tools.TERMINAL
    )
    kind = "cascading_soft" if upstream_broken and not checks["outcome"] else "soft"
    if upstream_broken and not checks["grounded"]:
        kind = "cascading_soft"
    return kind, "failed checks: " + ", ".join(reasons)


def score(case: dict, result, mode: str) -> CaseResult:
    answer = (result.answer or "").lower()
    tools_used = [s["tool"] for s in result.steps]

    arg_problems: list[str] = []
    valid = 0
    for step in result.steps:
        if step["tool"] == "(malformed)":
            arg_problems.append("provider rejected a truncated tool call")
            continue
        problems = validate_args(step["tool"], step["args"])
        arg_problems += problems
        valid += not problems

    checks = {
        "outcome": result.outcome in case["expect_outcomes"],
        "required_tools": all(t in tools_used for t in case["require_tools"]),
        "no_false_claims": not any(bad in answer for bad in case["forbid"]),
        "grounded": (
            not case["expect_evidence"]
            or result.outcome != "resolved"
            or any(ev.lower() in answer for ev in case["expect_evidence"])
        ),
        "args_valid": not arg_problems,
    }
    checks["all_ok"] = all(checks.values())

    failure_type, reason = classify(case, result, checks)
    return CaseResult(
        id=case["id"],
        mode=mode,
        outcome=result.outcome,
        completed=checks["all_ok"],
        iterations=result.iterations,
        max_steps=case["max_steps"],
        trajectory=[s["note"] for s in result.steps],
        tools_used=tools_used,
        arg_problems=arg_problems,
        tool_calls=len(result.steps),
        valid_tool_calls=valid,
        tokens_total=result.tokens.get("total_tokens", 0),
        tokens_by_role=result.tokens.get("by_role", {}),
        latency_ms=result.latency_ms,
        context_saved_chars=result.context_saved_chars,
        verifier_rejections=sum(1 for v in result.verifier_verdicts if not v.get("passed")),
        failure_type=failure_type,
        failure_reason=reason,
        answer=result.answer,
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
async def run_case(case: dict, *, verify: bool, mode: str) -> CaseResult:
    result = await agent.investigate(case["complaint"], verify=verify)
    return score(case, result, mode)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", help="case ids to run (default: all)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the single-agent run")
    ap.add_argument("--baseline-limit", type=int, default=0,
                    help="run the baseline on only the first N cases (0 = all). The "
                         "comparison needs a sample, not every case, and free-tier "
                         "daily token budgets are finite.")
    ap.add_argument("--no-fault", action="store_true", help="skip the failure-injection run")
    ap.add_argument("--pace", type=float, default=40.0,
                    help="seconds to wait between runs (free-tier token budgets)")
    args = ap.parse_args()

    selected = [c for c in CASES if not args.cases or c["id"] in args.cases]
    if not selected:
        sys.exit(f"no cases matched {args.cases}")

    if rag.collection().count() == 0:
        print("building the index...", rag.ingest())

    results: list[CaseResult] = []
    started = time.time()

    async def paced(case, *, verify, mode, first=False):
        if not first:
            await asyncio.sleep(args.pace)
        print(f"  [{mode}] {case['id']} ...", flush=True)
        r = await run_case(case, verify=verify, mode=mode)
        flag = "ok" if r.completed else (r.failure_type or "fail")
        print(f"      -> {r.outcome} in {r.iterations} steps, "
              f"{r.tokens_total} tokens [{flag}]", flush=True)
        return r

    print(f"multi-agent run ({len(selected)} cases)")
    for i, case in enumerate(selected):
        results.append(await paced(case, verify=True, mode="multi_agent", first=i == 0))

    if not args.no_baseline:
        print(f"\nsingle-agent baseline ({len(selected)} cases, verifier off)")
        for case in selected:
            results.append(await paced(case, verify=False, mode="single_agent"))

    fault_result = None
    if not args.no_fault:
        print("\nfailure injection: payment ledger unavailable")
        await asyncio.sleep(args.pace)
        agent_tools.FAULTS.add("payments_unavailable")
        try:
            fault_result = await run_case(FAULT_CASE, verify=True, mode="fault_injection")
            print(f"      -> {fault_result.outcome} in {fault_result.iterations} steps "
                  f"[{'ok' if fault_result.completed else fault_result.failure_type}]")
        finally:
            agent_tools.FAULTS.discard("payments_unavailable")
        results.append(fault_result)

    report = build_report(results, fault_result, elapsed=time.time() - started)
    (HERE / "results.md").write_text(report, encoding="utf-8")
    (HERE / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'results.md'} and results.json")
    print("\n" + report)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 1) if values else 0.0


def build_report(results: list[CaseResult], fault: CaseResult | None, elapsed: float) -> str:
    s = settings()
    multi = [r for r in results if r.mode == "multi_agent"]
    single = [r for r in results if r.mode == "single_agent"]
    blocked = [r for r in results if r.failure_type == PROVIDER_EXHAUSTED]

    def scored(rows: list[CaseResult]) -> list[CaseResult]:
        return [r for r in rows if r.failure_type != PROVIDER_EXHAUSTED]

    def rate(rows: list[CaseResult]) -> str:
        rows = scored(rows)
        if not rows:
            return "n/a"
        done = sum(r.completed for r in rows)
        return f"{done}/{len(rows)} ({100 * done / len(rows):.0f}%)"

    out: list[str] = [
        "# Agent evaluation results",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"model `{s.openai_model}` · temperature {s.temperature} · "
        f"max_steps {s.agent_max_steps} · max_revisions {s.agent_max_revisions} · "
        f"run took {elapsed / 60:.1f} min",
        "",
        "Produced by `python -m evaluation.harness`. Every check is deterministic "
        "(string and schema assertions, no LLM judge).",
        "",
    ]
    if blocked:
        out += [
            f"> **{len(blocked)} run(s) excluded from the rates below**: the provider "
            "refused them on a rate limit or daily quota, so the agent never acted. "
            "That measures the account's budget, not the agent, and folding it into "
            "the completion rate would misreport both. They are listed in the failure "
            "log as `provider_unavailable`.",
            "",
        ]
    out += [
        "## Headline",
        "",
        "| Metric | Multi-agent (with verifier) | Single-agent baseline |",
        "|---|---|---|",
        f"| Task completion rate | {rate(multi)} | {rate(single)} |",
        f"| Mean iterations | {_mean([r.iterations for r in scored(multi)])} | "
        f"{_mean([r.iterations for r in scored(single)])} |",
        f"| Mean tokens / query | {_mean([r.tokens_total for r in scored(multi)]):.0f} | "
        f"{_mean([r.tokens_total for r in scored(single)]):.0f} |",
        f"| Mean latency | {_mean([r.latency_ms for r in scored(multi)]):.0f} ms | "
        f"{_mean([r.latency_ms for r in scored(single)]):.0f} ms |",
    ]

    tool_calls = sum(r.tool_calls for r in scored(multi))
    valid_calls = sum(r.valid_tool_calls for r in scored(multi))
    pct = 100 * valid_calls / tool_calls if tool_calls else 0
    out += [
        f"| Tool-call correctness | {valid_calls}/{tool_calls} ({pct:.0f}%) | — |",
        "",
    ]

    if scored(multi) and scored(single):
        mt = _mean([r.tokens_total for r in scored(multi)])
        st = _mean([r.tokens_total for r in scored(single)])
        overhead = (mt - st) / st * 100 if st else 0
        out += [
            f"**Coordination cost.** The verifier adds **{overhead:+.0f}%** tokens "
            f"({st:.0f} → {mt:.0f} per query), measured over the "
            f"{len(scored(single))} case(s) run both ways. What that buys is in the "
            f"failure table below.",
            "",
        ]

    out += ["## Per-case results (multi-agent)", "",
            "| Case | Outcome | Steps (budget) | Tools used | Tokens | Verifier rejections | Result |",
            "|---|---|---|---|---|---|---|"]
    for r in multi:
        tools = ", ".join(f"`{t}`" for t in r.tools_used) or "—"
        verdict = "pass" if r.completed else f"**{r.failure_type}**"
        out.append(
            f"| `{r.id}` | {r.outcome} | {r.iterations} ({r.max_steps}) | {tools} | "
            f"{r.tokens_total} | {r.verifier_rejections} | {verdict} |"
        )

    out += ["", "## Trajectory length", "",
            "Whether the number of steps is reasonable for the complexity of the query.",
            "", "| Case | Steps | Budget | Within budget |", "|---|---|---|---|"]
    for r in multi:
        out.append(
            f"| `{r.id}` | {r.iterations} | {r.max_steps} | "
            f"{'yes' if r.iterations <= r.max_steps else 'NO'} |"
        )

    out += ["", "## Token cost per query", "",
            "| Case | Investigator | Verifier | Total | Single-agent | Overhead |",
            "|---|---|---|---|---|---|"]
    by_id = {r.id: r for r in single}
    for r in multi:
        inv = r.tokens_by_role.get("investigator", {}).get("total_tokens", 0)
        ver = r.tokens_by_role.get("verifier", {}).get("total_tokens", 0)
        base = by_id.get(r.id)
        base_total = base.tokens_total if base else 0
        delta = f"{(r.tokens_total - base_total) / base_total * 100:+.0f}%" if base_total else "—"
        out.append(
            f"| `{r.id}` | {inv} | {ver} | {r.tokens_total} | {base_total or '—'} | {delta} |"
        )

    failures = [r for r in results if r.failure_type]
    out += ["", "## Failure log", ""]
    if not failures:
        out.append("No failures recorded in this run.")
    else:
        out += ["| Case | Mode | Type | Why |", "|---|---|---|---|"]
        for r in failures:
            out.append(
                f"| `{r.id}` | {r.mode} | **{r.failure_type}** | {r.failure_reason} |"
            )
        out += ["", "Taxonomy: **provider_unavailable** = the provider refused the "
                    "request (rate limit / daily quota); the agent never acted, so it is "
                    "excluded from the rates. **hard** = no usable output (run errored or step budget "
                    "exhausted). **soft** = an answer was produced but it is wrong "
                    "(unacceptable terminal state, unsupported claim, or missing evidence). "
                    "**cascading soft** = the wrong answer follows from an earlier broken or "
                    "skipped step rather than from the final reasoning."]

    if fault is not None:
        out += ["", "## Failure injection: payment ledger unavailable", "",
                "`inspect_payments` raises `ToolUnavailable` for this run. The duplicate "
                "charge is impossible to confirm, so the only honest outcomes are to "
                "escalate or ask the customer.", "",
                f"- Outcome: **{fault.outcome}** after {fault.iterations} steps",
                f"- Behaved correctly: **{'yes' if fault.completed else 'no — ' + str(fault.failure_reason)}**",
                "- Trajectory:"]
        out += [f"  {i}. {note}" for i, note in enumerate(fault.trajectory, 1)]
        out += ["", "Answer given to the customer:", "", "> " +
                (fault.answer.replace("\n", "\n> ") or "(none)")]

    out += ["", "## Context engineering effect", "",
            "Bytes of tool output that never re-entered the prompt because payloads "
            "older than the most recent step were cleared and replaced by their notes:",
            "", "| Case | Steps | Chars cleared |", "|---|---|---|"]
    for r in multi:
        out.append(f"| `{r.id}` | {r.iterations} | {r.context_saved_chars:,} |")

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    asyncio.run(main())
