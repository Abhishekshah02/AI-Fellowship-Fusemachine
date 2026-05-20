"""Task 4: the think/act/correct loop.

Pipeline per request:
  1. Decompose the question (LLM, JSON)              [step: understand]
  2. Generate SQL from question + plan               [step: generate]
  3. Validate SQL (only SELECT, no DDL/DML)          [step: validate]
  4. Execute SQL                                     [step: execute]
  5. On error: ask LLM to fix it, retry (up to N).
  6. Summarize the rows into a natural-language answer.

The function returns a dict in the shape the assignment specifies plus
some extra telemetry (`attempts`, `decomposition`, `execution_ms`).
"""

import time
from typing import Any

from sqlalchemy.engine import Engine

from .config import AGENT_MAX_RETRIES
from .decomposer import decompose
from .executor import QueryExecutionError, execute
from .logger import log_event
from .nl_summarizer import summarize
from .sql_generator import fix, generate
from .validator import UnsafeQueryError, validate


def run(question: str, engine: Engine, max_retries: int = AGENT_MAX_RETRIES) -> dict[str, Any]:
    started = time.perf_counter()
    log_event("agent_started", question=question, max_retries=max_retries)

    plan = decompose(question)

    attempts: list[dict[str, Any]] = []
    sql: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    for attempt in range(1, max_retries + 1):
        try:
            sql_raw = (
                generate(question, plan)
                if attempt == 1
                else fix(question, sql or "", error or "")
            )
            sql = validate(sql_raw)
        except UnsafeQueryError as exc:
            error = f"unsafe query: {exc}"
            attempts.append({"attempt": attempt, "sql": sql_raw, "error": error})
            log_event("agent_unsafe", attempt=attempt, error=error, sql=sql_raw)
            continue
        except Exception as exc:  # noqa: BLE001 - LLM/network errors
            error = f"generation failed: {exc}"
            attempts.append({"attempt": attempt, "sql": None, "error": error})
            log_event("agent_generation_error", attempt=attempt, error=error)
            continue

        try:
            result = execute(engine, sql)
            attempts.append({"attempt": attempt, "sql": sql, "error": None})
            error = None
            break
        except QueryExecutionError as exc:
            error = str(exc)
            attempts.append({"attempt": attempt, "sql": sql, "error": error})
            log_event("agent_retry", attempt=attempt, error=error, sql=sql)

    total_ms = round((time.perf_counter() - started) * 1000, 2)

    if result is None:
        log_event(
            "agent_failed",
            question=question,
            attempts=len(attempts),
            last_error=error,
            total_ms=total_ms,
        )
        return {
            "question": question,
            "decomposition": plan,
            "sql": sql,
            "result": None,
            "summary": (
                "Sorry — I couldn't produce a working SQL query after "
                f"{len(attempts)} attempt(s). Last error: {error}"
            ),
            "status": "failed",
            "attempts": attempts,
            "execution_ms": total_ms,
        }

    summary = summarize(
        question=question,
        sql=sql or "",
        columns=result["columns"],
        rows=result["rows"],
        rowcount=result["rowcount"],
    )
    log_event(
        "agent_success",
        question=question,
        attempts=len(attempts),
        rowcount=result["rowcount"],
        total_ms=total_ms,
    )
    scalar_result = _to_scalar(result)
    return {
        "question": question,
        "decomposition": plan,
        "sql": sql,
        "result": scalar_result if scalar_result is not None else result["rows"],
        "columns": result["columns"],
        "rowcount": result["rowcount"],
        "summary": summary,
        "status": "success",
        "attempts": attempts,
        "execution_ms": total_ms,
    }


def _to_scalar(result: dict[str, Any]) -> Any:
    """Match the assignment's `result: 42` shape when the query returned a
    single cell (COUNT/SUM/AVG/MAX/MIN)."""
    if result["rowcount"] == 1 and len(result["columns"]) == 1:
        return result["rows"][0][0]
    return None
