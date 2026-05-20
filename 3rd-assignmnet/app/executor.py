"""Run a validated SELECT against Postgres and return a tidy payload."""

import time
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import QUERY_ROW_LIMIT
from .logger import log_event


class QueryExecutionError(Exception):
    def __init__(self, message: str, sql: str):
        super().__init__(message)
        self.sql = sql


def execute(engine: Engine, sql: str, row_limit: int = QUERY_ROW_LIMIT) -> dict[str, Any]:
    """Execute `sql` and return {columns, rows, rowcount, execution_ms, truncated}.

    Raises QueryExecutionError on database errors so the agent loop can
    capture the message and feed it back to the LLM.
    """
    start = time.perf_counter()
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            fetched = result.fetchmany(row_limit + 1)
    except SQLAlchemyError as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        # SQLAlchemy wraps the driver error; the original DB message is in orig.
        message = str(getattr(exc, "orig", exc)).strip()
        log_event("sql_error", sql=sql, error=message, execution_ms=elapsed_ms)
        raise QueryExecutionError(message, sql) from exc

    truncated = len(fetched) > row_limit
    rows = [list(r) for r in fetched[:row_limit]]
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    log_event(
        "sql_executed",
        sql=sql,
        execution_ms=elapsed_ms,
        rowcount=len(rows),
        truncated=truncated,
    )
    return {
        "columns": columns,
        "rows": rows,
        "rowcount": len(rows),
        "execution_ms": elapsed_ms,
        "truncated": truncated,
    }
