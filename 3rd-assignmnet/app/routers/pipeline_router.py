"""Task 3 endpoint: linear Text-to-SQL pipeline with a single retry on error."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..database import engine
from ..decomposer import decompose
from ..executor import QueryExecutionError, execute
from ..logger import log_event
from ..sql_generator import fix, generate
from ..validator import UnsafeQueryError, validate

router = APIRouter(prefix="/text2sql", tags=["text2sql"])


class T2SRequest(BaseModel):
    question: str = Field(..., min_length=3)


@router.post("")
def text_to_sql(req: T2SRequest) -> dict:
    question = req.question.strip()
    log_event("t2s_request", question=question)

    plan = decompose(question)

    try:
        sql = validate(generate(question, plan))
    except UnsafeQueryError as exc:
        return {
            "question": question,
            "decomposition": plan,
            "sql": None,
            "result": None,
            "status": "rejected",
            "error": f"unsafe query: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "question": question,
            "decomposition": plan,
            "sql": None,
            "result": None,
            "status": "failed",
            "error": f"generation failed: {exc}",
        }

    try:
        result = execute(engine, sql)
        return {
            "question": question,
            "decomposition": plan,
            "sql": sql,
            "result": result["rows"],
            "columns": result["columns"],
            "rowcount": result["rowcount"],
            "execution_ms": result["execution_ms"],
            "retried": False,
            "status": "success",
        }
    except QueryExecutionError as exc:
        first_error = str(exc)
        log_event("t2s_retry", question=question, error=first_error)
        try:
            sql_fixed = validate(fix(question, sql, first_error))
            result = execute(engine, sql_fixed)
            return {
                "question": question,
                "decomposition": plan,
                "sql": sql_fixed,
                "result": result["rows"],
                "columns": result["columns"],
                "rowcount": result["rowcount"],
                "execution_ms": result["execution_ms"],
                "retried": True,
                "first_error": first_error,
                "status": "success",
            }
        except Exception as exc2:  # noqa: BLE001
            return {
                "question": question,
                "decomposition": plan,
                "sql": sql,
                "result": None,
                "retried": True,
                "first_error": first_error,
                "error": str(exc2),
                "status": "failed",
            }
