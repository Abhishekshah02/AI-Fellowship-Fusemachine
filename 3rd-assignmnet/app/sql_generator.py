"""Task 3 step: turn (question, plan) into a SQL string via the LLM."""

import json
import re

from .llm_client import chat
from .logger import log_event
from .prompts import load
from .schema_info import SCHEMA_DESCRIPTION

_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def generate(question: str, plan: dict | None = None) -> str:
    template = load("generate_sql")
    prompt = template.format(
        schema=SCHEMA_DESCRIPTION,
        plan=json.dumps(plan or {}, indent=2),
        question=question,
    )
    raw = chat(
        system="You write PostgreSQL SELECT statements only.",
        user=prompt,
        purpose="generate_sql",
    )
    sql = _strip_fences(raw)
    log_event("sql_generated", question=question, sql=sql)
    return sql


def fix(question: str, previous_sql: str, error: str) -> str:
    template = load("fix_sql")
    prompt = template.format(
        schema=SCHEMA_DESCRIPTION,
        question=question,
        previous_sql=previous_sql,
        error=error,
    )
    raw = chat(
        system="You repair broken PostgreSQL SELECT statements.",
        user=prompt,
        purpose="fix_sql",
    )
    sql = _strip_fences(raw)
    log_event("sql_fixed", question=question, sql=sql, error=error)
    return sql


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip().rstrip(";").strip()
