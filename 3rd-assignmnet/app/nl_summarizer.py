"""Task 4 step: turn the SQL result into a one-line natural-language answer."""

from .llm_client import MissingAPIKeyError, chat
from .logger import log_event
from .prompts import load


def summarize(question: str, sql: str, columns: list[str], rows: list[list], rowcount: int) -> str:
    sample = rows[:50]
    template = load("summarize")
    prompt = template.format(
        question=question,
        sql=sql,
        columns=", ".join(columns),
        rows="\n".join(str(r) for r in sample) or "(no rows)",
        rowcount=rowcount,
    )
    try:
        answer = chat(
            system="You explain SQL results in one sentence.",
            user=prompt,
            purpose="summarize",
        )
    except MissingAPIKeyError:
        return _fallback(question, columns, rows, rowcount)

    log_event("summarized", question=question, answer=answer)
    return answer


def _fallback(question: str, columns: list[str], rows: list[list], rowcount: int) -> str:
    if rowcount == 0:
        return "The query returned no rows."
    if rowcount == 1 and len(columns) == 1:
        return f"{columns[0]}: {rows[0][0]}"
    return f"The query returned {rowcount} row(s) across columns: {', '.join(columns)}."
