"""Safe-query validation.

A generated SQL string is accepted only if:
  * it parses as exactly ONE statement,
  * that statement is a SELECT (or WITH ... SELECT),
  * no banned keyword appears as a standalone token anywhere in the script
    (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, GRANT, REVOKE, CREATE,
    COPY, CALL, MERGE).

Banned-token matching uses a word-boundary regex so that words like
'updated_at' or '"DropShipDate"' won't trip it.
"""

import re

import sqlparse

BANNED = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "GRANT", "REVOKE", "CREATE", "COPY", "CALL", "MERGE",
    "VACUUM", "REINDEX", "ATTACH", "DETACH",
}

_BANNED_RE = re.compile(
    r"\b(" + "|".join(BANNED) + r")\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when a query violates the read-only contract."""


def validate(sql: str) -> str:
    """Return the cleaned SQL or raise UnsafeQueryError."""
    if not sql or not sql.strip():
        raise UnsafeQueryError("empty SQL")

    cleaned = sql.strip().rstrip(";").strip()

    parsed = sqlparse.parse(cleaned)
    if len(parsed) != 1:
        raise UnsafeQueryError(
            f"expected exactly 1 statement, got {len(parsed)}"
        )

    stmt = parsed[0]
    stmt_type = (stmt.get_type() or "").upper()
    if stmt_type not in {"SELECT", "UNKNOWN"}:
        # sqlparse labels WITH ... SELECT as UNKNOWN; we handle that below.
        raise UnsafeQueryError(f"only SELECT is allowed, got {stmt_type}")

    first_keyword = _first_keyword(cleaned).upper()
    if first_keyword not in {"SELECT", "WITH"}:
        raise UnsafeQueryError(
            f"query must start with SELECT or WITH, got '{first_keyword}'"
        )

    hit = _BANNED_RE.search(_strip_strings_and_comments(cleaned))
    if hit:
        raise UnsafeQueryError(f"banned keyword detected: {hit.group(1).upper()}")

    return cleaned


def _first_keyword(sql: str) -> str:
    for token in sqlparse.parse(sql)[0].flatten():
        if token.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword.CTE):
            return token.value
    return ""


def _strip_strings_and_comments(sql: str) -> str:
    """Remove string literals and comments so banned-keyword regex can't
    false-positive on text content like a customer name 'UPDATE Inc'."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    no_strings = re.sub(r"'(?:''|[^'])*'", " ", no_line)
    return no_strings
