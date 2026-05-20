"""Smoke tests that don't require an LLM key — they exercise the safety
validator and the executor against the live DB."""

import pytest
from sqlalchemy import create_engine

from app.config import DATABASE_URL
from app.executor import QueryExecutionError, execute
from app.validator import UnsafeQueryError, validate

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def test_validator_accepts_select():
    assert validate("SELECT 1") == "SELECT 1"


def test_validator_strips_trailing_semicolon():
    assert validate("SELECT 1;") == "SELECT 1"


def test_validator_blocks_delete():
    with pytest.raises(UnsafeQueryError):
        validate("DELETE FROM customers")


def test_validator_blocks_drop_inside_with():
    with pytest.raises(UnsafeQueryError):
        validate("WITH x AS (SELECT 1) DROP TABLE customers")


def test_validator_blocks_multiple_statements():
    with pytest.raises(UnsafeQueryError):
        validate("SELECT 1; SELECT 2")


def test_validator_allows_keyword_inside_string():
    sql = "SELECT \"customerName\" FROM customers WHERE \"customerName\" = 'Drop Inc'"
    assert validate(sql).startswith("SELECT")


def test_executor_runs_count():
    out = execute(engine, 'SELECT COUNT(*) FROM customers')
    assert out["rowcount"] == 1
    assert out["rows"][0][0] == 122


def test_executor_surfaces_db_error():
    with pytest.raises(QueryExecutionError):
        execute(engine, 'SELECT "nope" FROM customers')
