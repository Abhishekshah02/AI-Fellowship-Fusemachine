# Architecture & Design Decisions

A short walk-through of the pipeline so a reviewer can see how the moving parts
connect and why each one exists.

```
              ┌──────────────────────────────────────────────────────┐
              │            FastAPI app (app/main.py)                 │
              │                                                      │
              │   POST /text2sql                POST /agent/sql       │
              │   (Task 3 pipeline)             (Task 4 agent)        │
              └────┬──────────────────────────────────┬───────────────┘
                   │                                  │
                   ▼                                  ▼
       ┌──────────────────────────┐   ┌──────────────────────────────┐
       │  decomposer.decompose()  │   │   agent.run() — the loop:     │
       │  (LLM JSON)              │   │   1. decompose                │
       └──────────────┬───────────┘   │   2. generate_sql             │
                      │               │   3. validate                  │
                      ▼               │   4. execute                   │
       ┌──────────────────────────┐   │   5. on error → fix_sql, retry│
       │ sql_generator.generate() │   │      (up to AGENT_MAX_RETRIES)│
       │  (LLM, text)             │   │   6. summarize                 │
       └──────────────┬───────────┘   └────────────┬──────────────────┘
                      ▼                            ▼
              validator.validate()  ─────►   executor.execute()
              (SELECT-only, sqlparse)         (SQLAlchemy → Postgres)
                                                          │
                                                          ▼
                                                  nl_summarizer
                                                  (LLM, text)
```

## Why each file exists

| File                                | Purpose                                                              |
|-------------------------------------|----------------------------------------------------------------------|
| `app/config.py`                     | Loads `.env` once. Single place to find tunables (`AGENT_MAX_RETRIES`, `QUERY_ROW_LIMIT`). |
| `app/database.py`                   | SQLAlchemy engine + session factory.                                 |
| `app/schema_info.py`                | Hand-curated schema string for prompts + `TABLE_COLUMNS` for fallback decomposer. Kept hand-curated so the LLM sees stable, semantic-friendly column docs (e.g. that MSRP is retail price). |
| `app/logger.py`                     | Plain logger plus `log_event()` that appends JSON lines to `logs/events.jsonl`. The benchmark + audits replay these. |
| `app/validator.py`                  | Sole gate between LLM output and Postgres. Single-statement parse, SELECT-only check, banned-keyword regex on strings-stripped SQL so `'Drop Inc.'` doesn't false-positive. |
| `app/executor.py`                   | Wraps the SELECT in a transaction, captures driver error text, times the call, truncates at `QUERY_ROW_LIMIT`. Raises `QueryExecutionError(message, sql)` so the agent loop can read both. |
| `app/llm_client.py`                 | One-place wrapper around the `google-generativeai` (Gemini) SDK. `chat()` + `chat_json()`. Logs token counts. |
| `app/prompts.py` + `prompts/*.txt`  | Prompts live as `.txt` files so they can be edited / diffed without touching Python. |
| `app/decomposer.py`                 | LLM JSON-mode call. Falls back to a rule-based stub if the LLM is unreachable, so the rest of the pipeline still runs. |
| `app/sql_generator.py`              | Two LLM functions: `generate(question, plan)` (first attempt) and `fix(question, prev_sql, error)` (correction attempt). Both strip stray markdown fences. |
| `app/nl_summarizer.py`              | Turns rows into a one-line English answer. Has a no-LLM fallback. |
| `app/agent.py`                      | The self-correcting loop. Per-attempt logs go through `log_event()`. |
| `app/routers/pipeline_router.py`    | Task 3 endpoint. Strict linear pipeline with **one** retry. |
| `app/routers/agent_router.py`       | Task 4 endpoint. Calls `agent.run()` and returns its dict. |
| `benchmark/ground_truth.json`       | Source of truth for the 50 benchmark questions + reference SQL. |
| `benchmark/build_task1_doc.py`      | Executes the ground-truth SQL once, writes Task 1 markdown + caches results. |
| `benchmark/run_eval.py`             | Hits a running FastAPI server with each question, grades the response against ground-truth rows (multiset equality, alias- and order-agnostic), writes per-endpoint scorecard. |

## Design decisions

### 1. LLM for decomposition AND generation (not just generation)

The assignment asks the agent to "understand" before generating. Two separate
LLM calls makes that visible — the decomposition is logged and returned with
the final response so a reviewer can see *why* the SQL was generated the way
it was. It also gives the SQL generator a structured anchor that improves
accuracy on the GROUP BY / aggregation questions.

### 2. Hand-curated schema in the prompt, not introspected at request time

Introspecting `information_schema` per call would be wasteful (the schema
doesn't change) and the LLM benefits from prose annotations we can't get from
the catalog — e.g. "MSRP is retail price, buyPrice is wholesale." That
distinction is critical for Q44 ("Average product price").

### 3. Validator runs *after* generation, *before* execution

It's not enough to prompt the model to write only SELECTs — that's advice,
not a guarantee. The validator is the only thing standing between a prompt-
injected `DROP TABLE` and the database. It uses `sqlparse` for statement
counting / DML detection plus a regex pass over a strings-stripped copy of
the SQL so customer names containing SQL keywords (rare but real: "UPDATE
Marketing LLC") don't trip it.

### 4. Retry contract: max 3 in Task 4, max 1 in Task 3

Matches the assignment spec exactly. Both retry policies feed the *full* DB
error message back to the model in the `fix_sql` prompt — that's what makes
the loop self-correcting rather than blind.

### 5. JSON-line event log

`logs/events.jsonl` is structured. Each LLM call, each retry, each execution
gets one line. That's what the benchmark runner uses to compute mean
`attempts` and self-repair rate, and it's also what an on-call engineer would
grep when a request goes wrong in prod.

### 6. Multiset row equality for grading

`benchmark/run_eval.py` doesn't try to compare SQL strings (`JOIN ... USING`
vs `JOIN ... ON` are equivalent but lexically different). It compares the
result rows as a multiset of normalized tuples. This is the same approach
used by Spider and BIRD — it's tolerant of equivalent SQL rewrites and
intolerant of "right tables, wrong filter" mistakes.

## Limits & what's intentionally out of scope

* **No row-level security / per-user limits.** The DB credential is shared.
  This is a learning project; in production each user would have their own
  read-only Postgres role.
* **No prompt-injection defense beyond the SELECT-only validator.** If a
  customer's display name contained `DROP TABLE` it wouldn't matter because
  the validator strips strings before scanning. But more clever attacks
  (sneaking a `;-- SELECT pg_sleep(...)` past sqlparse) are out of scope.
* **No caching of LLM responses.** Each benchmark run pays the API cost. A
  production deployment would cache by (question hash, schema version).
* **Synchronous endpoints.** For a 50-question benchmark, fine; for a
  user-facing chat UI, we'd switch to `async` handlers and stream the
  summary.
