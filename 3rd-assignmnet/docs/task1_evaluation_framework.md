# Task 1 — Part 2: Evaluation Framework for the Text-to-SQL Agent

> **Question the framework must answer:** _"How do we know the agent is actually
> generating the right SQL — not just SQL that happens to run?"_

A Text-to-SQL agent can fail in seven distinct ways. This framework defines a
metric for each one, then combines them into one headline score.

---

## 1. Failure modes (what we're measuring against)

| # | Failure mode                                | Example                                              |
|---|---------------------------------------------|------------------------------------------------------|
| 1 | Syntactically invalid SQL                   | missing comma, unterminated string                   |
| 2 | Refers to non-existent table / column       | `customers.country_code` (no such column)            |
| 3 | Wrong table chosen                          | uses `orders` when question is about `payments`      |
| 4 | Wrong column projected                      | returns `buyPrice` when asked for "price" → MSRP     |
| 5 | Wrong join condition                        | `ON c.customerNumber = e.employeeNumber`             |
| 6 | Missing / wrong aggregation                 | returns rows when question asks for COUNT            |
| 7 | Right SQL, wrong rows (semantic mismatch)   | filter is `country = 'usa'` but column is uppercase  |

Mode 1 is caught by Postgres. Modes 2–5 sometimes are (when they raise
errors), but often produce *wrong-but-runnable* SQL — that's the dangerous
class. The framework below targets it specifically.

---

## 2. Per-question metrics

For each benchmark question we record:

| Metric                  | How it's computed                                                                                       | Why it matters                                                          |
|-------------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| `generated_ok`          | Bool. The agent returned non-empty SQL (didn't crash mid-LLM call).                                     | Sanity check.                                                           |
| `validated_ok`          | Bool. SQL passed the safety validator (single SELECT, no DDL/DML).                                      | Catches modes 1 & "agent tried to write" — also a security gate.        |
| `executed_ok`           | Bool. Postgres accepted the query.                                                                      | Catches modes 1, 2, sometimes 5.                                        |
| `tables_match`          | Set-equality of tables in generated SQL vs ground-truth SQL (parsed via `sqlparse`).                    | Catches mode 3.                                                         |
| `columns_match`         | Jaccard similarity of projected columns vs ground truth.                                                | Catches mode 4 (partial credit).                                        |
| `result_set_equal`      | Bool. Generated rows == ground-truth rows as **multisets** of tuples (column order- and alias-agnostic).| **The headline correctness signal.** Catches mode 7.                    |
| `scalar_equal`          | For 1×1 result, numeric-tolerant equality with ground truth.                                            | Single-source-of-truth for COUNT/SUM/AVG questions.                     |
| `retry_count`           | How many self-correction attempts the agent used (0–N).                                                 | Health metric for the loop.                                             |
| `retry_recovered`       | Bool. First attempt failed, later attempt succeeded.                                                    | Measures the value of the agentic loop.                                 |
| `latency_total_ms`      | Wall-clock from request received to summary returned.                                                   | Production-readiness.                                                   |
| `latency_db_ms`         | Sum of `execution_ms` across attempts.                                                                  | Separates LLM cost from DB cost.                                        |
| `summary_quality`       | (Optional) LLM-judge or human 1–5 rating.                                                               | Catches "right SQL, garbled English."                                   |

`result_set_equal` is intentionally permissive about column order and
aliasing: `SELECT a, b` and `SELECT b AS x, a AS y` are scored equal if
their sorted-tuple multisets match.

---

## 3. Aggregate scores

For a benchmark of N questions:

```
exact_match_rate     = mean(result_set_equal)            # the headline number
execution_success    = mean(executed_ok)
schema_grounding     = mean(tables_match AND columns_match >= 0.8)
agent_self_repair    = sum(retry_recovered) / sum(first_attempt_failed)
mean_attempts        = mean(retry_count + 1)
p50, p95 latency_ms
```

A single agent build is reported as one row:

| build       | exact_match | exec_success | schema_grounding | self_repair | mean_attempts | p95 latency |
|-------------|-------------|--------------|------------------|-------------|---------------|-------------|
| baseline    |   0.74      |     0.92     |       0.86       |    0.40     |     1.1       |   3,210 ms  |

The goal of every subsequent change is to push `exact_match` up without
collapsing `exec_success` (a model that refuses everything has 100%
exec_success vacuously, so always read the two together).

---

## 4. Why "execution match" is the right primary metric

Two competing approaches are common:

* **String / AST equality of SQL** — brittle. `JOIN ... USING (x)` vs
  `JOIN ... ON a.x = b.x` are semantically identical but lexically different.
  We'd be penalizing rewrites that don't matter.
* **Execution match on rows** — robust. If two SQL queries return the same
  rows on the same data, they're functionally equivalent for the user, even
  if their syntax differs. This is the metric Spider, BIRD, and most modern
  Text-to-SQL benchmarks use.

The downside of execution match is overfitting to the seed data (a query
filtered on `country = 'USA'` happens to return identical rows to one
filtered on `country LIKE '%S%'` on our 122-row table). We mitigate this by:

1. Keeping the ground-truth SQL in version control so reviewers can spot
   accidentally-passing queries.
2. Reporting `tables_match` and `columns_match` alongside, so a passing
   query that ignored the asked-for columns is flagged.

---

## 5. Per-category breakdown

The 50 benchmark questions span four categories. We report metrics per
category because failure shape differs between them:

| Category                         | Q range  | What typically fails                              |
|----------------------------------|----------|---------------------------------------------------|
| Simple SELECT (single table)     | 1–20     | almost never — sanity baseline                    |
| JOIN (2 tables)                  | 21–30    | wrong join condition, ambiguous column references |
| GROUP BY / aggregation per key   | 31–40    | forgets GROUP BY, picks wrong key                 |
| Scalar aggregate                 | 41–50    | column ambiguity ("price" → buyPrice vs MSRP)     |

---

## 6. How the framework is used in Tasks 3 & 4

* **Task 3** (`POST /text2sql`, one retry): we record `generated_ok`,
  `validated_ok`, `executed_ok`, `result_set_equal`, `retry_count` (0 or 1),
  `retry_recovered`, and latencies. The headline number is `exact_match_rate`.
* **Task 4** (`POST /agent/sql`, up to 3 retries): same metrics plus
  `summary_quality` and a deeper `agent_self_repair` rate (since there are
  more retry slots, this should improve).

Both surfaces are exercised by `benchmark/run_eval.py`, which iterates the
ground-truth list, calls the running FastAPI app, joins each response with
the matching ground-truth row, and writes:

* `benchmark/results_<endpoint>.json` — full per-question record
* `docs/task3_eval_report.md` / `docs/task4_eval_report.md` — markdown
  scorecard with the table from §3 plus a per-question grid

That makes the framework concrete and re-runnable: any code change can be
benchmarked with one command.
