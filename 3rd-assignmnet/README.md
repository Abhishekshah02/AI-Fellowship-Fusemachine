# 3rd Assignment — Agentic Text-to-SQL System

FastAPI + PostgreSQL + Google Gemini (free tier). Covers Tasks 1–4 of Week 3 in one repo.

| Task | Deliverable                                                                                  |
|------|----------------------------------------------------------------------------------------------|
| 1    | `docs/task1_ground_truth.md` (50 questions + verified SQL + live results) and `docs/task1_evaluation_framework.md` |
| 2    | `docs/task2_decomposition.md`                                                                |
| 3    | `POST /text2sql` — linear pipeline with one retry                                            |
| 4    | `POST /agent/sql` — agentic loop with up to 3 self-correction attempts + NL summary          |
| Eval | `benchmark/run_eval.py` produces `docs/task3_eval_report.md` and `docs/task4_eval_report.md` |

See `docs/architecture.md` for the design walk-through.

---

## Prerequisites

* Python 3.11+
* Docker Desktop (used to bring up the classicmodels Postgres from
  `../2nd-assignment/`)
* A free `GOOGLE_API_KEY` from https://aistudio.google.com/app/apikey

## One-time setup

```bash
# 1. start the classicmodels DB (from the 2nd-assignment folder)
cd ../2nd-assignment
docker compose up -d

# 2. set up this project
cd ../3rd-assignmnet
python -m venv .venv
source .venv/Scripts/activate           # (on Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                    # then edit .env and paste your GOOGLE_API_KEY
```

## Run

```bash
# regenerate Task 1 deliverable (executes every ground-truth SQL live)
python benchmark/build_task1_doc.py

# start the API
uvicorn app.main:app --reload --port 8000
#  -> http://localhost:8000/docs    interactive Swagger UI
#  -> POST /text2sql                Task 3
#  -> POST /agent/sql               Task 4
```

Quick try with curl:

```bash
curl -X POST http://localhost:8000/agent/sql \
     -H 'content-type: application/json' \
     -d '{"question":"Count customers per country"}'
```

Sample response (shape — actual numbers from your DB):

```json
{
  "question": "Count customers per country",
  "decomposition": {
    "intent": "Count customers grouped by country",
    "tables": ["customers"],
    "columns": ["customers.country"],
    "filters": [],
    "joins": [],
    "aggregations": ["COUNT(*)"],
    "group_by": ["customers.country"]
  },
  "sql": "SELECT \"country\", COUNT(*) AS \"customerCount\" FROM customers GROUP BY \"country\" ORDER BY \"customerCount\" DESC",
  "result": [["USA", 36], ["Germany", 13], ["France", 12], ...],
  "columns": ["country", "customerCount"],
  "rowcount": 27,
  "summary": "Customers are spread across 27 countries; the USA leads with 36, followed by Germany (13) and France (12).",
  "status": "success",
  "attempts": [{"attempt": 1, "sql": "SELECT ...", "error": null}],
  "execution_ms": 412.5
}
```

## Run the benchmark

The server must be running. In a second shell:

```bash
source .venv/Scripts/activate

# Task 4 (agent, 3 retries) — recommended
python benchmark/run_eval.py --endpoint /agent/sql

# Task 3 (linear pipeline, 1 retry)
python benchmark/run_eval.py --endpoint /text2sql

# Smoke (just first 5 questions)
python benchmark/run_eval.py --endpoint /agent/sql --limit 5
```

Each run writes:

* `benchmark/results_<task>.json` — raw per-question records
* `docs/<task>_eval_report.md` — markdown scorecard (headline metrics + per-question grid)

## Tests

```bash
pytest -q
```

Smoke tests don't need a Gemini key — they exercise the safety validator
and the live DB executor only.

## Project layout

```
3rd-assignmnet/
├── app/
│   ├── main.py              # FastAPI app + routes
│   ├── config.py            # env loading
│   ├── database.py          # SQLAlchemy engine + session
│   ├── schema_info.py       # frozen schema description for prompts
│   ├── logger.py            # JSON-line event log + stdout/file
│   ├── validator.py         # SELECT-only safety gate
│   ├── executor.py          # runs SQL, returns columns/rows/timing
│   ├── llm_client.py        # Gemini wrapper
│   ├── prompts.py           # loads templates from prompts/
│   ├── decomposer.py        # Task 2: question -> structured plan
│   ├── sql_generator.py     # Task 3: plan -> SQL, and fix-on-error
│   ├── nl_summarizer.py     # Task 4: rows -> one-line answer
│   ├── agent.py             # Task 4: the self-correcting loop
│   └── routers/
│       ├── pipeline_router.py     # POST /text2sql
│       └── agent_router.py        # POST /agent/sql
├── prompts/
│   ├── decompose.txt
│   ├── generate_sql.txt
│   ├── fix_sql.txt
│   └── summarize.txt
├── benchmark/
│   ├── ground_truth.json           # 50 questions + reference SQL
│   ├── ground_truth_results.json   # cached results from running ground-truth SQL
│   ├── build_task1_doc.py          # builds Task 1 markdown
│   └── run_eval.py                 # grades a live server against ground truth
├── docs/
│   ├── task1_ground_truth.md
│   ├── task1_evaluation_framework.md
│   ├── task2_decomposition.md
│   ├── architecture.md
│   ├── task3_pipeline_eval_report.md   # produced by run_eval.py
│   └── task4_agent_eval_report.md      # produced by run_eval.py
├── tests/
│   └── test_smoke.py
├── logs/                    # app.log + events.jsonl (gitignored)
├── requirements.txt
└── .env.example
```
