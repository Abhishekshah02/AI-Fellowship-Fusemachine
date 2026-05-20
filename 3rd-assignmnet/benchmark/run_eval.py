"""Hit a running FastAPI server with every benchmark question, grade each
response against the ground-truth rows, and write a markdown scorecard.

Usage:
    python benchmark/run_eval.py --endpoint /agent/sql  --base-url http://localhost:8000
    python benchmark/run_eval.py --endpoint /text2sql   --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmark" / "ground_truth_results.json"


def _normalize(value):
    """Make values from JSON / DB roundtrips comparable as multisets of tuples."""
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(_normalize(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize(v)) for k, v in value.items()))
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, int):
        return value
    try:
        # numeric-looking strings — common from JSON of Decimal
        return round(float(value), 4)
    except (TypeError, ValueError):
        return str(value)


def rows_equal(a: list[list], b: list[list]) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return sorted(_normalize(a)) == sorted(_normalize(b))


def grade(expected: dict, actual: dict) -> dict:
    sql = actual.get("sql")
    status = actual.get("status")
    rowcount = actual.get("rowcount")

    # extract rows in a shape-independent way
    if isinstance(actual.get("result"), list):
        actual_rows = actual["result"]
    elif actual.get("result") is not None:
        actual_rows = [[actual["result"]]]
    else:
        actual_rows = []

    expected_rows = expected["rows"]

    executed_ok = status == "success" and sql is not None
    rows_match = rows_equal(actual_rows, expected_rows) if executed_ok else False

    return {
        "generated_ok": sql is not None,
        "validated_ok": status in {"success", "failed"} and sql is not None,
        "executed_ok": executed_ok,
        "rows_match": rows_match,
        "actual_rowcount": rowcount if rowcount is not None else len(actual_rows),
        "expected_rowcount": expected["rowcount"],
        "attempts": len(actual.get("attempts", []) or []) or 1,
        "retried": (len(actual.get("attempts", []) or []) or 1) > 1
                    or bool(actual.get("retried", False)),
        "execution_ms": actual.get("execution_ms"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--endpoint", required=True,
                        choices=["/agent/sql", "/text2sql"])
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N questions (for smoke testing)")
    args = parser.parse_args()

    label = "task4_agent" if args.endpoint == "/agent/sql" else "task3_pipeline"
    out_json = Path(args.out_json or ROOT / "benchmark" / f"results_{label}.json")
    out_md = Path(args.out_md or ROOT / "docs" / f"{label}_eval_report.md")

    questions = json.loads(RESULTS.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    url = args.base_url.rstrip("/") + args.endpoint
    print(f"Evaluating {len(questions)} questions against {url}")

    records = []
    started = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json={"question": q["question"]}, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            payload = {"status": "transport_error", "error": str(exc)}
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        scoring = grade(q, payload)
        records.append({
            "id": q["id"],
            "question": q["question"],
            "expected_sql": q["sql"],
            "actual_sql": payload.get("sql"),
            "actual_status": payload.get("status"),
            "actual_summary": payload.get("summary"),
            "client_latency_ms": elapsed,
            "scoring": scoring,
            "raw_response": payload,
        })
        flag = "OK " if scoring["rows_match"] else ("xx " if not scoring["executed_ok"] else "?? ")
        print(f"  [{flag}] Q{q['id']:02d} {q['question'][:48]:48s}  "
              f"rows={scoring['actual_rowcount']}/{scoring['expected_rowcount']}  "
              f"attempts={scoring['attempts']}  {elapsed:.0f}ms")

    total_elapsed = round(time.perf_counter() - started, 2)
    n = len(records)

    summary = {
        "endpoint": args.endpoint,
        "n_questions": n,
        "exact_match_rate": sum(r["scoring"]["rows_match"] for r in records) / n,
        "execution_success_rate": sum(r["scoring"]["executed_ok"] for r in records) / n,
        "first_try_success_rate": sum(
            1 for r in records
            if r["scoring"]["executed_ok"] and not r["scoring"]["retried"]
        ) / n,
        "self_repair_recovered": sum(
            1 for r in records
            if r["scoring"]["executed_ok"] and r["scoring"]["retried"]
        ),
        "self_repair_failed": sum(
            1 for r in records
            if not r["scoring"]["executed_ok"] and r["scoring"]["retried"]
        ),
        "mean_attempts": statistics.mean(r["scoring"]["attempts"] for r in records),
        "p50_client_ms": statistics.median(r["client_latency_ms"] for r in records),
        "p95_client_ms": sorted(r["client_latency_ms"] for r in records)[int(0.95 * n)],
        "total_wall_seconds": total_elapsed,
    }

    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2),
                        encoding="utf-8")
    out_md.write_text(_render_md(label, args.endpoint, summary, records),
                      encoding="utf-8")
    print()
    print(f"Summary: {json.dumps(summary, indent=2)}")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


def _render_md(label, endpoint, summary, records):
    lines = [
        f"# {'Task 3' if label.startswith('task3') else 'Task 4'} Evaluation Report",
        "",
        f"**Endpoint:** `POST {endpoint}`  ",
        f"**Questions:** {summary['n_questions']}  ",
        f"**Run wall-time:** {summary['total_wall_seconds']}s",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Exact-match rate (rows equal to ground truth) | **{summary['exact_match_rate']*100:.1f}%** |",
        f"| Execution success rate                       | {summary['execution_success_rate']*100:.1f}% |",
        f"| First-try success rate (no retry needed)     | {summary['first_try_success_rate']*100:.1f}% |",
        f"| Self-repair recovered (failed then succeeded)| {summary['self_repair_recovered']} |",
        f"| Self-repair failed (exhausted retries)       | {summary['self_repair_failed']} |",
        f"| Mean attempts per question                   | {summary['mean_attempts']:.2f} |",
        f"| Client-side latency p50 / p95                | {summary['p50_client_ms']:.0f} ms / {summary['p95_client_ms']:.0f} ms |",
        "",
        "## Per-question grid",
        "",
        "| # | Question | Generated SQL | Executed | Rows match | Retried | Final status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        s = r["scoring"]
        lines.append(
            f"| Q{r['id']:02d} "
            f"| {r['question']} "
            f"| {'Yes' if s['generated_ok'] else 'No'} "
            f"| {'Yes' if s['executed_ok'] else 'No'} "
            f"| {'Yes' if s['rows_match'] else 'No'} "
            f"| {'Yes' if s['retried'] else 'No'} "
            f"| {r['actual_status']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
