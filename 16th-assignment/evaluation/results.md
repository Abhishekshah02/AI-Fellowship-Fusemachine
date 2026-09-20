# Agent evaluation results

Generated 2026-09-11 06:10 UTC · model `openai/gpt-oss-120b` · temperature 0.2 · max_steps 10 · max_revisions 2 · run took 16.1 min

Produced by `python -m evaluation.harness`. Every check is deterministic (string and schema assertions, no LLM judge).

## Headline

| Metric | Multi-agent (with verifier) | Single-agent baseline |
|---|---|---|
| Task completion rate | 7/7 (100%) | 3/7 (43%) |
| Mean iterations | 4.6 | 5.4 |
| Mean tokens / query | 10218 | 8623 |
| Mean latency | 33378 ms | 13629 ms |
| Tool-call correctness | 27/27 (100%) | — |

**Coordination cost.** The verifier adds **+19%** tokens (8623 → 10218 per query), measured over the 7 case(s) run both ways. What that buys is in the failure table below.

## Per-case results (multi-agent)

| Case | Outcome | Steps (budget) | Tools used | Tokens | Verifier rejections | Result |
|---|---|---|---|---|---|---|
| `dup_charge` | resolved | 5 (7) | `inspect_order`, `inspect_payments`, `search_policy`, `resolve_case` | 11638 | 1 | pass |
| `refund_pending_in_window` | resolved | 4 (7) | `inspect_order`, `inspect_payments`, `search_policy`, `resolve_case` | 7896 | 0 | pass |
| `refund_already_paid` | resolved | 6 (7) | `inspect_order`, `inspect_payments`, `search_policy`, `inspect_payments`, `search_policy`, `resolve_case` | 10707 | 0 | pass |
| `failed_authorisation` | resolved | 7 (7) | `inspect_order`, `search_policy`, `inspect_payments`, `search_policy`, `resolve_case` | 19218 | 2 | pass |
| `missing_order_id` | needs_input | 1 (4) | `request_information` | 1128 | 0 | pass |
| `unknown_order` | needs_input | 2 (4) | `inspect_order`, `request_information` | 2226 | 0 | pass |
| `lost_parcel_escalation` | resolved | 7 (7) | `inspect_order`, `inspect_payments`, `search_policy`, `search_policy`, `resolve_case` | 18714 | 2 | pass |

## Trajectory length

Whether the number of steps is reasonable for the complexity of the query.

| Case | Steps | Budget | Within budget |
|---|---|---|---|
| `dup_charge` | 5 | 7 | yes |
| `refund_pending_in_window` | 4 | 7 | yes |
| `refund_already_paid` | 6 | 7 | yes |
| `failed_authorisation` | 7 | 7 | yes |
| `missing_order_id` | 1 | 4 | yes |
| `unknown_order` | 2 | 4 | yes |
| `lost_parcel_escalation` | 7 | 7 | yes |

## Token cost per query

| Case | Investigator | Verifier | Total | Single-agent | Overhead |
|---|---|---|---|---|---|
| `dup_charge` | 8013 | 3625 | 11638 | 17136 | -32% |
| `refund_pending_in_window` | 6147 | 1749 | 7896 | 9431 | -16% |
| `refund_already_paid` | 9154 | 1553 | 10707 | 9187 | +17% |
| `failed_authorisation` | 13425 | 5793 | 19218 | 13965 | +38% |
| `missing_order_id` | 1128 | 0 | 1128 | 1139 | -1% |
| `unknown_order` | 2226 | 0 | 2226 | 2230 | -0% |
| `lost_parcel_escalation` | 13057 | 5657 | 18714 | 7270 | +157% |

## Failure log

| Case | Mode | Type | Why |
|---|---|---|---|
| `dup_charge` | single_agent | **hard** | no conclusion within 7 steps |
| `refund_pending_in_window` | single_agent | **cascading_soft** | failed checks: outcome |
| `refund_already_paid` | single_agent | **hard** | no conclusion within 7 steps |
| `lost_parcel_escalation` | single_agent | **hard** | no conclusion within 7 steps |
| `dup_charge_no_ledger` | fault_injection | **hard** | no conclusion within 6 steps |

Taxonomy: **provider_unavailable** = the provider refused the request (rate limit / daily quota); the agent never acted, so it is excluded from the rates. **hard** = no usable output (run errored or step budget exhausted). **soft** = an answer was produced but it is wrong (unacceptable terminal state, unsupported claim, or missing evidence). **cascading soft** = the wrong answer follows from an earlier broken or skipped step rather than from the final reasoning.

## Failure injection: payment ledger unavailable

`inspect_payments` raises `ToolUnavailable` for this run. The duplicate charge is impossible to confirm, so the only honest outcomes are to escalate or ask the customer.

- Outcome: **exhausted** after 5 steps
- Behaved correctly: **no — no conclusion within 6 steps**
- Trajectory:
  1. inspect_order(SA-10231) -> status=SHIPPED, placed=2026-08-28, total=$189.5, payment=Visa ****4412, refund=none recorded
  2. inspect_payments -> UNAVAILABLE: payment ledger service is unreachable (injected fault)
  3. search_policy('duplicate charge refund policy') -> 5 chunks: payments_and_invoices.md#5, payments_and_invoices.md#4, refunds_and_returns.md#4, refunds_and_returns.md#0, refunds_and_returns.md#5
  4. inspect_payments REPEATED -- you already ran this at step 2 and the result has not changed. Do NOT run it again. Either conclude with what you have, or take a genuinely different action.
  5. search_policy REPEATED -- you already ran this at step 3 and the result has not changed. Do NOT run it again. Either conclude with what you have, or take a genuinely different action.

Answer given to the customer:

> I could not get to the bottom of this within my investigation limit, so I have handed it to a human agent (ticket TCK-269359E4).

## Context engineering effect

Bytes of tool output that never re-entered the prompt because payloads older than the most recent step were cleared and replaced by their notes:

| Case | Steps | Chars cleared |
|---|---|---|
| `dup_charge` | 5 | 3,371 |
| `refund_pending_in_window` | 4 | 2,525 |
| `refund_already_paid` | 6 | 3,306 |
| `failed_authorisation` | 7 | 6,832 |
| `missing_order_id` | 1 | 0 |
| `unknown_order` | 2 | 68 |
| `lost_parcel_escalation` | 7 | 6,040 |
