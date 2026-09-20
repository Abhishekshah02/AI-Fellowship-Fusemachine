# Deliverables — W16 (Task 3: Agentify the Assistant)

Every requirement from the W16 PDF and the file that satisfies it.
Paths are relative to `16th-assignment/`.

**Separation from W15.** W15 lives in `../15th-assigment/` and is untouched —
that folder is exactly what was submitted for W15. This folder is a complete,
standalone copy of it *plus* the agentic feature, so each assignment can be
reviewed and run on its own. [§7 of the README](README.md#7-what-carried-over-from-w15)
lists precisely which files are new.

---

## Required deliverables

| # | PDF deliverable | Where |
|---|---|---|
| 1 | **Updated source code** — the W15 assistant | `app/` (`main.py`, `llm.py`, `rag.py`, `router.py`, `tools.py`, `reliability.py`, `schemas.py`, `config.py`, `ui.py`) |
| 1 | **Updated source code** — the new agentic feature | `app/agent.py`, `app/agent_tools.py`, `app/verifier.py`, `app/tokens.py`, `data/payments.json` |
| 2 | **Updated README** — documentation requirements (a, b, c) | [`README.md` § W16 write-up](README.md#w16-write-up) |
| 2 | **Updated README** — additional requirements 1-4 | same section, one page total |
| 3 | **Updated architecture diagram** — shows the agentic loop | [`docs/architecture-agent.png`](docs/architecture-agent.png) (source `.svg`) |
| 3 | …and the multi-agent coordination structure | same diagram: investigator loop, verifier sub-agent, pass/fail paths, escalation |
| 4 | **Evaluation harness** — source code | [`evaluation/harness.py`](evaluation/harness.py), cases in [`evaluation/cases.py`](evaluation/cases.py) |
| 4 | **Evaluation harness** — results report | [`evaluation/results.md`](evaluation/results.md) (+ `results.json`) |

---

## Core functionality

| Requirement | How it is met | Where |
|---|---|---|
| **Add an agentic feature** that a fixed single-pass pipeline cannot handle | Dispute investigation: the agent reads an order, then the payment ledger, then the policy that applies to what it found, and chooses its own ending. Not the W15 RAG pipeline in a loop — it is a different toolset over different data (`payments.json`) with model-chosen control flow. | `app/agent.py` |
| **One sentence on why a fixed pipeline is not sufficient** | Stated in the README before the implementation section. | [README](README.md#the-feature-and-why-a-fixed-pipeline-cannot-do-it) |
| Loop **runs more than one iteration** | Typically 4-7 iterations; the harness reports trajectory length per case. | `evaluation/results.md` |
| Model decides **search again / another tool / ask the user** | Six tools: three investigative (loop continues) and three terminal (`resolve_case`, `request_information`, `escalate_case`). `request_information` is the "ask the user" branch and is exercised by the `missing_order_id` case. | `app/agent_tools.py` |
| **Clearly defined stopping condition**, cannot run indefinitely | `AGENT_MAX_STEPS` (10) hard-caps iterations; `AGENT_MAX_REVISIONS` (2) caps verifier round-trips; repeated malformed tool calls are capped at 2. Hitting any limit escalates to a human instead of answering. The model is also told how many steps remain and must conclude on the last one. | `app/agent.py`, `app/config.py` |

## Context engineering

| Requirement | How it is met | Where |
|---|---|---|
| Apply **at least one technique** where it gives a clear benefit | **Clearing tool results**, carried by structured notes: raw payloads older than the current step leave the prompt and are replaced by a deterministic one-line digest; full payloads move to the evidence store for the verifier. | `app/agent.py:_messages()`, `app/agent_tools.py` |
| Document **which technique, where, and what problem it solves** | README section (a), written from the actual failure it fixed (context growth hitting the provider's tokens-per-minute limit mid-investigation), not from the general concept. | [README § a](README.md#a-context-engineering-technique--clearing-tool-results-carried-by-structured-notes) |
| Measure its effect | Harness reports chars withheld from the prompt per case. | `evaluation/results.md` § Context engineering effect |

## Documentation requirements

| Section | Where |
|---|---|
| **a. Context engineering technique** (which / where / what problem) | [README § a](README.md#a-context-engineering-technique--clearing-tool-results-carried-by-structured-notes) |
| **b. Agentic pattern** (single vs multi-agent + why, using the class frameworks) | [README § b](README.md#b-agentic-pattern--multi-agent-investigator--verifier) — multi-agent, justified by context isolation and the self-verification paradox, with the sequential-bottleneck trade-off stated |
| **c. Evaluation harness** (built from scratch, four minimum metrics) | [README § c](README.md#c-evaluation-harness) + `evaluation/` |

## Additional requirements

| # | Requirement | Where |
|---|---|---|
| 1 | **Skill vs. agent** — one sentence on whether a Skill would have sufficed | [README](README.md#additional-requirement-1--skill-vs-agent) — partly; the static know-how is prompt content, but a Skill cannot make control flow data-dependent |
| 2 | **Token and cost accounting** per query, multi-agent vs single-agent baseline | `app/tokens.py`; harness runs every case twice and prints per-case + mean overhead | 
| 3 | **Failure injection test** and how the agent responds | `agent_tools.FAULTS` + `ToolUnavailable`; harness runs it automatically; trajectory and answer in `evaluation/results.md` § Failure injection |
| 4 | **Tool vs. agent boundary** paragraph | [README](README.md#additional-requirement-4--tool-vs-agent-boundary) — the payment system is modelled as a bounded tool call; the reasoning, and what would change the decision, are given |
| 5 | **Write-up ≈ one page** | The whole "W16 write-up" section is one page; running instructions and file maps are outside it |

---

## Verification status

| Checked | How |
|---|---|
| 35 automated tests pass | `pytest -q` — 20 from W15 plus 15 new for the agent, all offline with a stubbed provider: multi-iteration loop, terminal actions, verifier rejection → revision → escalation, unverifiable answer → escalation, step budget as a hard stop, context clearing, fault injection, token ledger, harness scoring and taxonomy |
| Agent verified live | Run against Groq (`openai/gpt-oss-20b`); trajectories, verdicts and token counts in `evaluation/results.md` |
| Evaluation harness | Run end to end, including the single-agent baseline and the failure-injection case |
| Failure injection | Ledger made unavailable; scored result recorded in the report |

## Running it

```bash
cp .env.example .env          # free Groq key: https://console.groq.com/keys
docker compose up --build     # api :8080, ui :8501 (UI has Chat and Investigate modes)

python -m evaluation.harness  # regenerates evaluation/results.md
pytest -q                     # 35 offline tests
```

`.env` is not committed — it holds the API key. `.env.example` is the template.
