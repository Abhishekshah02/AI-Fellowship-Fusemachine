# Deliverables — W15 Problem Set

Every bullet from the assignment PDF, and the file that satisfies it.
All paths are relative to `15th-assigment/`.

---

## Task 1 — Build an AI Assistant (Applied AI)

### Required deliverables

| # | PDF deliverable | Where it is |
|---|---|---|
| 1 | **Source code** | [`app/`](app/) — 9 modules · [`scripts/`](scripts/) · [`tests/`](tests/) |
| 2 | **Dockerfile** | [`Dockerfile`](Dockerfile) |
| 3 | **README** | [`README.md`](README.md) |
| 4 | **Architecture diagram** | [`docs/architecture.png`](docs/architecture.png) · [`docs/architecture.svg`](docs/architecture.svg) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (4 Mermaid diagrams) |

### Core functionality

| Requirement | Implementation | File |
|---|---|---|
| **LLM Integration** — connect to a major provider | Claude via the Anthropic SDK, plus any OpenAI-compatible endpoint (Groq / Gemini / OpenRouter / vLLM). `PROVIDER` selects which leads. | `app/llm.py` |
| **Prompt Engineering** — system prompts | `SYSTEM_PROMPT`: a six-rule contract (ground every claim, cite chunk ids, prefer tools, escalation rules, never handle card data, length limit). `JSON_INSTRUCTION` is deliberately separate. | `app/llm.py` |
| **Prompt Engineering** — tune `temperature` / `top_p` | `TEMPERATURE=0.2`, `TOP_P=0.9`, sent to every model that accepts them. `supports_sampling()` routes thinking-era Claude models to `output_config.effort` instead, because they reject the sampling params with a 400. | `app/config.py`, `app/llm.py` |
| **Structured Output** — valid JSON | `AssistantAnswer` is the single contract; `json_schema()` renders it closed (`additionalProperties: false`, all fields required) and it constrains decoding — `output_config.format` on Claude, `response_format` on the OpenAI path. Validated with Pydantic before the response leaves the process. | `app/schemas.py`, `app/llm.py` |
| **Tool Calling** — function calling to external tools | 3 tools, bounded loop, implemented for **both** provider APIs: `lookup_order`, `search_knowledge_base`, `create_support_ticket`. | `app/tools.py`, `app/llm.py` |

### Technical implementation

| Requirement | Implementation | File |
|---|---|---|
| **Document ingestion & chunking** | `.md` / `.txt` / `.pdf` → split on markdown headings, then a 900-char / 150-overlap window for long sections. Ids are `file#position`, so re-ingest replaces rather than duplicates. | `app/rag.py` |
| **Embeddings in a vector database** | Chroma, persistent, cosine HNSW. `all-MiniLM-L6-v2` (384-dim) on onnxruntime — no torch, no API cost. 6 documents → 35 chunks. | `app/rag.py`, `data/kb/` |
| **Local deployment — serve an open-source model with vLLM** | `vllm` service serving `mistralai/Mistral-7B-Instruct-v0.3` on its OpenAI-compatible endpoint; the app talks to it through the same code path as any hosted provider. Behind the `gpu` compose profile so the stack still runs without a GPU. | `docker-compose.yml`, `app/llm.py` |
| **Containerization with Docker** | One image, non-root user, cached dependency layer, embedding model **and** vector index baked in at build time, `HEALTHCHECK`. | `Dockerfile` |

---

## Task 2 — Productionize the AI Assistant

### Required deliverables

| # | PDF deliverable | Where it is |
|---|---|---|
| 1 | **Updated source code** | [`app/`](app/) — adds `ui.py`, `router.py` (ONNX), `reliability.py`; [`scripts/bench.py`](scripts/bench.py) |
| 2 | **Docker Compose configuration** | [`docker-compose.yml`](docker-compose.yml) — `api`, `ui`, GPU-profiled `vllm`, named volumes |
| 3 | **Architecture diagram** | [`docs/architecture.png`](docs/architecture.png) — includes the deployment topology band |

### Application

| Requirement | Implementation | File |
|---|---|---|
| Simple web UI | Streamlit chat: history, raw-JSON panel per answer, per-message badges (intent / provider / latency / cached / escalated), live health panel, re-index and metrics buttons. | `app/ui.py` |
| UI connected to the AI backend | Talks to the API over HTTP only — it holds no key, no model and no index, so the two scale separately. | `app/ui.py`, `docker-compose.yml` |

### Model optimization

| Requirement | Implementation | File |
|---|---|---|
| Convert the model to ONNX | The W14 intent router is trained and exported to ONNX; served by **onnxruntime alone — torch is not in the serving image**. Shipped artifact: 429 KB, **0.9981 accuracy / 0.9985 macro-F1** on 2,688 held-out Bitext messages. | `scripts/train_router.py`, `app/router.py`, `models/router/` |
| …or justify why not applicable | The *generative* model is not converted, and the README says why: Claude is a hosted API, and a 7B decoder is served by vLLM, whose PagedAttention + continuous batching beats an ONNX export for autoregressive serving. ONNX earns its place on the small classifier that runs on **every** request — ~1 ms CPU and zero tokens instead of an LLM call. | `README.md` § Model optimisation |
| Inference optimizations | CPU execution provider, 128-token truncation, `lru_cache`d session, confidence floor below which the router stays silent. | `app/router.py` |

### Performance engineering

| Requirement | Implementation | File |
|---|---|---|
| Concurrent / async request handling | Async end to end; Chroma and onnxruntime are synchronous so they run in `asyncio.to_thread` and never block the loop. | `app/main.py`, `app/rag.py` |
| Optimize latency and throughput | Intent routing and vector retrieval are independent → one `asyncio.gather`, so a request pays for the slower, not the sum. Prompt caching on the frozen Claude prefix. Benchmark script reports throughput and p50/p95. | `app/main.py`, `app/llm.py`, `scripts/bench.py` |
| Prompt/response caching *(bonus)* | `cachetools.TTLCache` keyed on normalised message + history. Measured **0 ms** on a repeat question. Failures are never cached. | `app/reliability.py`, `app/main.py` |

### Reliability

| Requirement | Implementation | File |
|---|---|---|
| Retry mechanism | `tenacity`, exponential backoff with jitter, `MAX_RETRIES=3`, only on genuinely transient errors — and the predicate knows **both** SDKs' exception classes, so a provider 429 cannot slip past. Both SDKs' built-in retries are disabled so the policy exists in one place. | `app/llm.py` |
| Rate limiting | Async token bucket; returns `429` with a `Retry-After` header. | `app/reliability.py`, `app/main.py` |
| Fallback model/provider | Three tiers: primary → the other provider → degraded. Both real tiers have tools and structured output, so a fallback is a full answer. | `app/llm.py` |
| Error handling & graceful degradation | Tool failures return to the model as `{"error": ...}` instead of raising; a refusal or unparseable answer demotes a tier; with every provider down the API still returns a valid `ChatResponse` carrying the closest policy text and `escalate: true`. Provider quirks (structured-output modes, charset-less JSON) are negotiated and repaired rather than crashed on. | `app/llm.py`, `app/tools.py` |

### Deployment

| Requirement | Implementation | File |
|---|---|---|
| Dockerize the complete application | `api` + `ui` from one image; `vllm` opt-in. | `Dockerfile`, `docker-compose.yml` |
| Deployment instructions | Quick start, per-provider key setup, compose commands, and scale-out notes (Redis for the limiter and cache, Chroma server for the index). | `README.md` §1 and §4 |
| *(Bonus)* Deploy to Azure / AWS / GCP | **Not deployed.** Azure Container Apps commands are given, but no cloud deployment was performed. | `README.md` § Deployment |

---

## Verification status

| Checked | How |
|---|---|
| 20 automated tests pass | `pytest -q` — no API key or network needed; covers both tool loops with stubbed clients |
| Live end-to-end on a real provider | 4 scenarios through Groq: RAG answer with citation, order lookup via tool call, two-tool escalation that opened ticket `TCK-951FBE1E`, and a cancellation check. Cache hit measured at 0 ms. See README § Verified end-to-end runs |
| ONNX router | Trained, exported, and serving — `models/router/metrics.json` |
| Docker image | `docker build` completes, all layers including the in-image index build |
| `docker compose` config | `docker compose config` validates; both profiles resolve |
| vLLM service | **Not run** — no NVIDIA GPU on the development machine. The code path it uses is the same one exercised live against Groq. |

---

## Running it

Full instructions in [`README.md`](README.md). Shortest path:

```bash
cp .env.example .env     # paste a free Groq key: https://console.groq.com/keys
docker compose up --build
```

Then open <http://localhost:8501>.

`.env` is **not** committed — it holds the API key. `.env.example` is the template,
with four free providers documented. With no key at all the app still starts and
answers from the degraded tier.
