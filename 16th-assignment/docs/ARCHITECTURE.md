# ShopAssist AI — Architecture

W15 single-pass paths are below; the W16 agentic loop is section 0.

## 0. W16: the agentic loop

```mermaid
flowchart TB
    C([Complaint]) --> D{"Decide next action<br/>model picks ONE tool"}
    D -->|investigative| T["inspect_order · inspect_payments<br/>search_policy"]
    T --> N["Write note into CASE FILE<br/>clear the older raw payload"]
    N --> D
    D -->|"terminal: request_information"| ASK([Ask the customer])
    D -->|"terminal: escalate_case"| ESC([Ticket for a human])
    D -->|"terminal: resolve_case"| V{{"Verifier sub-agent<br/>isolated context"}}
    V -->|pass| OUT([Answer the customer])
    V -->|fail| R["Append problems<br/>to the case file"]
    R --> D
    V -->|"fail, out of revisions<br/>or verifier unreachable"| ESC
    D -.->|"step budget spent"| ESC
    T <--> DB[(orders · payments · policy chunks)]
    N --> EV[("Evidence store<br/>out of the prompt")]
    EV --> V
```

Bounds: 10 steps, 2 verifier revisions, 2 malformed-call retries. Every
exhausted bound escalates to a human rather than answering anyway.

Rendered version: [architecture-agent.png](architecture-agent.png).

---

## W15 diagrams

## 1. System overview

```mermaid
flowchart LR
    U([Customer]) --> UI["Streamlit UI<br/>:8501"]
    UI -->|"POST /chat"| API

    subgraph API["FastAPI service · :8080 · async"]
        direction TB
        RL["Rate limiter<br/>token bucket"] --> CACHE["Response cache<br/>TTL + LRU"]
        CACHE -->|miss| PAR{{"asyncio.gather"}}
        PAR --> RTR["Intent router<br/>ONNX · 11 classes"]
        PAR --> RET["Vector search<br/>top-k chunks"]
        RTR --> ORCH["Orchestrator<br/>prompt + tool loop"]
        RET --> ORCH
        ORCH --> VAL["Schema validation<br/>AssistantAnswer"]
    end

    RET <--> VDB[("Chroma<br/>MiniLM-L6-v2<br/>384-dim")]
    ORCH -->|"tier 1 or 2"| CLAUDE["Claude<br/>tools + JSON schema"]
    ORCH -->|"tier 1 or 2"| OAI["OpenAI-compatible<br/>Groq · Gemini · OpenRouter<br/>vLLM · Ollama"]
    ORCH -->|"tier 3"| DEG["Retrieved policy text<br/>+ escalate"]
    ORCH <--> TOOLS["Tools<br/>lookup_order · search_kb<br/>create_support_ticket"]
    TOOLS <--> DB[("orders.json<br/>tickets.jsonl")]
    VAL --> UI

    KB[/"Knowledge base<br/>.md · .txt · .pdf"/] --> ING["Ingest → chunk → embed"] --> VDB
```

## 2. Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as FastAPI
    participant R as ONNX router
    participant V as Chroma
    participant L as LLM provider
    participant T as Tools

    C->>A: POST /chat {message, history}
    A->>A: rate limit → cache lookup
    par concurrent
        A->>R: classify(message)
        R-->>A: (REFUND, 0.92)
    and
        A->>V: query(message, k=4)
        V-->>A: 4 chunks + scores
    end
    A->>L: system + context + intent prior + tools + json_schema
    L-->>A: tool_use(lookup_order)
    A->>T: execute
    T-->>A: order JSON
    A->>L: tool_result
    L-->>A: schema-valid JSON answer
    A->>A: validate → cache → record metrics
    A-->>C: 200 ChatResponse
```

## 3. Degradation ladder

`PROVIDER` decides which of the two real providers is tier 1; the other is tier 2.
They are peers — both call tools and both return schema-valid JSON — so a fallback is
a complete answer, not a reduced one.

| Tier | Provider | Tools | Structured JSON | Triggered when |
|------|----------|-------|-----------------|----------------|
| 1 | the one named by `PROVIDER` | yes | constrained decoding | normal operation |
| 2 | the other one | yes | constrained decoding | tier 1 errors after N retries, or its key is unset |
| 3 | none | no | built server-side from the schema | both providers down |

Structured output differs per API and is handled per provider:

| Provider | Mechanism |
|---|---|
| Claude | `output_config.format.json_schema` — always available |
| OpenAI-compatible | `response_format` `json_schema` → `json_object` → prompt instruction, negotiated at runtime on the first `400` and remembered |

Tier 3 still returns a valid `ChatResponse`: the closest policy text, `escalate: true`,
`confidence: 0.0`. The client contract never breaks, so the UI never has to special-case
an outage.

## 4. Component decisions

| Concern | Choice | Why |
|---|---|---|
| Vector DB | Chroma, persistent (SQLite + HNSW) | embedded, no extra service, cosine HNSW, swap for a server by changing one line in `rag.py` |
| Embeddings | `all-MiniLM-L6-v2` via onnxruntime (Chroma default) | 384-dim, CPU-only, no torch in the serving image, no API cost |
| Chunking | markdown-heading split, then 900/150 sliding window | keeps one policy topic per chunk; the window only kicks in for long sections |
| Intent router | W14 classifier exported to ONNX | classification is not a job for a 1M-context LLM: ~1 ms on CPU vs ~1 s and a token bill |
| Providers | one interface, two implementations | Claude and OpenAI-compatible are peers; the fallback ladder is just the list reversed |
| Retries | tenacity, in the provider layer | one visible policy; both SDKs' own retries are disabled (`max_retries=0`) so behaviour is not doubled |
| Rate limit | in-process token bucket | one replica, no Redis; swap for a shared bucket when you run more than one |
| Cache | `cachetools.TTLCache` keyed on normalised message + history | repeat FAQs return in ~1 ms; failures are never cached |
| Concurrency | async FastAPI, blocking work in `asyncio.to_thread` | Chroma and onnxruntime are sync — off the loop they don't block other requests |

## 5. Deployment topology

```mermaid
flowchart TB
    subgraph host["docker compose"]
        ui["ui · streamlit :8501"] --> api["api · uvicorn :8080"]
        api --> vol1[("volume: chroma")]
        api --> vol2[("volume: tickets")]
        api -.->|profile gpu| vllm["vllm :8000<br/>NVIDIA GPU"]
    end
    api -->|https| anthropic["Claude API"]
    api -->|https| hosted["Groq / Gemini / OpenRouter<br/>OpenAI-compatible"]
```

The `vllm` service sits behind the `gpu` profile: `docker compose up` runs the
assistant on any laptop, `docker compose --profile gpu up` adds the local
open-source fallback on a GPU host.
