"""FastAPI service.

Request path: rate limit -> cache -> (intent router || vector search, concurrently)
-> LLM with tools -> validated JSON out.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response

from . import agent, llm, rag, router
from .config import settings
from .reliability import cache_key, limiter, metrics, response_cache
from .schemas import (
    ChatRequest,
    ChatResponse,
    InvestigateRequest,
    InvestigateResponse,
    json_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("shopassist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = settings()
    # Warm the embedding model + index once, at boot, not on the first customer.
    if rag.collection().count() == 0:
        log.info("empty index, ingesting %s", s.kb_dir)
        log.info("ingested: %s", await asyncio.to_thread(rag.ingest))
    log.info("onnx intent router: %s", "loaded" if router.available() else "absent (LLM classifies)")
    yield


app = FastAPI(
    title="ShopAssist AI",
    version="1.0.0",
    description="RAG + tool-calling customer support assistant.",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    s = settings()
    return {
        "status": "ok",
        "index_chunks": rag.collection().count(),
        "provider": s.provider,
        "claude_model": s.primary_model,
        "claude_key_configured": bool(s.anthropic_api_key),
        "openai_model": s.openai_model,
        "openai_base_url": s.openai_base_url,
        "onnx_router": router.available(),
    }


@app.get("/metrics")
async def get_metrics():
    return {**metrics.snapshot(), "cache_entries": len(response_cache)}


@app.get("/schema")
async def get_schema():
    """The JSON schema every /chat answer is constrained to."""
    return json_schema()


@app.post("/ingest")
async def ingest():
    return await asyncio.to_thread(rag.ingest)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response) -> ChatResponse:
    started = time.perf_counter()
    metrics.bump("requests")

    if not await limiter.take():
        metrics.bump("rate_limited")
        raise HTTPException(
            status_code=429,
            detail="Too many requests.",
            headers={"Retry-After": str(limiter.retry_after())},
        )

    key = cache_key(req.message, req.history)
    if (hit := response_cache.get(key)) is not None:
        metrics.bump("cache_hits")
        response.headers["X-Cache"] = "HIT"
        return hit.model_copy(
            update={"cached": True, "latency_ms": int((time.perf_counter() - started) * 1000)}
        )

    # Retrieval and intent routing are independent -- run them together.
    hits, routed = await asyncio.gather(
        rag.asearch(req.message),
        asyncio.to_thread(router.classify, req.message),
    )

    answer, provider, model, degraded_reason = await llm.generate(
        req.message, req.history, hits, rag.as_context(hits), routed[0] if routed else None
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    metrics.observe(latency_ms)
    out = ChatResponse(
        **answer.model_dump(),
        model=model,
        provider=provider,
        cached=False,
        latency_ms=latency_ms,
        router_intent=routed[0] if routed else None,
        router_confidence=routed[1] if routed else None,
        degraded_reason=degraded_reason,
    )
    if provider != "degraded":  # never cache a failure
        response_cache[key] = out
    response.headers["X-Cache"] = "MISS"
    return out


@app.post("/investigate", response_model=InvestigateResponse)
async def investigate(req: InvestigateRequest, response: Response) -> InvestigateResponse:
    """W16 agentic loop: investigate a dispute, verify the answer, then respond.

    Deliberately not cached -- an investigation has side effects (it can open a
    ticket) and its trajectory is the point.
    """
    metrics.bump("agent_requests")

    if not await limiter.take():
        metrics.bump("rate_limited")
        raise HTTPException(
            status_code=429,
            detail="Too many requests.",
            headers={"Retry-After": str(limiter.retry_after())},
        )

    result = await agent.investigate(req.complaint, verify=req.verify)
    metrics.bump(f"agent_{result.outcome}")
    metrics.observe(result.latency_ms)
    response.headers["X-Agent-Iterations"] = str(result.iterations)
    return InvestigateResponse(**vars(result))
