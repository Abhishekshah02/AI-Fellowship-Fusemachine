"""Concurrency benchmark: fire N requests at /chat with a fixed concurrency and
report latency percentiles and throughput.

    python scripts/bench.py --n 40 --concurrency 8
    python scripts/bench.py --n 40 --concurrency 8 --unique   # defeat the cache
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

PROMPTS = [
    "Where is my order SA-10231?",
    "How long do refunds take on a credit card?",
    "Can I cancel order SA-10244?",
    "My refund for SA-10099 still has not arrived and I am furious.",
    "Do you ship internationally and who pays the import duty?",
    "How do I change the delivery address after ordering?",
    "What does ShopAssist Plus cost and what do I get?",
    "My payment was declined twice, what now?",
]


async def one(client: httpx.AsyncClient, msg: str) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    r = await client.post("/chat", json={"message": msg})
    ms = (time.perf_counter() - t0) * 1000
    provider = r.json().get("provider", "?") if r.status_code == 200 else str(r.status_code)
    return r.status_code, ms, provider


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--unique", action="store_true", help="suffix each prompt to bypass the cache")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    msgs = [PROMPTS[i % len(PROMPTS)] for i in range(args.n)]
    if args.unique:
        msgs = [f"{m} (ref {i})" for i, m in enumerate(msgs)]

    async with httpx.AsyncClient(base_url=args.url, timeout=180) as client:

        async def guarded(m: str):
            async with sem:
                return await one(client, m)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(m) for m in msgs))
        wall = time.perf_counter() - t0

    ok = [ms for code, ms, _ in results if code == 200]
    providers: dict[str, int] = {}
    for code, _, prov in results:
        providers[prov] = providers.get(prov, 0) + 1

    print(f"requests      : {args.n} at concurrency {args.concurrency}")
    print(f"succeeded     : {len(ok)}")
    print(f"wall clock    : {wall:.2f} s")
    print(f"throughput    : {len(ok) / wall:.2f} req/s")
    if ok:
        ok.sort()
        print(f"latency p50   : {statistics.median(ok):.0f} ms")
        print(f"latency p95   : {ok[min(len(ok) - 1, int(0.95 * len(ok)))]:.0f} ms")
        print(f"latency max   : {ok[-1]:.0f} ms")
    print(f"outcomes      : {providers}")


if __name__ == "__main__":
    asyncio.run(main())
