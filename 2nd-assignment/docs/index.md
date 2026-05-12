# Classic Models Customer API

Layered FastAPI application connected to a PostgreSQL database that ships in a
Docker container. Implements Twelve-Factor principles: config in `.env`, the
database as a backing service, async concurrency for aggregated counts.

## Run

```powershell
docker compose up -d
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs>.

## Endpoints

- `GET /customers/` — paginated list
- `GET /customers/{customerNumber}` — one customer (404 if missing)
- `POST /customers/` — create
- `PUT /customers/{customerNumber}` — partial update
- `DELETE /customers/{customerNumber}` — delete
- `GET /customers/{customerNumber}/orders` — related orders
- `GET /customers/{customerNumber}/payments` — related payments
- `GET /{table}/count` — row count per table (8 tables)
- `GET /overall_counts` — all 8 counts concurrently via `asyncio.gather`
