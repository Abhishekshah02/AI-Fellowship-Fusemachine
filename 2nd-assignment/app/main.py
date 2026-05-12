from fastapi import FastAPI

from .count_router import router as count_router
from .logger import logger
from .router import router as customer_router

app = FastAPI(
    title="Classic Models Customer API",
    description="Layered Customer API for the classicmodels Postgres database.",
    version="1.0.0",
)

# IMPORTANT: include count_router BEFORE customer_router
# so /customers/count matches before /customers/{customer_number}
app.include_router(count_router)
app.include_router(customer_router)


@app.get("/", tags=["root"])
def root():
    logger.info("GET /")
    return {
        "message": "Welcome to the Classic Models Customer API",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["root"])
def health():
    return {"status": "ok"}
