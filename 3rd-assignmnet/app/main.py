from fastapi import FastAPI

from .logger import logger
from .routers.agent_router import router as agent_router
from .routers.pipeline_router import router as pipeline_router

app = FastAPI(
    title="ClassicModels Text-to-SQL Agent",
    description=(
        "Task 3 (POST /text2sql): linear Text-to-SQL pipeline with one retry. "
        "Task 4 (POST /agent/sql): agentic SQL with 3-retry self-correction and "
        "natural-language summary."
    ),
    version="1.0.0",
)

app.include_router(pipeline_router)
app.include_router(agent_router)


@app.get("/", tags=["root"])
def root():
    logger.info("GET /")
    return {
        "message": "ClassicModels Text-to-SQL Agent",
        "docs_url": "/docs",
        "endpoints": ["/text2sql", "/agent/sql"],
    }


@app.get("/health", tags=["root"])
def health():
    return {"status": "ok"}
