"""Task 4 endpoint: full agentic SQL with 3-retry self-correction + summary."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..agent import run
from ..database import engine

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=3)


@router.post("/sql")
def agent_sql(req: AgentRequest) -> dict:
    return run(req.question.strip(), engine)
