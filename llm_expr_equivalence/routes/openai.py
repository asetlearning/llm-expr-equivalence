from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from llm_expr_equivalence.config.settings import Settings, get_settings
from llm_expr_equivalence.equivalence_llm.equivalence_openai import LLMEquivalenceOpenAI

__all__ = ["openai_router"]

openai_router = APIRouter(tags=["openai"])


class ExpressionComparisonRequest(BaseModel):
    expr1: str
    expr2: str


@openai_router.post("/openai/{model_name}")
def health_check(
    model_name: str,
    request: ExpressionComparisonRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """
    Health check endpoint.
    """

    if model_name not in [""]:
        raise HTTPException(status_code=400, detail="Invalid model name specified.")

    llm = LLMEquivalenceOpenAI(model_name)
    result = llm.equal(expr1=request.expr1, expr2=request.expr2)
    return {"result": str(result)}
