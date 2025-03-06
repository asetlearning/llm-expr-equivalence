from time import time
from typing import Annotated

from fastapi import APIRouter, Depends

from llm_expr_equivalence.config.settings import Settings, get_settings

__all__ = ["health_router"]

health_router = APIRouter(tags=["health"])


@health_router.get("/health/")
def health_check(settings: Annotated[Settings, Depends(get_settings)]):
    """
    Health check endpoint.
    """
    return {"status": "ok", "time": time(), "environment": settings.environment}
