from dotenv import load_dotenv
from fastapi import FastAPI

from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.routes import health_router, openai_router

if not load_dotenv():
    logger.error("Failed to load .env file.")

app = FastAPI()

app.include_router(health_router)
app.include_router(openai_router)
