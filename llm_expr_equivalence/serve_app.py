import sys

from dotenv import load_dotenv
from fastapi import FastAPI

from llm_expr_equivalence.routes import health_router, openai_router

if not load_dotenv():
    print("Failed to load .env file.", flush=True, file=sys.stderr)

app = FastAPI()

app.include_router(health_router)
app.include_router(openai_router)
