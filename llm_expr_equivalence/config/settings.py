import logging
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["get_settings", "logger"]


# Configure logging format with time, place (file:line:function), and action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s:%(levelname)s [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Suppress httpx info logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("llm_equiv")


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    environment: str = "local"
    bind_to: str = "[::]:8000"
    mathserver_url: str  = ""
    parser_url: str = ""

    openai_api_key: SecretStr = SecretStr("")
    openai_api_api_base_url: str = ""
    nebius_api_key: SecretStr = SecretStr("")

    # Google Cloud settings
    gemini_api_key: SecretStr = SecretStr("")
    google_cloud_service_storage_creds_json: str = "llm-equivalence-storage-creds.json"
    google_cloud_project: str = "llm-equivalence"
    google_cloud_location: str = "us-central1"
    google_vertex_client_secret_json: str = "llm-equivalence-client-secret.json"
    google_vertex_access_token_json: str = "llm-equivalence-ai-token.json"
    google_cloud_input_bucket: str = "llm-equivalence-batch"
    google_cloud_output_bucket: str = "llm-equivalence-results"

    verbose: bool = False
    agent_verbose: bool = False

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[call-arg]
logger.info(settings.model_dump_json(indent=2))


@lru_cache()
def get_settings() -> Settings:
    return settings
