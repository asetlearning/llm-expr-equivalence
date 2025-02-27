import logging
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["get_settings", "logger"]


logger = logging.getLogger("GRD")
logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    """
    Settings for the application.
    """

    environment: str = "local"
    bind_to: str = "[::]:8000"
    mathserver_url: str  = ""
    parser_url: str = ""

    openai_api_key: SecretStr = ""

    verbose: bool = False
    agent_verbose: bool = False

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[call-arg]
logger.info(settings.model_dump_json(indent=2))


@lru_cache()
def get_settings():
    return settings
