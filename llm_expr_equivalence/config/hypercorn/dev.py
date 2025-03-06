from llm_expr_equivalence.config.settings import get_settings

settings = get_settings()

__all__ = ["bind", "accesslog", "errorlog", "use_reloader", "debug", "loglevel"]

bind = [settings.bind_to]
accesslog = "-"
errorlog = "-"

use_reloader = True if settings.environment == "local" else False
debug = True if settings.environment == "local" else False
loglevel = "info" if settings.environment == "local" else "error"
