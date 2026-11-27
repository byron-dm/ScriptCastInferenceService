import os
import sys

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import uvicorn

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(asctime)s - %(levelprefix)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "use_colors": None,
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "app": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}


def _load_app():
    """Import the FastAPI app, surfacing startup errors cleanly.

    ``app.config`` instantiates ``Settings`` at import time, so an invalid
    ``--cache-dir`` raises ``RuntimeError`` during the import. We catch it
    here to print an actionable message instead of a raw traceback, while
    still allowing ``INFERENCE_DEBUG=1`` to surface the full traceback.
    """
    from app.config import settings
    from app.main import app
    return settings, app


def main() -> int:
    debug = os.getenv("INFERENCE_DEBUG", "").lower() in {"1", "true", "yes"}

    try:
        settings, app = _load_app()
    except Exception as exc:
        if debug:
            raise
        print(
            f"Failed to start inference service: {exc}",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
        log_config=LOG_CONFIG,
        timeout_keep_alive=900
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
