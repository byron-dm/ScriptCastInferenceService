import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import uvicorn

from app.config import settings
from app.main import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info",
    )