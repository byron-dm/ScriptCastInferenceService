import argparse
import os
from pathlib import Path
import sys


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ScriptCast Python Inference Service"
    )
    default_base = Path(__file__).resolve().parent.parent
    default_cache = default_base / "models"

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("SCRIPTCAST_HOST", "127.0.0.1"),
        help="Host address to bind the service",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SCRIPTCAST_PORT", "8000")),
        help="Port to bind the service",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("SCRIPTCAST_MODELS_CACHE", str(default_cache)),
        help="Base directory where model weights are stored",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face personal access token",
    )

    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


class Settings:

    def __init__(self) -> None:
        args = parse_arguments()
        self.HOST: str = args.host
        self.PORT: int = args.port

        clean_cache_dir = args.cache_dir.strip('\'"').rstrip("\\/")
        self.MODELS_DIR: Path = Path(clean_cache_dir).resolve()
        self.HF_TOKEN: str | None = args.hf_token

        self.DEVICE: str = "cuda"
        self.STABLE_AUDIO_MODEL_ID: str = "stabilityai/stable-audio-3-medium"

        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(self.MODELS_DIR)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.MODELS_DIR)
        if self.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.HF_TOKEN


settings = Settings()