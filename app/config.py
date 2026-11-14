import argparse
import os
from pathlib import Path
import sys

import torch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Audio Inference Service"
    )
    default_base = Path(__file__).resolve().parent.parent
    default_cache = default_base / "models"

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("INFERENCE_HOST", "127.0.0.1"),
        help="Host address to bind the service",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("INFERENCE_PORT", "8000")),
        help="Port to bind the service",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("INFERENCE_MODELS_CACHE", str(default_cache)),
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


def _resolve_models_dir(raw_cache_dir: str) -> Path:
    """Resolve and validate the models cache directory.

    On Windows, ``Path.mkdir(parents=True)`` cannot create a drive root
    (e.g. ``E:\\``); if the drive is not mounted, ``mkdir`` fails with a
    cryptic ``FileNotFoundError``. We validate the path here and raise a
    clear ``RuntimeError`` instead.
    """
    clean_cache_dir = raw_cache_dir.strip('\'"').rstrip("\\/")
    if not clean_cache_dir:
        raise RuntimeError(
            "The --cache-dir value is empty. Pass a valid directory, "
            "e.g. --cache-dir C:\\scriptcast"
        )

    models_dir = Path(clean_cache_dir).resolve()

    # On Windows, the drive root must already exist; mkdir(parents=True)
    # cannot create it. Fail with context if it does not.
    if os.name == "nt":
        drive = models_dir.drive  # e.g. "E:" or "\\\\server\\share"
        if drive and not Path(drive + os.sep).exists():
            raise RuntimeError(
                f"Cannot use --cache-dir '{models_dir}': drive "
                f"'{drive}' is not available. Check that the drive is "
                f"mounted or pass a different --cache-dir."
            )

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create --cache-dir '{models_dir}': {exc}. "
            f"Check the path and permissions, or pass a different "
            f"--cache-dir."
        ) from exc

    return models_dir


class Settings:

    def __init__(self) -> None:
        args = parse_arguments()
        self.HOST: str = args.host
        self.PORT: int = args.port

        self.MODELS_DIR: Path = _resolve_models_dir(args.cache_dir)
        self.HF_TOKEN: str | None = args.hf_token

        self.DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.STABLE_AUDIO_MODEL_ID: str = "stabilityai/stable-audio-3-medium"
        self.TANGOFLUX_MODEL_ID: str = "declare-lab/TangoFlux"

        self.BREEZE_CODE_PATH: Path = self.MODELS_DIR / "breeze-tts"
        self.BREEZE_MODEL_PATH: Path = self.MODELS_DIR / "breeze-tts-2"

        self.OMNIVOICE_MODEL_PATH: Path = self.MODELS_DIR / "omnivoice"

        os.environ["HF_HOME"] = str(self.MODELS_DIR)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.MODELS_DIR)
        if self.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.HF_TOKEN


settings = Settings()