import os
import random
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from app.config import settings
from app.engines.base import (
    BaseInferenceEngine,
    VoiceDesignProducer,
    VoiceProducer,
)

SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})


class BreezeEngine(BaseInferenceEngine, VoiceProducer, VoiceDesignProducer):
    """Breeze TTS 2 engine. Invokes infer.py as a subprocess."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or str(settings.BREEZE_MODEL_PATH)
        self._device: str = settings.DEVICE
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            code_root = settings.BREEZE_CODE_PATH
            infer_script = code_root / "infer.py"
            if not infer_script.is_file():
                raise RuntimeError(
                    f"infer.py not found at {infer_script}. Clone the repo with: "
                    f"git clone https://github.com/breezeblue-ai/breeze-tts.git {code_root}"
                )

            model_path = Path(self._model_path)
            if not model_path.is_dir() or not any(model_path.iterdir()):
                raise RuntimeError(
                    f"Breeze checkpoint not found at {model_path}. Download it with: "
                    f"huggingface-cli download BreezeBlue/Breeze-TTS-2 --local-dir {model_path}"
                )

            self._loaded = True

    def unload_model(self) -> None:
        with self._lock:
            self._loaded = False

    def _prepare_output_path(self, output_file_path: str) -> Path:
        if not output_file_path:
            raise ValueError("output_file_path cannot be empty")

        dest_path = Path(output_file_path)
        if not dest_path.is_absolute():
            raise ValueError("output_file_path must be an absolute path")
        if dest_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension: '{dest_path.suffix}'. "
                f"Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        parent = dest_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Cannot create directory {parent}: {exc}") from exc
        if not os.access(parent, os.W_OK):
            raise ValueError(f"No write permission for {parent}")

        return dest_path

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        # infer.py passes the seed straight to np.random.seed(), which rejects -1.
        # Translate "random" (-1 or None) into a valid 32-bit seed here.
        if seed is None or seed == -1:
            return random.randint(0, 2**32 - 1)
        return seed

    def _run_infer(self, cmd: list[str]) -> None:
        code_root = settings.BREEZE_CODE_PATH
        result = subprocess.run(
            cmd,
            cwd=str(code_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Breeze inference failed.\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    def generate_voice(
        self,
        text: str,
        output_file_path: str,
        reference_audio_path: str | None = None,
        reference_text: str | None = None,
        instruction: str | None = None,
        cfg_scale: float = 4.0,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        if not reference_audio_path:
            raise ValueError("reference_audio_path is required for Breeze TTS 2")
        if not Path(reference_audio_path).is_file():
            raise ValueError(f"Reference audio not found: {reference_audio_path}")

        if not reference_text or not reference_text.strip():
            raise ValueError(
                "reference_text is required and must match the exact spoken "
                "content of the reference audio."
            )

        dest_path = self._prepare_output_path(output_file_path)

        if not self.is_loaded:
            self.load_model()

        cmd = [
            sys.executable,
            str(settings.BREEZE_CODE_PATH / "infer.py"),
            str(self._model_path),
            "--text", text,
            "--ref-audio", reference_audio_path,
            "--ref-text", reference_text,
            "--output", str(dest_path),
            "--cfg-scale", str(cfg_scale),
            "--seed", str(self._resolve_seed(seed)),
        ]
        if instruction:
            cmd += ["--instruction", instruction]

        self._run_infer(cmd)

        if not dest_path.is_file():
            raise RuntimeError(
                f"infer.py exited successfully but no file was written to {dest_path}"
            )

        return str(dest_path)

    def generate_voice_design(
        self,
        text: str,
        instruction: str,
        output_file_path: str,
        cfg_scale: float = 4.0,
        seed: int | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        if not instruction or not instruction.strip():
            raise ValueError(
                "instruction is required for voice design and must describe "
                "the target speaker and delivery style."
            )

        dest_path = self._prepare_output_path(output_file_path)

        if not self.is_loaded:
            self.load_model()

        cmd = [
            sys.executable,
            str(settings.BREEZE_CODE_PATH / "infer.py"),
            str(self._model_path),
            "--text", text,
            "--instruction", instruction,
            "--output", str(dest_path),
            "--cfg-scale", str(cfg_scale),
            "--seed", str(self._resolve_seed(seed)),
        ]

        self._run_infer(cmd)

        if not dest_path.is_file():
            raise RuntimeError(
                f"infer.py exited successfully but no file was written to {dest_path}"
            )

        return str(dest_path)