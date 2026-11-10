import os
import random
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.config import settings
from app.engines.base import BaseInferenceEngine, VoiceProducer

SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})

# OmniVoice language names as expected by model.generate().
SUPPORTED_LANGUAGES = frozenset(
    {
        "english", "spanish", "chinese", "japanese", "korean",
        "german", "french", "russian", "portuguese", "italian",
        "arabic", "hindi", "dutch", "polish", "turkish",
    }
)


class OmniVoiceEngine(BaseInferenceEngine, VoiceProducer):
    """OmniVoice voice cloning engine (zero-shot, no style instruction)."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or str(settings.OMNIVOICE_MODEL_PATH)
        self._model = None
        self._sample_rate: int = 24000
        self._device: str = settings.DEVICE
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            model_path = Path(self._model_path)
            if not model_path.is_dir() or not any(model_path.iterdir()):
                raise RuntimeError(
                    f"OmniVoice checkpoint not found at {model_path}. "
                    f"Download it with: huggingface-cli download "
                    f"k2-fsa/OmniVoice --local-dir {model_path}"
                )

            try:
                from omnivoice import OmniVoice
            except ImportError as exc:
                raise ImportError(
                    f"Failed to import OmniVoice. Ensure it is installed "
                    f"in the active environment: {exc}"
                ) from exc

            self._model = OmniVoice.from_pretrained(
                str(model_path),
                device_map=self._device,
                dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    def _atomic_write(dest_path: Path, data: np.ndarray, samplerate: int) -> None:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(dest_path.parent),
            prefix=f".{dest_path.stem}.",
            suffix=dest_path.suffix,
        )
        os.close(fd)
        try:
            sf.write(
                file=tmp_name,
                data=data,
                samplerate=samplerate,
                format="WAV" if dest_path.suffix.lower() == ".wav" else "FLAC",
                subtype="PCM_16",
            )
            os.replace(tmp_name, dest_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def generate_voice(
            self,
            text: str,
            output_file_path: str,
            reference_audio_path: str | None = None,
            reference_text: str | None = None,
            instruction: str | None = None,
            cfg_scale: float = 4.0,
            speed: float = 1.0,
            language: str | None = None,
            seed: int | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        if not reference_audio_path:
            raise ValueError("reference_audio_path is required for OmniVoice")
        if not Path(reference_audio_path).is_file():
            raise ValueError(f"Reference audio not found: {reference_audio_path}")

        if language is not None and language.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )

        # `instruction` and `cfg_scale` are accepted but ignored.
        # OmniVoice has no style-instruction mode; use Breeze for that.
        dest_path = self._prepare_output_path(output_file_path)

        if not self.is_loaded:
            self.load_model()

        if self._model is None:
            raise RuntimeError("Model not loaded after load_model()")

        if seed is not None and seed != -1:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            if self._device == "cuda":
                torch.cuda.manual_seed_all(seed)

        generate_kwargs = {
            "text": text.strip(),
            "ref_audio": reference_audio_path,
        }
        if reference_text and reference_text.strip():
            generate_kwargs["ref_text"] = reference_text.strip()
        if language:
            generate_kwargs["language"] = language.lower()

        audio = self._model.generate(**generate_kwargs)

        # OmniVoice returns a list of np.ndarray chunks.
        if isinstance(audio, list):
            audio_np = np.concatenate(
                [np.asarray(a, dtype=np.float32).squeeze() for a in audio]
            )
        elif isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy().squeeze()
        else:
            audio_np = np.asarray(audio, dtype=np.float32).squeeze()

        if audio_np.size == 0:
            raise RuntimeError("OmniVoice produced no audio")

        max_val = float(np.max(np.abs(audio_np)))
        if max_val > 0:
            audio_np = audio_np / max_val
        audio_np = np.clip(audio_np, -1.0, 1.0)

        self._atomic_write(dest_path, audio_np, samplerate=self._sample_rate)

        return str(dest_path)
