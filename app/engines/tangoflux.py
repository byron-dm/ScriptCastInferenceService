import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.config import settings
from app.engines.base import AudioCueProducer, BaseInferenceEngine
from app.utils.io_utils import prepare_output_path
from app.utils.seeding import isolated_seed


class TangoFluxEngine(BaseInferenceEngine, AudioCueProducer):
    """TangoFlux engine for fast sound effect generation.

    TangoFlux is optimized for short SFX (max 30s) and generates 44.1kHz
    stereo audio very quickly. It runs in-process via its Python API.
    """

    MAX_DURATION_SECONDS = 30.0
    SAMPLE_RATE = 44100

    def __init__(self, model_id: str = "declare-lab/TangoFlux") -> None:
        self.model_id = model_id
        self._model = None
        self._device: str = settings.DEVICE
        # TangoFlux uses a single model instance; serialize generation.
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            try:
                from tangoflux import TangoFluxInference
            except ImportError as exc:
                raise ImportError(
                    f"Failed to import TangoFlux. Ensure it is installed: "
                    f"pip install git+https://github.com/declare-lab/TangoFlux --no-deps. "
                    f"Original error: {exc}"
                ) from exc

            # TangoFlux handles device placement internally via accelerate,
            # but we can hint it by setting the default device if needed.
            self._model = TangoFluxInference(name=self.model_id)

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    def generate_cue(
            self,
            prompt: str,
            duration_seconds: float,
            output_file_path: str,
            steps: int = 50,
            cfg_scale: float = 4.0,  # TangoFlux uses CFG scale, default ~4
            sampler_type: str = "pingpong",  # Not used by TangoFlux, kept for compatibility
            seed: int | None = None,
    ) -> str:
        # 1) Validate before loading the model.
        if not prompt or not prompt.strip():
            raise ValueError("prompt cannot be empty")

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")

        if duration_seconds > self.MAX_DURATION_SECONDS:
            raise ValueError(
                f"TangoFlux supports a maximum of {self.MAX_DURATION_SECONDS}s. "
                f"Received {duration_seconds}s."
            )

        dest_path = prepare_output_path(output_file_path)

        # 2) Serialize generation.
        with self._infer_lock:
            if not self.is_loaded:
                self.load_model()

            if self._model is None:
                raise RuntimeError("Model not loaded after load_model()")

            # 3) Generate. TangoFlux returns a torch tensor (channels, samples).
            # Note: TangoFlux recommends 50 steps for quality, 25 for speed.
            # We map cfg_scale directly; TangoFlux docs suggest 3.5-4.5.
            with isolated_seed(
                    None if seed is None or seed == -1 else seed,
                    device=self._device,
            ):
                audio = self._model.generate(
                    prompt,
                    steps=steps,
                    duration=duration_seconds,
                )

            # 4) Normalize and write.
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.asarray(audio, dtype=np.float32)

            # Ensure shape is (samples, channels) for soundfile.
            if audio_np.ndim == 2 and audio_np.shape[0] == 2:
                audio_np = audio_np.T  # transpose from (channels, samples) to (samples, channels)

            # Normalize peaks to avoid clipping.
            max_val = float(np.max(np.abs(audio_np)))
            if max_val > 0:
                audio_np = audio_np / max_val
            audio_np = np.clip(audio_np, -1.0, 1.0)

            self._atomic_write(dest_path, audio_np, samplerate=self.SAMPLE_RATE)

        return str(dest_path)
