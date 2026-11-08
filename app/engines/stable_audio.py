import os
import tempfile
import threading
from pathlib import Path

from einops import rearrange
import numpy as np
import soundfile as sf
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import (
    generate_diffusion_cond_inpaint,
)
import torch

from app.config import settings
from app.engines.base import AudioCueProducer, BaseInferenceEngine

# Samplers soportados por stable-audio-tools (no los de ComfyUI).
SUPPORTED_SAMPLERS = frozenset(
    {"pingpong", "euler", "heun", "dpmpp-2m-sde", "dpmpp-3m-sde", "rk4"}
)
SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})


class StableAudioEngine(BaseInferenceEngine, AudioCueProducer):
    """Inference engine implementing Stable Audio 3 for cue/soundtrack generation."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id: str = model_id or settings.STABLE_AUDIO_MODEL_ID
        self._model = None
        self._model_config: dict | None = None
        self._device: str = settings.DEVICE
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._model_config is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            model, model_config = get_pretrained_model(self.model_id)
            model = model.to(self._device)
            if self._device == "cuda":
                model = model.to(torch.float16)

            self._model = model
            self._model_config = model_config

    def unload_model(self) -> None:
        with self._lock:
            self._model = None
            self._model_config = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _prepare_output_path(self, output_file_path: str) -> Path:
        """Valida y prepara la ruta de salida antes de gastar VRAM."""
        if not output_file_path:
            raise ValueError("output_file_path no puede estar vacío")

        dest_path = Path(output_file_path)

        if not dest_path.is_absolute():
            raise ValueError("output_file_path debe ser una ruta absoluta")

        if dest_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extensión no soportada: '{dest_path.suffix}'. "
                f"Usa: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        parent = dest_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"No se puede crear el directorio {parent}: {exc}"
            ) from exc

        if not os.access(parent, os.W_OK):
            raise ValueError(f"Sin permisos de escritura en {parent}")

        if dest_path.exists() and not os.access(dest_path, os.W_OK):
            raise ValueError(
                f"El archivo existente no es escribible: {dest_path}"
            )

        return dest_path

    @staticmethod
    def _atomic_write(
        dest_path: Path,
        data: np.ndarray,
        samplerate: int,
        output_format: str,
    ) -> None:
        """Escribe a un temporal y hace os.replace (atómico en mismo FS)."""
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
                format=output_format,
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
        steps: int = 8,
        cfg_scale: float = 1.0,
        sampler_type: str = "pingpong",
        seed: int | None = None,
    ) -> str:
        # 1) Validaciones antes de gastar VRAM
        if not prompt or not prompt.strip():
            raise ValueError("prompt no puede estar vacío")

        if sampler_type not in SUPPORTED_SAMPLERS:
            raise ValueError(
                f"sampler_type '{sampler_type}' no soportado. "
                f"Usa: {', '.join(sorted(SUPPORTED_SAMPLERS))}"
            )

        dest_path = self._prepare_output_path(output_file_path)

        # 2) Cargar modelo si hace falta
        if not self.is_loaded:
            self.load_model()

        if self._model is None or self._model_config is None:
            raise RuntimeError("Modelo no cargado tras load_model()")

        # 3) Validar duración contra el máximo del modelo
        sample_rate = self._model_config["sample_rate"]
        max_sample_size = self._model_config["sample_size"]
        max_seconds = max_sample_size / sample_rate

        if duration_seconds <= 0:
            raise ValueError("duration_seconds debe ser > 0")

        if duration_seconds > max_seconds:
            raise ValueError(
                f"duration_seconds={duration_seconds} excede el máximo "
                f"del modelo ({max_seconds:.2f}s)"
            )

        requested_samples = int(duration_seconds * sample_rate)
        sample_size = min(requested_samples, max_sample_size)

        # 4) Generar
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": int(duration_seconds),
        }]

        if seed is None:
            seed = int(np.random.randint(0, 2147483647))

        torch.manual_seed(seed)
        if self._device == "cuda":
            torch.cuda.manual_seed_all(seed)

        output = generate_diffusion_cond_inpaint(
            model=self._model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type=sampler_type,
            device=self._device,
            seed=seed,
        )

        output = rearrange(output, "b d n -> d (b n)")
        output = output[:, :requested_samples]

        output = output.to(torch.float32)
        max_val = torch.max(torch.abs(output))
        if max_val > 0:
            output = output.div(max_val)
        output = output.clamp(-1.0, 1.0)

        audio_np: np.ndarray = output.cpu().numpy().T

        # 5) Escribir de forma atómica
        ext = dest_path.suffix.lower()
        output_format = "FLAC" if ext == ".flac" else "WAV"

        self._atomic_write(
            dest_path=dest_path,
            data=audio_np,
            samplerate=sample_rate,
            output_format=output_format,
        )

        # 6) Devolver exactamente la ruta que el cliente pidió
        return str(dest_path)