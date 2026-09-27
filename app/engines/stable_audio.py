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


class StableAudioEngine(BaseInferenceEngine, AudioCueProducer):
    """Inference engine implementing Stable Audio 3 for cue/soundtrack generation."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id: str = model_id or settings.STABLE_AUDIO_MODEL_ID
        self._model = None
        self._model_config: dict | None = None
        self._device: str = (
            settings.DEVICE if torch.cuda.is_available() else "cpu"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._model_config is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return

        model, model_config = get_pretrained_model(self.model_id)
        model = model.to(self._device)
        if self._device == "cuda":
            model = model.to(torch.float16)

        self._model = model
        self._model_config = model_config

    def unload_model(self) -> None:
        self._model = None
        self._model_config = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        if not self.is_loaded:
            self.load_model()

        assert self._model is not None and self._model_config is not None

        sample_rate = self._model_config["sample_rate"]
        max_sample_size = self._model_config["sample_size"]
        requested_samples = int(duration_seconds * sample_rate)
        sample_size = min(requested_samples, max_sample_size)

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
        dest_path = Path(output_file_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        ext = dest_path.suffix.lower()
        output_format = "FLAC" if ext == ".flac" else "WAV"

        sf.write(
            file=str(dest_path),
            data=audio_np,
            samplerate=sample_rate,
            format=output_format,
            subtype="PCM_16",
        )

        return str(dest_path)
