import logging
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from huggingface_hub import snapshot_download
import numpy as np
import soundfile as sf
import torch

from app.config import settings
from app.engines.base import BaseInferenceEngine, VoiceProducer

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("modelscope").setLevel(logging.WARNING)

SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})


class CosyVoiceEngine(BaseInferenceEngine, VoiceProducer):
    """CosyVoice3 voice cloning engine (zero-shot, no style instruction)."""

    def __init__(self, repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512") -> None:
        self.repo_id: str = repo_id
        self._model = None
        self._sample_rate: int = 24000
        self._device: str = settings.DEVICE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_repo_present(self) -> Path:
        cosy_root = settings.MODELS_DIR / "CosyVoice"

        if not (cosy_root / "cosyvoice").is_dir():
            settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git", "clone", "--recursive",
                    "https://github.com/FunAudioLLM/CosyVoice.git",
                    str(cosy_root),
                ],
                check=True,
            )
        else:
            matcha_dir = cosy_root / "third_party" / "Matcha-TTS"
            if not matcha_dir.exists() or not any(matcha_dir.iterdir()):
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(cosy_root),
                    check=True,
                )

        return cosy_root

    def _register_sys_paths(self, cosy_root: Path) -> None:
        cosy_root_str = str(cosy_root)
        matcha_str = str(cosy_root / "third_party" / "Matcha-TTS")

        if cosy_root_str not in sys.path:
            sys.path.insert(0, cosy_root_str)
        if matcha_str not in sys.path:
            sys.path.insert(0, matcha_str)

    def _ensure_normalizer_assets(self) -> None:
        ttsfrd_dir = settings.MODELS_DIR / "CosyVoice-ttsfrd"
        zip_path = ttsfrd_dir / "resource.zip"
        unpacked_marker = ttsfrd_dir / "resource"

        if not unpacked_marker.exists():
            ttsfrd_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id="FunAudioLLM/CosyVoice-ttsfrd",
                local_dir=str(ttsfrd_dir),
                token=settings.HF_TOKEN,
            )
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(str(ttsfrd_dir))

        os.environ["MODELSCOPE_CACHE"] = str(settings.MODELS_DIR)

    def load_model(self) -> None:
        if self.is_loaded:
            return

        cosy_root = self._ensure_repo_present()
        self._register_sys_paths(cosy_root)
        self._ensure_normalizer_assets()

        model_target_dir = settings.MODELS_DIR / "CosyVoice3-0.5B"

        if not model_target_dir.exists() or not any(model_target_dir.iterdir()):
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(model_target_dir),
                token=settings.HF_TOKEN,
            )

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise ImportError(
                f"Failed to import AutoModel from {cosy_root}: {exc}"
            ) from exc

        self._model = AutoModel(model_dir=str(model_target_dir))

        core = getattr(self._model, "model", self._model)
        if hasattr(core, "llm") and isinstance(core.llm, torch.nn.Module):
            core.llm.to(dtype=torch.bfloat16)

        sample_rate = getattr(self._model, "sample_rate", None)
        if sample_rate is not None:
            self._sample_rate = sample_rate

    def unload_model(self) -> None:
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
            raise ValueError(
                "reference_audio_path is required for CosyVoice cloning"
            )
        if not Path(reference_audio_path).is_file():
            raise ValueError(f"Reference audio not found: {reference_audio_path}")

        # `instruction` and `reference_text` are accepted but ignored.
        # CosyVoice3's instruct2 mode does not reliably support free-form
        # instructions, so we use zero-shot cloning instead.
        dest_path = self._prepare_output_path(output_file_path)

        if not self.is_loaded:
            self.load_model()

        if self._model is None:
            raise RuntimeError("Model not loaded after load_model()")

        if seed is not None and seed != -1:
            torch.manual_seed(seed)
            if self._device == "cuda":
                torch.cuda.manual_seed_all(seed)

        generator = self._model.inference_zero_shot(
            tts_text=text.strip(),
            prompt_text="",
            prompt_wav=reference_audio_path,
            stream=False,
        )

        audio_segments: list[torch.Tensor] = []
        for chunk in generator:
            if "tts_speech" in chunk and chunk["tts_speech"] is not None:
                audio_segments.append(chunk["tts_speech"].cpu())

        if not audio_segments:
            raise RuntimeError("CosyVoice did not produce any audio segments")

        combined_tensor = torch.cat(audio_segments, dim=-1).to(torch.float32)

        max_val = torch.max(torch.abs(combined_tensor))
        if max_val > 0:
            combined_tensor = combined_tensor.div(max_val)
        combined_tensor = combined_tensor.clamp(-1.0, 1.0)

        audio_np: np.ndarray = combined_tensor.squeeze().numpy()

        output_format = "FLAC" if dest_path.suffix.lower() == ".flac" else "WAV"
        sf.write(
            file=str(dest_path),
            data=audio_np,
            samplerate=self._sample_rate,
            format=output_format,
            subtype="PCM_16",
        )

        return str(dest_path)