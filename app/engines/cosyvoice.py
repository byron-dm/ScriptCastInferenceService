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
from app.engines.base import BaseInferenceEngine, DialogueProducer

# Suppress network polling noise from external model hubs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("modelscope").setLevel(logging.WARNING)


class CosyVoiceEngine(BaseInferenceEngine, DialogueProducer):
    """Inference engine implementing CosyVoice3 inside the configured cache directory."""

    def __init__(self, repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512") -> None:
        self.repo_id: str = repo_id
        self._model = None
        self._sample_rate: int = 24000
        self._device: str = settings.DEVICE if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        """Indicates whether weights are in memory."""
        return self._model is not None

    def _ensure_repo_present(self) -> Path:
        """Clones or updates CosyVoice and its submodules inside settings.MODELS_DIR."""
        cosy_root = settings.MODELS_DIR / "CosyVoice"

        if not (cosy_root / "cosyvoice").is_dir():
            settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--recursive",
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
        """Injects CosyVoice root and Matcha-TTS into runtime sys.path."""
        cosy_root_str = str(cosy_root)
        matcha_str = str(cosy_root / "third_party" / "Matcha-TTS")

        if cosy_root_str not in sys.path:
            sys.path.insert(0, cosy_root_str)
        if matcha_str not in sys.path:
            sys.path.insert(0, matcha_str)

    def _ensure_normalizer_assets(self) -> None:
        """Downloads CosyVoice text normalizer resources from HF and unpacks them."""
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
        """Prepares assets and loads native model weights onto the target compute device."""
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

        # Instantiate native CosyVoice3 wrapper
        self._model = AutoModel(model_dir=str(model_target_dir))

        # Explicitly harmonize LLM submodules to BFloat16 to prevent matrix dtype mismatch
        core = getattr(self._model, "model", self._model)
        if hasattr(core, "llm") and isinstance(core.llm, torch.nn.Module):
            core.llm.to(dtype=torch.bfloat16)

        sample_rate = getattr(self._model, "sample_rate", None)
        if sample_rate is not None:
            self._sample_rate = sample_rate

    def unload_model(self) -> None:
        """Frees device VRAM and garbage-collects model references."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_dialogue(
        self,
        script: str,
        output_file_path: str,
        speaker_id: str | None = None,
        instruct_text: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Runs dialogue inference via cross-lingual synthesis and saves audio to disk."""
        if not self.is_loaded:
            self.load_model()

        assert self._model is not None

        if seed is not None:
            torch.manual_seed(seed)
            if self._device == "cuda":
                torch.cuda.manual_seed_all(seed)

        reference_audio = (
            speaker_id
            or str(settings.MODELS_DIR / "CosyVoice" / "asset" / "zero_shot_prompt.wav")
        )

        clean_text = script.strip()
        formatted_prompt = f"You are a helpful assistant.<|endofprompt|><|en|>{clean_text}"

        generator = self._model.inference_cross_lingual(
            tts_text=formatted_prompt,
            prompt_wav=reference_audio,
            stream=False,
        )

        audio_segments: list[torch.Tensor] = []
        for chunk in generator:
            if "tts_speech" in chunk and chunk["tts_speech"] is not None:
                audio_segments.append(chunk["tts_speech"].cpu())

        if not audio_segments:
            raise RuntimeError("CosyVoice did not produce any audio segments")

        combined_tensor = torch.cat(audio_segments, dim=-1).to(torch.float32)

        # Normalize audio peak safely without shape transpositions
        max_val = torch.max(torch.abs(combined_tensor))
        if max_val > 0:
            combined_tensor = combined_tensor.div(max_val)
        combined_tensor = combined_tensor.clamp(-1.0, 1.0)

        dest_path = Path(output_file_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        audio_np: np.ndarray = combined_tensor.squeeze().numpy()
        ext = dest_path.suffix.lower()
        output_format = "FLAC" if ext == ".flac" else "WAV"

        sf.write(
            file=str(dest_path),
            data=audio_np,
            samplerate=self._sample_rate,
            format=output_format,
            subtype="PCM_16",
        )

        return str(dest_path)