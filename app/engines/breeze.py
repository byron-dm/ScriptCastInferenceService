from pathlib import Path
from app.engines.base import BaseInferenceEngine, VoiceProducer


class BreezeEngine(BaseInferenceEngine, VoiceProducer):
    """Engine handling TTS and single-voice generation with Breeze."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self.is_loaded:
            return
        # Ready for HF model card integration
        pass

    def unload_model(self) -> None:
        self._model = None

    def generate_voice(
        self,
        text: str,
        output_file_path: str,
        reference_audio_path: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> str:
        if not self.is_loaded:
            self.load_model()

        dest_path = Path(output_file_path).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Placeholder: will be connected once the HF card pipeline is supplied
        return str(dest_path)