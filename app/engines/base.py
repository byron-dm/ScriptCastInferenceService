from abc import ABC, abstractmethod


class BaseInferenceEngine(ABC):
    """Base lifecycle contract for all inference engines."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Indicates whether weights are in memory."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Initializes model weights onto the target compute device."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Frees device VRAM and garbage-collects model references."""
        pass


class AudioCueProducer(ABC):
    """Contract for text-to-audio/sound-design generation engines."""

    @abstractmethod
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
        pass


class DialogueProducer(ABC):
    """Contract for multirole or conversational dialogue generation engines."""

    @abstractmethod
    def generate_dialogue(
        self,
        script: str,
        output_file_path: str,
        speaker_id: str | None = None,
        instruct_text: str | None = None,
        seed: int | None = None,
    ) -> str:
        pass


class VoiceProducer(ABC):
    """Contract for single-speaker TTS and voice-cloning synthesis engines."""

    @abstractmethod
    def generate_voice(
        self,
        text: str,
        output_file_path: str,
        reference_audio_path: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> str:
        pass