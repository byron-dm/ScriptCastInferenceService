from typing import Dict, Type, TypeVar

from app.engines.base import (
    AudioCueProducer,
    BaseInferenceEngine,
    VoiceProducer,
)
from app.engines.breeze import BreezeEngine
from app.engines.cosyvoice import CosyVoiceEngine
from app.engines.stable_audio import StableAudioEngine

T = TypeVar("T")


class EngineManager:
    """Manages lifecycle and routing for registered inference engines."""

    def __init__(self) -> None:
        self._engines: Dict[str, BaseInferenceEngine] = {
            "stable-audio-3": StableAudioEngine(),
            "breeze": BreezeEngine(),
            "cosyvoice": CosyVoiceEngine(),
        }

    def get_engine_by_type(self, engine_name: str, expected_type: Type[T]) -> T:
        if engine_name not in self._engines:
            available = ", ".join(self._engines.keys())
            raise ValueError(
                f"Engine '{engine_name}' not registered. Available: {available}"
            )

        engine = self._engines[engine_name]
        if not isinstance(engine, expected_type):
            raise TypeError(
                f"Engine '{engine_name}' does not implement {expected_type.__name__}"
            )

        return engine

    def get_status(self) -> dict:
        return {
            name: engine.is_loaded for name, engine in self._engines.items()
        }


engine_manager = EngineManager()