from typing import Dict, Type, TypeVar, List

from app.engines.base import (
    BaseInferenceEngine,
)
from app.engines.breeze import BreezeEngine
from app.engines.omnivoice import OmniVoiceEngine
from app.engines.stable_audio import StableAudioEngine

T = TypeVar("T")


class EngineManager:
    """Manages lifecycle and routing for registered inference engines."""

    def __init__(self) -> None:
        self._engines: Dict[str, BaseInferenceEngine] = {
            "stable-audio-3": StableAudioEngine(),
            "breeze": BreezeEngine(),
            "omnivoice": OmniVoiceEngine(),
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

    def get_engines_by_interface(self, expected_type: Type[T]) -> List[str]:
        """Return a list of engine names that implement the given interface."""
        return [
            name for name, engine in self._engines.items()
            if isinstance(engine, expected_type)
        ]

    def get_status(self) -> dict:
        return {
            name: engine.is_loaded for name, engine in self._engines.items()
        }

    def unload_engine(self, engine_name: str) -> None:
        """Unload a specific engine to free VRAM."""
        if engine_name not in self._engines:
            available = ", ".join(self._engines.keys())
            raise ValueError(
                f"Engine '{engine_name}' not registered. Available: {available}"
            )
        self._engines[engine_name].unload_model()


engine_manager = EngineManager()
