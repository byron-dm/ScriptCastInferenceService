from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.engines.base import AudioCueProducer, VoiceDesignProducer, VoiceProducer
from app.engines.manager import engine_manager

app = FastAPI(
    title="Audio Inference Service",
    version="1.5.0",
    description=(
        "Multi-engine audio synthesis microservice "
        "(Audio, Voice Cloning, Voice Design)"
    ),
)

MAX_AUDIO_DURATION_SECONDS = 285.0
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}
VOICE_CLONE_ENGINES = {"breeze", "omnivoice"}
VOICE_DESIGN_ENGINES = {"breeze"}


# --- DTOs ---


class RenderAudioRequest(BaseModel):
    engine_name: str = Field(
        default="stable-audio-3", description="Target sound/cue engine"
    )
    prompt: str = Field(
        ..., min_length=1, description="Prompt describing sound or music"
    )
    duration_seconds: float = Field(
        ...,
        gt=0.0,
        le=MAX_AUDIO_DURATION_SECONDS,
        description="Duration in seconds",
    )
    output_file_path: str = Field(
        ..., description="Target file path (.flac or .wav)"
    )
    seed: int | None = Field(default=None, description="Signed 32-bit seed")
    steps: int = Field(default=8, ge=1, le=250, description="Inference steps")
    cfg_scale: float = Field(default=1.0, ge=1.0, le=15.0)
    sampler_type: str = Field(default="pingpong")

    @field_validator("output_file_path")
    @classmethod
    def validate_output_path(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("output_file_path must be an absolute path")
        if p.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise ValueError("output_file_path must end with .wav or .flac")
        return v


class RenderVoiceCloneRequest(BaseModel):
    engine_name: str = Field(
        default="breeze",
        description="Voice cloning engine: breeze or omnivoice",
    )
    text: str = Field(..., min_length=1, description="Speech content to synthesize")
    output_file_path: str = Field(
        ..., description="Target file path (.flac or .wav)"
    )
    reference_audio_path: str = Field(
        ..., description="Absolute path to the reference audio to clone"
    )
    reference_text: str | None = Field(
        default=None,
        description=(
            "Exact transcript of the reference audio. "
            "Required for breeze; optional for omnivoice (Whisper transcribes)."
        ),
    )
    instruction: str | None = Field(
        default=None,
        description="Voice direction instruction (used by breeze; ignored by omnivoice)",
    )
    cfg_scale: float = Field(
        default=4.0, ge=0.1, le=10.0, description="CFG scale (used by breeze)"
    )
    speed: float = Field(
        default=1.0, gt=0.2, le=3.0,
        description="Speech rate (currently unused by both engines)",
    )
    language: str | None = Field(
        default=None,
        description=(
            "Language of the text to synthesize (e.g. 'spanish'). "
            "Used by omnivoice; ignored by breeze."
        ),
    )
    seed: int | None = Field(default=None, description="Seed; -1 or null = random")


class RenderVoiceDesignRequest(BaseModel):
    engine_name: str = Field(
        default="breeze",
        description="Voice design engine. Only 'breeze' is supported.",
    )
    text: str = Field(..., min_length=1, description="Speech content to synthesize")
    instruction: str = Field(
        ...,
        min_length=1,
        description="Description of the target voice and delivery style",
    )
    output_file_path: str = Field(
        ..., description="Target file path (.flac or .wav)"
    )
    cfg_scale: float = Field(
        default=4.0, ge=0.1, le=10.0, description="CFG scale for instruction adherence"
    )
    seed: int | None = Field(
        default=None,
        description="Seed; -1 or null = random. Different seeds yield different voices.",
    )


class RenderResponse(BaseModel):
    status: str
    file_path: str


# --- Endpoints ---


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "engines": engine_manager.get_status(),
        "device": settings.DEVICE,
        "cache_dir": str(settings.MODELS_DIR),
    }


@app.post("/render_audio", response_model=RenderResponse)
def render_audio(request: RenderAudioRequest):
    try:
        engine = engine_manager.get_engine_by_type(
            request.engine_name, AudioCueProducer
        )
        saved_path = engine.generate_cue(
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            output_file_path=request.output_file_path,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            sampler_type=request.sampler_type,
            seed=request.seed,
        )
        return RenderResponse(status="success", file_path=saved_path)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/render_voice_clone", response_model=RenderResponse)
def render_voice_clone(request: RenderVoiceCloneRequest):
    try:
        if request.engine_name not in VOICE_CLONE_ENGINES:
            raise ValueError(
                f"Unsupported voice cloning engine '{request.engine_name}'. "
                f"Available: {', '.join(sorted(VOICE_CLONE_ENGINES))}"
            )

        if request.engine_name == "breeze" and not request.reference_text:
            raise ValueError(
                "reference_text is required when engine_name is 'breeze'"
            )

        engine = engine_manager.get_engine_by_type(
            request.engine_name, VoiceProducer
        )
        saved_path = engine.generate_voice(
            text=request.text,
            output_file_path=request.output_file_path,
            reference_audio_path=request.reference_audio_path,
            reference_text=request.reference_text,
            instruction=request.instruction,
            cfg_scale=request.cfg_scale,
            speed=request.speed,
            language=request.language,
            seed=request.seed,
        )
        return RenderResponse(status="success", file_path=saved_path)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/render_voice_design", response_model=RenderResponse)
def render_voice_design(request: RenderVoiceDesignRequest):
    try:
        if request.engine_name not in VOICE_DESIGN_ENGINES:
            raise ValueError(
                f"Voice design is only supported by: "
                f"{', '.join(sorted(VOICE_DESIGN_ENGINES))}. "
                f"Received engine_name='{request.engine_name}'."
            )

        engine = engine_manager.get_engine_by_type(
            request.engine_name, VoiceDesignProducer
        )
        saved_path = engine.generate_voice_design(
            text=request.text,
            instruction=request.instruction,
            output_file_path=request.output_file_path,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
        )
        return RenderResponse(status="success", file_path=saved_path)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))