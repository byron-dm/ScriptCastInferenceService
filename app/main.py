from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.engines.base import AudioCueProducer, DialogueProducer, VoiceProducer
from app.engines.manager import engine_manager

app = FastAPI(
    title="ScriptCast Inference Service",
    version="1.2.0",
    description="Multi-engine audio synthesis microservice (Audio, Dialogue, Voice)",
)


# --- DTOs ---


class RenderAudioRequest(BaseModel):
    engine_name: str = Field(
        default="stable-audio-3", description="Target sound/cue engine"
    )
    prompt: str = Field(..., description="Prompt describing sound or music")
    duration_seconds: float = Field(
        ..., gt=0.0, description="Duration in seconds"
    )
    output_file_path: str = Field(
        ..., description="Target file path (.flac or .wav)"
    )
    seed: int | None = Field(default=None, description="Signed 32-bit seed")
    steps: int = Field(default=8, ge=1, le=250, description="Inference steps")
    cfg_scale: float = Field(default=1.0, ge=1.0, le=15.0)
    sampler_type: str = Field(default="pingpong")


class RenderDialogueRequest(BaseModel):
    """Schema for dialogue generation requests across engines."""

    engine_name: str
    script: str
    output_file_path: str
    prompt_speech_path: str | None = None
    speaker_id: str | None = None
    instruct_text: str | None = None
    seed: int | None = None

    class Config:
        populate_by_name = True
        extra = "ignore"


class RenderVoiceRequest(BaseModel):
    engine_name: str = Field(
        default="breeze", description="Target voice synthesis engine"
    )
    text: str = Field(..., description="Speech content to synthesize")
    output_file_path: str = Field(
        ..., description="Target file path (.flac or .wav)"
    )
    reference_audio_path: str | None = Field(
        default=None, description="Reference audio for zero-shot cloning"
    )
    speed: float = Field(default=1.0, gt=0.2, le=3.0, description="Speech rate")
    seed: int | None = Field(default=None)


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


@app.post("/render_dialogue", response_model=RenderResponse)
def render_dialogue(request: RenderDialogueRequest):
    try:
        engine = engine_manager.get_engine_by_type(
            request.engine_name, DialogueProducer
        )

        reference_path = request.prompt_speech_path or request.speaker_id

        output_path = engine.generate_dialogue(
            script=request.script,
            output_file_path=request.output_file_path,
            speaker_id=reference_path,
            instruct_text=request.instruct_text,
            seed=request.seed,
        )
        return RenderResponse(status="success", file_path=output_path)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/render_voice", response_model=RenderResponse)
def render_voice(request: RenderVoiceRequest):
    try:
        engine = engine_manager.get_engine_by_type(
            request.engine_name, VoiceProducer
        )
        saved_path = engine.generate_voice(
            text=request.text,
            output_file_path=request.output_file_path,
            reference_audio_path=request.reference_audio_path,
            speed=request.speed,
            seed=request.seed,
        )
        return RenderResponse(status="success", file_path=saved_path)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))