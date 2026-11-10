# Audio Inference Service

A local Python inference microservice that powers the audio pipeline of **[ScriptCast](https://thesiloforge.com/scriptcast/)**.

While this service was built for ScriptCast, it operates as a standalone, stateless HTTP microservice that routes requests to multiple inference engines.

## Engines

| Type | Contract | Engine | Model |
|------|----------|--------|-------|
| Cue / Soundtrack | AudioCueProducer | StableAudioEngine | stabilityai/stable-audio-3-medium |
| Voice Cloning | VoiceProducer | BreezeEngine | Breeze TTS 2 |
| Voice Cloning | VoiceProducer | OmniVoiceEngine | k2-fsa/OmniVoice |
| Voice Design | VoiceDesignProducer | BreezeEngine | Breeze TTS 2 |

## Requirements

- **Python 3.10+**
- **NVIDIA GPU** with CUDA 12.8 (tested on RTX 30/40 series)
- **PyTorch 2.8.0 + CUDA 12.8**
- **~10 GB of VRAM** to load models (Breeze ~7.7 GB, OmniVoice ~4.4 GB, Stable Audio ~6 GB)
- **~20 GB of disk** for checkpoints and Hugging Face cache

## Installation

### 1. Create and activate the virtual environment

    python -m venv .venv
    .venv\Scripts\activate

### 2. Install PyTorch 2.8.0 with CUDA 12.8

    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

Verify:

    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    # Should print: 2.8.0+cu128 True

### 3. Install the rest of the dependencies

    pip install -r requirements.txt

Note on stable-audio-tools: its pyproject.toml pins torch==2.7.1. Since we use 2.8.0 here, install it with --no-deps to prevent pip from overwriting your torch version:

    pip uninstall stable-audio-tools -y
    pip install stable-audio-tools --no-deps

### 4. (Optional) Flash Attention

To accelerate Stable Audio inference, install the flash_attn wheel matching your environment (torch 2.8.0 + CUDA 12.8 + cp310):

    pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp310-cp310-win_amd64.whl" --no-deps

Verify:

    python -c "from flash_attn import flash_attn_func; print('ok')"

If skipped, Stable Audio still works, just slower.

### 5. (Optional) SoX

torchaudio on Windows will warn if SoX is missing. It's not blocking, but you can install it:

    choco install sox
    # or: scoop install sox

## Model Checkpoints

All checkpoints live under MODELS_DIR, which defaults to models/ in the project root. You can override this with --cache-dir at launch.

### Stable Audio 3
Downloads automatically on the first call to /render_audio via get_pretrained_model. No manual action required.

### Breeze TTS 2
Requires manual download:

    # Inference code
    git clone https://github.com/breezeblue-ai/breeze-tts.git E:\scriptcast\breeze-tts

    # Checkpoint (~7 GB)
    huggingface-cli download BreezeBlue/Breeze-TTS-2 --local-dir E:\scriptcast\breeze-tts-2

    # Repo dependencies (without touching torch)
    pip install -r E:\scriptcast\breeze-tts\requirements.txt --no-deps

### OmniVoice
Requires manual download:

    huggingface-cli download k2-fsa/OmniVoice --local-dir E:\scriptcast\omnivoice

## Running

    python run.py --cache-dir E:\scriptcast --hf-token hf_XXXXXXXXXXXX

| Argument | Default | Description |
|---|---|---|
| --host | 127.0.0.1 | Bind address |
| --port | 8000 | Port |
| --cache-dir | ./models | Root directory for checkpoints |
| --hf-token | HF_TOKEN env | Hugging Face token |

The service is available at http://127.0.0.1:8000. Interactive documentation (Swagger UI) is at http://127.0.0.1:8000/docs.

## Endpoints

### GET /health

Service status and loaded engines.

    {
      "status": "healthy",
      "engines": {
        "stable-audio-3": false,
        "breeze": false,
        "omnivoice": false
      },
      "device": "cuda",
      "cache_dir": "E:\\scriptcast"
    }

false means the model is not loaded in memory yet (lazy-loaded on first request).

### POST /render_audio

Generates a cue, sound effect, or music fragment with Stable Audio 3.

Minimal payload:

    {
      "prompt": "rain on a tin roof, distant thunder",
      "duration_seconds": 10,
      "output_file_path": "E:/scriptcast/out/rain.flac"
    }

Full payload:

    {
      "engine_name": "stable-audio-3",
      "prompt": "rain on a tin roof, distant thunder",
      "duration_seconds": 10,
      "output_file_path": "E:/scriptcast/out/rain.flac",
      "seed": 42,
      "steps": 8,
      "cfg_scale": 1.0,
      "sampler_type": "pingpong"
    }

| Field | Required | Default | Notes |
|---|---|---|---|
| prompt | Yes | — | Audio description |
| duration_seconds | Yes | — | Max ~285 s |
| output_file_path | Yes | — | Absolute path .wav or .flac |
| seed | No | random | — |
| steps | No | 8 | 8 is fast mode (LCM) |
| cfg_scale | No | 1.0 | Lower = faster |
| sampler_type | No | pingpong | euler, heun, dpmpp-*-sde, rk4 |

### POST /render_voice_clone

Clones a voice from a reference audio. Available engines: breeze and omnivoice.

Payload for Breeze (supports voice direction via instruction):

    {
      "engine_name": "breeze",
      "text": "Certain houses, like certain persons, manage somehow to proclaim at once their character for evil.",
      "output_file_path": "E:/scriptcast/out/clone_breeze.wav",
      "reference_audio_path": "E:/scriptcast/character - narrator.flac",
      "reference_text": "And, perhaps, with houses the same principle is operative...",
      "instruction": "Slow and hushed, conviction building to a shiver",
      "cfg_scale": 4.0,
      "seed": 42
    }

Payload for OmniVoice (no voice direction, but accepts language):

    {
      "engine_name": "omnivoice",
      "text": "El silencio de la casa se rompió cuando la puerta crujió al abrirse.",
      "output_file_path": "E:/scriptcast/out/clone_omni.flac",
      "reference_audio_path": "E:/scriptcast/character - narrator.flac",
      "reference_text": "And, perhaps, with houses the same principle is operative...",
      "language": "spanish",
      "seed": 42
    }

| Field | Required | Breeze | OmniVoice |
|---|---|---|---|
| text | Yes | Yes | Yes |
| output_file_path | Yes | Yes | Yes |
| reference_audio_path | Yes | Yes | Yes |
| reference_text | Breeze only | Yes exact | Optional (Whisper transcribes) |
| instruction | No | Yes voice direction | Ignored |
| cfg_scale | No | Yes default 4.0 | Ignored |
| language | No | Ignored | Yes e.g. "spanish" |
| seed | No | -1 = random | -1 = random |

### POST /render_voice_design

Designs a new voice from scratch based on a description. Breeze only.

    {
      "engine_name": "breeze",
      "text": "The house had been waiting for someone like him.",
      "instruction": "A deep, gravelly male voice in his late fifties, slow and deliberate, with a faint British accent.",
      "output_file_path": "E:/scriptcast/out/design.flac",
      "cfg_scale": 4.0,
      "seed": 12345
    }

| Field | Required | Notes |
|---|---|---|
| text | Yes | What the voice will say |
| instruction | Yes | Describes the voice and style |
| output_file_path | Yes | Absolute path .wav or .flac |
| cfg_scale | No | Default 4.0 |
| seed | No | Different seed = different voice |

Important: instruction must be in the same language as text. Describing in English while the text is in Spanish degrades quality.

## Project Structure

    ScriptCastInferenceService/
    ├── app/
    │   ├── __init__.py
    │   ├── config.py              # Settings, MODELS_DIR, engine paths
    │   ├── main.py                # FastAPI app, DTOs, and endpoints
    │   └── engines/
    │       ├── __init__.py
    │       ├── base.py            # ABCs: BaseInferenceEngine, AudioCueProducer,
    │       │                      #       VoiceProducer, VoiceDesignProducer
    │       ├── manager.py         # EngineManager (registry and routing)
    │       ├── stable_audio.py    # StableAudioEngine
    │       ├── breeze.py          # BreezeEngine (subprocess to infer.py)
    │       └── omnivoice.py       # OmniVoiceEngine (direct Python API)
    ├── models/                    # Checkpoints (or --cache-dir)
    ├── requirements.txt
    ├── run.py                     # Entry point + timestamped logging
    └── README.md

## Design Notes

- **Lazy loading**: models load on the first request that uses them, not at startup. This keeps the service boot fast and only allocates VRAM for what's actually used.
- **Breeze as a subprocess**: the Breeze repo doesn't expose an importable package, so infer.py is executed as a child process. This isolates failures (if Breeze crashes, the service survives) at the cost of reloading the model per request.
- **OmniVoice in-process**: imported directly because it exposes a clean Python API (OmniVoice.from_pretrained(...).generate(...)).
- **Atomic writes**: all engines write to a temp file and call os.replace, so the client never sees a half-written file.
- **Path validation**: output_file_path must be absolute and end in .wav or .flac. Validated at the DTO level (422) and inside the engine (400) before spending VRAM.

## Troubleshooting

### No module named 'flash_attn'
Install the wheel matching your torch 2.8.0 + CUDA 12.8 + cp310 (see section 4). If you can't find it, ignore it: Stable Audio works without Flash Attention.

### operator torchvision::nms does not exist
torch and torchvision are out of sync. Reinstall both together:

    pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

### Seed must be between 0 and 2**32 - 1
The engine translates -1/null into a valid seed before passing it to NumPy. If you see this error, you're running infer.py manually with --seed -1; use a value >= 0.

### Template 'ref_clone_tata' does not define a negative prompt but cfg_scale=4.0
Breeze's infer.py does not accept cfg_scale > 1 with the pure clone template. Use cfg_scale=1.0 or pass instruction to trigger Voice Direction mode.

### SoX could not be found
torchaudio warning. Not blocking. Install SoX to silence it.

### OmniVoice takes a long time on the first request
It loads the checkpoint (~2.45 GB) into VRAM. Subsequent requests are fast.

## Licenses

- **Stable Audio 3**: Stability AI Community License (check commercial terms).
- **Breeze TTS 2**: weights are under a research / non-commercial license by RESONIA, INC. Commercial use requires written authorization.
- **OmniVoice**: Apache 2.0, but the model card explicitly prohibits impersonation, fraud, and unauthorized cloning.