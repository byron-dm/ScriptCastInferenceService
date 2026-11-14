"""Persistent Breeze worker.

Loads the Breeze runtime once and serves generation requests over NDJSON on
stdin/stdout. It reuses the same modules as breeze-tts/infer.py (runtime,
templates, streaming runtime) without importing or modifying infer.py, so the
upstream CLI keeps working unchanged.

Note: Breeze's internal modules use `print()` to stdout for diagnostics.
We redirect sys.stdout to sys.stderr during initialization and generation to
keep the NDJSON channel clean.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
FAST_CONFIG_NAME = "fast.json"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@contextmanager
def _suppress_stdout():
    """Redirect stdout to stderr to prevent Breeze's prints from corrupting
    the NDJSON protocol channel."""
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original_stdout


class BreezeRuntime:
    """Loads Breeze once and generates audio per request."""

    def __init__(self, model_path: Path, code_root: Path) -> None:
        # Add Breeze to sys.path at runtime.
        sys.path.insert(0, str(code_root))

        # Dynamic imports avoid IDE static analysis errors.
        runtime_mod = importlib.import_module("breeze_infer.runtime")
        templates_mod = importlib.import_module("breeze_infer.templates")
        fast_streaming_mod = importlib.import_module("models.fast_streaming")
        warmup_mod = importlib.import_module("models.warmup_profile")

        self._set_all_seeds = runtime_mod.set_all_seeds
        self._get_template = templates_mod.get_template
        self._prepare_inputs = templates_mod.prepare_inputs
        self._select_template_name = templates_mod.select_template_name
        self._load_warmup_profile = warmup_mod.load_warmup_profile

        FastBreezeStreamingRuntime = fast_streaming_mod.FastBreezeStreamingRuntime
        FastStreamingConfig = fast_streaming_mod.FastStreamingConfig

        _log(f"loading Breeze runtime from {model_path}")

        # Suppress Breeze's internal print() calls during load.
        with _suppress_stdout():
            tokenizer, model, audio_tokenizer = runtime_mod.load_runtime(
                model_path,
                device=runtime_mod.resolve_device(),
                attn_implementation="eager",
            )
            runtime_mod.update_generation_config_for_breeze(model)

            config = FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
                fast_all=None,
                fast_text_encoder=False,
                fast_backbone_prefill=False,
                fast_backbone_decode=False,
                fast_depth_decoder=False,
                fast_codec=False,
                repetition_penalty=REPETITION_PENALTY,
            )
            self._runtime = FastBreezeStreamingRuntime(
                model, audio_tokenizer, config, tokenizer=tokenizer
            )

            if self._runtime.fast_enabled:
                profile = self._load_warmup_profile(code_root / "configs" / FAST_CONFIG_NAME)
                from dataclasses import replace
                profile = replace(
                    profile, codec_chunk_frames=self._runtime.codec_chunk_frames
                )
                manifest = self._runtime.warmup_from_profile(profile)
                _log(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms")

        self._tokenizer = tokenizer
        self._audio_tokenizer = audio_tokenizer
        self._model = model

        _log("Breeze runtime ready")

    def generate(self, args: dict) -> str:
        """Run one generation and return the output path."""
        import soundfile as sf

        text = args["text"]
        output_path = Path(args["output_file_path"])
        instruction = args.get("instruction")

        ref_audio_raw = args.get("reference_audio_path")
        reference_audio_path = str(ref_audio_raw) if ref_audio_raw is not None else None

        ref_text_raw = args.get("reference_text")
        reference_text = str(ref_text_raw) if ref_text_raw is not None else None

        cfg_scale = float(args.get("cfg_scale", DEFAULT_CFG_SCALE))
        seed = int(args.get("seed", 42))

        if reference_audio_path is not None:
            if not Path(reference_audio_path).is_file():
                raise FileNotFoundError(
                    f"Reference audio not found: {reference_audio_path}"
                )

        request = {
            "id": "single-request",
            "text": str(text),
            "speaker": "S0",
        }
        if instruction:
            request["instruction"] = str(instruction).strip()

        if reference_audio_path is not None:
            request["ref_audio_path"] = reference_audio_path
            if reference_text:
                request["ref_text"] = reference_text.strip()

        template_name = self._select_template_name(request)
        self._set_all_seeds(seed)

        # Suppress Breeze's print() during inference as well.
        with _suppress_stdout():
            inputs = self._prepare_inputs(
                self._tokenizer,
                self._audio_tokenizer,
                self._model,
                [request],
                self._get_template(template_name),
                guidance_scale=cfg_scale,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with sf.SoundFile(
                output_path,
                mode="w",
                samplerate=self._runtime.sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as output_file:
                for chunk in self._runtime.iter_audio_chunks(
                    inputs, request_id="single-request", seed=seed
                ):
                    output_file.write(chunk.audio)

        return str(output_path)


def main() -> int:
    if len(sys.argv) != 3:
        _log("usage: _breeze_worker.py <model_path> <code_root>")
        return 2

    model_path = Path(sys.argv[1])
    code_root = Path(sys.argv[2])

    try:
        runtime = BreezeRuntime(model_path, code_root)
    except Exception as exc:
        _reply({"ok": False, "error": f"worker init failed: {exc}"})
        _log(traceback.format_exc())
        return 1

    _reply({"ok": True, "ready": True})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"invalid request JSON: {exc}"})
            continue

        cmd = req.get("cmd")
        args = req.get("args") or {}

        try:
            if cmd == "generate":
                file_path = runtime.generate(args)
            elif cmd == "shutdown":
                _reply({"ok": True, "bye": True})
                return 0
            else:
                raise ValueError(f"unknown cmd: {cmd!r}")
            _reply({"ok": True, "file_path": file_path})
        except Exception as exc:
            _reply({"ok": False, "error": str(exc)})
            _log(traceback.format_exc())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())