"""Persistent Breeze worker.

Loads the Breeze runtime once and serves generation requests over NDJSON on
stdin/stdout. It reuses the same modules as breeze-tts/infer.py (runtime,
templates, streaming runtime) without importing or modifying infer.py, so the
upstream CLI keeps working unchanged.

Note: Breeze and its native C++/CUDA dependencies may write directly to OS file
descriptor 1 (stdout). We duplicate fd 1 into a private channel for NDJSON
replies and point fd 1 to fd 2 (stderr) so the protocol pipe stays 100% clean.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path

DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
FAST_CONFIG_NAME = "fast.json"

# Duplicate the original stdout file descriptor (fd 1) for NDJSON responses,
# then redirect OS-level fd 1 -> fd 2 (stderr) so neither Python print() nor
# C++/CUDA libraries can corrupt the JSON protocol stream.
_PROTOCOL_FD = os.dup(1)
os.dup2(2, 1)
_PROTOCOL_OUT = os.fdopen(_PROTOCOL_FD, "w", encoding="utf-8", buffering=1)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(payload: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _PROTOCOL_OUT.flush()


class BreezeRuntime:
    """Loads Breeze once and generates audio per request."""

    def __init__(self, model_path: Path, code_root: Path) -> None:
        sys.path.insert(0, str(code_root))

        runtime_mod = importlib.import_module("breeze_infer.runtime")
        templates_mod = importlib.import_module("breeze_infer.templates")
        fast_streaming_mod = importlib.import_module("models.fast_streaming")
        warmup_mod = importlib.import_module("models.warmup_profile")

        self._set_all_seeds = runtime_mod.set_all_seeds
        self._get_template = templates_mod.get_template
        self._prepare_inputs = templates_mod.prepare_inputs
        self._select_template_name = templates_mod.select_template_name
        self._load_warmup_profile = warmup_mod.load_warmup_profile

        fast_breeze_streaming_runtime = fast_streaming_mod.FastBreezeStreamingRuntime
        fast_streaming_config = fast_streaming_mod.FastStreamingConfig

        _log(f"loading Breeze runtime from {model_path}")

        tokenizer, model, audio_tokenizer = runtime_mod.load_runtime(
            model_path,
            device=runtime_mod.resolve_device(),
            attn_implementation="eager",
        )
        runtime_mod.update_generation_config_for_breeze(model)

        config = fast_streaming_config(
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
        self._runtime = fast_breeze_streaming_runtime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )

        if self._runtime.fast_enabled:
            profile = self._load_warmup_profile(
                code_root / "configs" / FAST_CONFIG_NAME
            )
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

    def _cleanup_request(self, req_id: str) -> None:
        """Best-effort release of KV cache / residual state in Breeze runtime."""
        for cleanup_method in ("close_request", "reset_request", "reset"):
            fn = getattr(self._runtime, cleanup_method, None)
            if callable(fn):
                try:
                    fn(req_id)
                except TypeError:
                    fn()
                except Exception as exc:
                    _log(f"Non-fatal cleanup warning ({cleanup_method}): {exc}")
                break

    def generate(self, args: dict) -> str:
        """Run one generation and return the output path."""
        import tempfile
        import time
        import soundfile as sf

        t0 = time.perf_counter()
        text = args["text"]
        output_path = Path(args["output_file_path"])
        instruction = args.get("instruction")

        ref_audio_raw = args.get("reference_audio_path")
        reference_audio_path = (
            str(ref_audio_raw) if ref_audio_raw is not None else None
        )

        ref_text_raw = args.get("reference_text")
        reference_text = str(ref_text_raw) if ref_text_raw is not None else None

        cfg_scale = float(args.get("cfg_scale", DEFAULT_CFG_SCALE))
        seed = int(args.get("seed", 42))

        if reference_audio_path is not None:
            if not Path(reference_audio_path).is_file():
                raise FileNotFoundError(
                    f"Reference audio not found: {reference_audio_path}"
                )

        req_id = f"req-{uuid.uuid4().hex[:12]}"

        request = {
            "id": req_id,
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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix,
        )
        os.close(fd)

        try:
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

            with sf.SoundFile(
                tmp_name,
                mode="w",
                samplerate=self._runtime.sample_rate,
                channels=1,
                subtype="PCM_16",
            ) as output_file:
                for chunk in self._runtime.iter_audio_chunks(
                    inputs, request_id=req_id, seed=seed
                ):
                    output_file.write(chunk.audio)
                output_file.flush()

            os.replace(tmp_name, output_path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        finally:
            self._cleanup_request(req_id)

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Breeze finished generation but output file is missing or empty: {output_path}"
            )

        elapsed = time.perf_counter() - t0
        _log(f"Breeze finished {req_id} in {elapsed:.1f}s -> {output_path}")
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