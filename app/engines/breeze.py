import atexit
import json
import logging
import os
import queue
import random
import subprocess
import sys
import threading
from pathlib import Path

from app.config import settings
from app.engines.base import (
    BaseInferenceEngine,
    VoiceDesignProducer,
    VoiceProducer,
)
from app.utils.io_utils import prepare_output_path

logger = logging.getLogger(__name__)

WORKER_SCRIPT = (
    Path(__file__).resolve().parent.parent / "workers" / "_breeze_worker.py"
)
WORKER_READY_TIMEOUT_S = 600.0
WORKER_REQUEST_TIMEOUT_S = 600.0


class BreezeEngine(BaseInferenceEngine, VoiceProducer, VoiceDesignProducer):
    """Breeze TTS 2 engine backed by a persistent worker subprocess.

    The worker loads the runtime once and serves requests over NDJSON on
    stdin/stdout. It reuses Breeze's modules directly (without importing or
    modifying breeze-tts/infer.py), so the upstream CLI is untouched.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or str(settings.BREEZE_MODEL_PATH)
        self._device: str = settings.DEVICE
        # `_lock` protects load/unload; `_infer_lock` serializes inference.
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._worker: subprocess.Popen | None = None
        atexit.register(self._shutdown_worker)

    @property
    def is_loaded(self) -> bool:
        return self._worker is not None and self._worker.poll() is None

    # --- lifecycle -------------------------------------------------------

    def load_model(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return

            code_root = settings.BREEZE_CODE_PATH
            infer_script = code_root / "infer.py"
            if not infer_script.is_file():
                raise RuntimeError(
                    f"infer.py not found at {infer_script}. Clone the repo with: "
                    f"git clone https://github.com/breezeblue-ai/breeze-tts.git {code_root}"
                )

            model_path = Path(self._model_path)
            if not model_path.is_dir() or not any(model_path.iterdir()):
                raise RuntimeError(
                    f"Breeze checkpoint not found at {model_path}. Download it with: "
                    f"huggingface-cli download BreezeBlue/Breeze-TTS-2 --local-dir {model_path}"
                )

            if not WORKER_SCRIPT.is_file():
                raise RuntimeError(
                    f"Breeze worker script not found at {WORKER_SCRIPT}"
                )

            self._start_worker(model_path, code_root)

    def unload_model(self) -> None:
        with self._lock:
            self._shutdown_worker()

    # --- worker plumbing -------------------------------------------------

    def _start_worker(self, model_path: Path, code_root: Path) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        worker = subprocess.Popen(
            [
                sys.executable,
                str(WORKER_SCRIPT),
                str(model_path),
                str(code_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit; diagnostics go to the service's stderr
            cwd=str(code_root),
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        try:
            ready = self._read_response(worker, timeout_s=WORKER_READY_TIMEOUT_S)
        except Exception:
            self._kill_process(worker)
            raise

        if not ready.get("ok"):
            self._kill_process(worker)
            raise RuntimeError(
                f"Breeze worker failed to start: {ready.get('error', 'unknown')}"
            )
        if not ready.get("ready"):
            self._kill_process(worker)
            raise RuntimeError(
                "Breeze worker did not report readiness; check its stderr."
            )

        self._worker = worker

    def _shutdown_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        if worker.poll() is None and worker.stdin is not None:
            try:
                worker.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                worker.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._kill_process(worker)
        else:
            self._kill_process(worker)

    @staticmethod
    def _kill_process(worker: subprocess.Popen) -> None:
        if worker.poll() is None:
            try:
                worker.kill()
            except OSError:
                pass
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _read_response(worker: subprocess.Popen, timeout_s: float) -> dict:
        """Read a single NDJSON line from worker stdout with a hard timeout."""
        assert worker.stdout is not None
        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def _reader() -> None:
            try:
                line = worker.stdout.readline()  # type: ignore[union-attr]
                result_queue.put(line)
            except BaseException as exc:
                result_queue.put(exc)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        try:
            item = result_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Breeze worker did not respond within {timeout_s}s."
            ) from exc

        if isinstance(item, BaseException):
            raise RuntimeError(
                f"Error reading from Breeze worker stdout: {item}"
            ) from item

        if not item:
            raise RuntimeError(
                "Breeze worker closed stdout unexpectedly; "
                "check its stderr for the underlying error."
            )

        try:
            return json.loads(item)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Breeze worker returned invalid JSON: {item!r}"
            ) from exc

    def _restart_worker(self) -> None:
        """Forcefully terminate a hung/dead worker and spawn a fresh instance."""
        logger.warning("Restarting Breeze worker subprocess...")
        with self._lock:
            old_worker = self._worker
            self._worker = None
            if old_worker is not None:
                self._kill_process(old_worker)
        self.load_model()

    def _request(self, payload: dict) -> dict:
        if not self.is_loaded:
            self.load_model()

        worker = self._worker
        if worker is None or worker.stdin is None:
            raise RuntimeError("Breeze worker is not running")

        try:
            worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            return self._read_response(worker, timeout_s=WORKER_REQUEST_TIMEOUT_S)
        except TimeoutError:
            logger.error(
                "Breeze worker timed out after %ss; restarting subprocess.",
                WORKER_REQUEST_TIMEOUT_S,
            )
            self._restart_worker()
            raise RuntimeError(
                f"Breeze inference timed out after {WORKER_REQUEST_TIMEOUT_S}s "
                "and the worker was restarted."
            )
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            # If the worker crashed or broke the pipe, reset it so the next
            # request can recover cleanly instead of failing permanently.
            if not self.is_loaded:
                logger.warning("Breeze worker died during request; cleaning up.")
                with self._lock:
                    if self._worker is not None:
                        self._kill_process(self._worker)
                        self._worker = None
            raise RuntimeError(
                f"Breeze worker communication failed: {exc}"
            ) from exc

    # --- validation ------------------------------------------------------

    @staticmethod
    def _resolve_seed(seed: int | None) -> int:
        if seed is None or seed == -1:
            return random.randint(0, 2**32 - 1)
        return seed

    # --- public API ------------------------------------------------------

    def generate_voice(
        self,
        text: str,
        output_file_path: str,
        reference_audio_path: str | None = None,
        reference_text: str | None = None,
        instruction: str | None = None,
        cfg_scale: float = 4.0,
        speed: float = 1.0,
        language: str | None = None,  # ignored by Breeze
        seed: int | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        if not reference_audio_path:
            raise ValueError("reference_audio_path is required for Breeze TTS 2")
        if not Path(reference_audio_path).is_file():
            raise ValueError(f"Reference audio not found: {reference_audio_path}")

        if not reference_text or not reference_text.strip():
            raise ValueError(
                "reference_text is required and must match the exact spoken "
                "content of the reference audio."
            )

        has_instruction = bool(instruction and instruction.strip())
        if has_instruction:
            mode = "voice-direction"
            effective_cfg_scale = cfg_scale
        else:
            mode = "voice-clone"
            effective_cfg_scale = 1.0
            if cfg_scale != 1.0:
                logger.info(
                    "Breeze %s: ignoring cfg_scale=%s (no instruction provided; "
                    "forcing 1.0). Add 'instruction' to enable Voice Direction.",
                    mode,
                    cfg_scale,
                )

        logger.info(
            "Breeze %s: cfg_scale=%s seed=%s text_len=%d has_ref_text=%s",
            mode,
            effective_cfg_scale,
            seed,
            len(text),
            bool(reference_text),
        )

        dest_path = prepare_output_path(output_file_path)

        args = {
            "text": text,
            "output_file_path": str(dest_path),
            "reference_audio_path": reference_audio_path,
            "reference_text": reference_text,
            "instruction": instruction,
            "cfg_scale": effective_cfg_scale,
            "seed": self._resolve_seed(seed),
        }

        with self._infer_lock:
            response = self._request({"cmd": "generate", "args": args})
            if not response.get("ok"):
                raise RuntimeError(
                    f"Breeze inference failed: {response.get('error', 'unknown')}"
                )
            file_path = response.get("file_path")
            if not file_path or not Path(str(file_path)).is_file():
                raise RuntimeError(
                    f"Breeze reported success but no file was written to {dest_path}"
                )

        return str(dest_path)

    def generate_voice_design(
        self,
        text: str,
        instruction: str,
        output_file_path: str,
        cfg_scale: float = 4.0,
        seed: int | None = None,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        if not instruction or not instruction.strip():
            raise ValueError(
                "instruction is required for voice design and must describe "
                "the target speaker and delivery style."
            )

        dest_path = prepare_output_path(output_file_path)

        args = {
            "text": text,
            "instruction": instruction,
            "output_file_path": str(dest_path),
            "cfg_scale": cfg_scale,
            "seed": self._resolve_seed(seed),
        }

        with self._infer_lock:
            response = self._request({"cmd": "generate", "args": args})
            if not response.get("ok"):
                raise RuntimeError(
                    f"Breeze inference failed: {response.get('error', 'unknown')}"
                )
            file_path = response.get("file_path")
            if not file_path or not Path(str(file_path)).is_file():
                raise RuntimeError(
                    f"Breeze reported success but no file was written to {dest_path}"
                )

        return str(dest_path)