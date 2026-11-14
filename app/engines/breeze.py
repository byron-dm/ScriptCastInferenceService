import atexit
import json
import os
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

SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})

WORKER_SCRIPT = Path(__file__).with_name("_breeze_worker.py")
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
            ready = self._read_response(worker)
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
    def _read_response(worker: subprocess.Popen) -> dict:
        assert worker.stdout is not None
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(
                "Breeze worker closed stdout unexpectedly; "
                "check its stderr for the underlying error."
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Breeze worker returned invalid JSON: {line!r}"
            ) from exc

    def _request(self, payload: dict) -> dict:
        if not self.is_loaded:
            self.load_model()
        worker = self._worker
        if worker is None or worker.stdin is None:
            raise RuntimeError("Breeze worker is not running")

        try:
            worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                f"Breeze worker is not accepting requests: {exc}"
            ) from exc

        return self._read_response(worker)

    # --- validation ------------------------------------------------------

    def _prepare_output_path(self, output_file_path: str) -> Path:
        if not output_file_path:
            raise ValueError("output_file_path cannot be empty")

        dest_path = Path(output_file_path)
        if not dest_path.is_absolute():
            raise ValueError("output_file_path must be an absolute path")
        if dest_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension: '{dest_path.suffix}'. "
                f"Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        parent = dest_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Cannot create directory {parent}: {exc}") from exc
        if not os.access(parent, os.W_OK):
            raise ValueError(f"No write permission for {parent}")

        return dest_path

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

        dest_path = self._prepare_output_path(output_file_path)

        args = {
            "text": text,
            "output_file_path": str(dest_path),
            "reference_audio_path": reference_audio_path,
            "reference_text": reference_text,
            "instruction": instruction,
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

        dest_path = self._prepare_output_path(output_file_path)

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