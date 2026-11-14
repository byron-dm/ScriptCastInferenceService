import os
from pathlib import Path


SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac"})


def prepare_output_path(output_file_path: str) -> Path:
    """Validate and prepare an absolute output path for audio files.

    Creates parent directories if needed and checks that the destination is
    writable. Returns the resolved Path. Raises ValueError with a clear
    message on any validation failure.

    This is the single source of truth for output-path validation across all
    engines.
    """
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

    if dest_path.exists() and not os.access(dest_path, os.W_OK):
        raise ValueError(f"Existing file is not writable: {dest_path}")

    return dest_path