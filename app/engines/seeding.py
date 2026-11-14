from contextlib import contextmanager
import random
from typing import Iterator

import numpy as np
import torch


@contextmanager
def isolated_seed(seed: int | None, device: str = "cpu") -> Iterator[None]:
    """Temporarily seed Python, NumPy and PyTorch, then restore global state.

    This avoids polluting the process-wide RNG state (which would affect the
    HTTP server and any other code using `random`, `np.random` or `torch`).
    It also makes seeding safe if inference is ever parallelized: each call
    restores the previous state on exit, so concurrent calls do not corrupt
    each other's seeds.

    If `seed` is None, the RNGs are left untouched (random per-call behavior).
    """
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
