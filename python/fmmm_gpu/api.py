"""Public Python API; the compiled extension is intentionally an implementation detail."""

from __future__ import annotations

import importlib
from typing import Sequence


class FmmmError(RuntimeError):
    """Raised when the CUDA FMMM backend is unavailable or rejects an input."""


def _backend():
    try:
        return importlib.import_module("planar_cuda_refactored")
    except ImportError as exc:
        raise FmmmError(
            "CUDA FMMM extension is not installed; build fmmm_gpu and check CUDA/Python compatibility"
        ) from exc


def capabilities() -> dict:
    """Return the backend capability metadata exposed by this package."""
    return {"backend": "fmmm-gpu", "cuda": True, "api": "fmmm_seed"}


def seed_layout(
    nodes: Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
    *,
    width: int,
    height: int,
    seed: int = 0,
    iterations: int = 3,
    config: str = "ogdf-foxglove-202510-profile-1",
    layers_enabled: bool = True,
) -> dict:
    """Generate one GPU FMMM layout candidate and return its publication record."""
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be greater than 1")
    backend = _backend()
    result = backend.fmmm_seed(
        [(float(x), float(y)) for x, y in nodes],
        [(int(source), int(target), int(layer)) for source, target, layer in edges],
        max_width=int(width),
        max_height=int(height),
        seed=int(seed),
        config=str(config),
        iterations=int(iterations),
        layers_enabled=bool(layers_enabled),
    )
    if not isinstance(result, dict):
        raise FmmmError("backend returned a non-dict publication record")
    return result
