"""Public Python API for the standalone CUDA FMMM layout engine.

The compiled native extension is treated as an implementation detail:
callers interact with typed dataclasses, convenient array helpers, and
graph framework integrations without managing device memory or pybind ABI seams.

Why does this facade exist?
Because raw pybind bindings have a habit of leaking C++ conventions (and
occasional segmentation faults) directly into Python space. This module
sanitizes inputs, handles flexible graph representations (integers, strings,
NetworkX graphs), and wraps results into backward-compatible dataclasses.
"""

from __future__ import annotations

import collections.abc
import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple, Union


class FmmmError(RuntimeError):
    """Raised when the CUDA FMMM backend is unavailable, crashes, or rejects an input."""


@dataclass(frozen=True)
class ObjectiveScore:
    """Lexicographic crossing objective (K, nK, Phi, C) produced by the GPU oracle."""

    valid: bool = True
    exact_valid: bool = True
    K: int = 0
    nK: int = 0
    Phi: int = 0
    C: int = 0
    edge_crossings: list[int] = field(default_factory=list)
    proposal_status: str = "accepted"
    fallback_required: bool = False
    fallback_reason: str = "none"

    @property
    def E(self) -> float:
        """Alias for energy / primary penalty term for backward compatibility."""
        return float(self.K)

    @property
    def O(self) -> float:
        """Combined total objective value (crossings + geometric penalties)."""
        return float(self.K + self.C)


@dataclass(frozen=True)
class TimingBreakdown:
    """Execution latency breakdown recorded on device and host streams in milliseconds."""

    layout_ms: float = -1.0
    device_ms: float = -1.0
    upload_ms: float = -1.0
    score_ms: float = -1.0
    download_ms: float = -1.0

    @property
    def total_ms(self) -> float:
        """Total latency across all recorded phases."""
        positive_times = [
            t for t in (self.layout_ms, self.device_ms, self.upload_ms, self.score_ms, self.download_ms)
            if t > 0.0
        ]
        return sum(positive_times) if positive_times else (self.layout_ms if self.layout_ms > 0 else 0.0)


@dataclass(frozen=True)
class LayoutRecord(collections.abc.Mapping):
    """Structured publication record returned by the CUDA layout engine.

    Implements the Mapping protocol so legacy code doing `result["coordinates"]`
    keeps working without requiring immediate migration to attribute access.
    """

    status: str
    coordinates: list[tuple[float, float]]
    score: ObjectiveScore
    timing: TimingBreakdown
    admitted: bool
    seed: int
    config: str
    reason: str = "none"
    width: float = 1000.0
    height: float = 1000.0
    layer_index: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_numpy(self):
        """Convert final layout coordinates into a 2D float64 NumPy array of shape (N, 2).

        Raises ImportError with actionable guidance if NumPy is not installed.
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "NumPy is required for to_numpy(). Install it via: pip install numpy"
            ) from exc

        return np.asarray(self.coordinates, dtype=np.float64)

    def to_torch(self, device: str = "cpu"):
        """Convert final layout coordinates into a 2D float64 PyTorch tensor of shape (N, 2).

        Raises ImportError with actionable guidance if PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for to_torch(). Install it via: pip install torch"
            ) from exc

        return torch.tensor(self.coordinates, dtype=torch.float64, device=device)

    # --- Mapping protocol methods for dict backward compatibility ------------
    def __getitem__(self, key: str) -> Any:
        if key == "coordinates":
            return self.coordinates
        if key == "status":
            return self.status
        if key == "score":
            return self.raw.get("score", self.score)
        if key == "timing" or key == "timings":
            return self.raw.get("timing", self.timing)
        if key == "width":
            return self.width
        if key == "height":
            return self.height
        if key == "layer_index":
            return self.layer_index
        if key in self.raw:
            return self.raw[key]
        raise KeyError(key)

    def __iter__(self):
        keys = {"status", "coordinates", "score", "timing", "width", "height", "layer_index"}
        keys.update(self.raw.keys())
        return iter(keys)

    def __len__(self) -> int:
        keys = {"status", "coordinates", "score", "timing", "width", "height", "layer_index"}
        keys.update(self.raw.keys())
        return len(keys)


_CUDA_MODULE = None


def _load_cuda_module():
    """Dynamically discover and load the compiled CUDA extension.

    Supports both the clean standalone name `_fmmm_cuda` and the legacy
    ABI name `planar_cuda_refactored`. Caches the result on success.
    """
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE

    last_err = None
    for module_name in ("_fmmm_cuda", "planar_cuda_refactored"):
        try:
            _CUDA_MODULE = importlib.import_module(module_name)
            return _CUDA_MODULE
        except ImportError as e:
            last_err = e

    raise FmmmError(
        f"CUDA FMMM extension is not installed or could not be loaded. "
        f"Build it with ./build.ps1 (Windows) or ./build.sh (Linux). "
        f"(Attempted '_fmmm_cuda' and 'planar_cuda_refactored'; last error: {last_err})"
    ) from last_err


def _backend():
    return _load_cuda_module()


def capabilities() -> dict[str, Any]:
    """Return backend capability metadata exposed by this package."""
    return {
        "backend": "fmmm-gpu",
        "cuda": True,
        "api": "fmmm_seed",
        "objective": "(K, nK, Phi, C)",
        "features": [
            "cuda_graph_capture",
            "aabb_crossing_pruning",
            "tiled_shared_memory",
            "early_stopping",
        ],
    }


def _normalize_nodes(
    nodes: Union[int, Sequence[Sequence[float]], Any],
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    """Normalize input nodes into a list of (x, y) float coordinate pairs.

    If an integer `node_count` is passed instead of explicit coordinates,
    we synthesize an initial circular placement within the interior canvas.
    This saves callers from having to manually fabricate initial dummy coordinates.
    """
    if isinstance(nodes, int):
        if nodes <= 0:
            raise ValueError(f"node_count must be positive; got {nodes}")
        if nodes == 1:
            return [(width * 0.5, height * 0.5)]

        center_x, center_y = width * 0.5, height * 0.5
        radius = min(width, height) * 0.35
        coords = []
        for i in range(nodes):
            theta = (2.0 * math.pi * i) / nodes
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            coords.append((float(x), float(y)))
        return coords

    if hasattr(nodes, "tolist"):
        nodes = nodes.tolist()

    out: list[tuple[float, float]] = []
    for item in nodes:
        if len(item) != 2:
            raise ValueError(f"Each node coordinate must be an (x, y) pair; got {item!r}")
        out.append((float(item[0]), float(item[1])))

    if not out:
        raise ValueError("node list cannot be empty")
    return out


def _normalize_edges(edges: Any) -> list[tuple[int, int, int]]:
    """Normalize edges into (source, target, layer) integer 3-tuples.

    Automatically promotes 2-tuples (u, v) to layer 0 so casual callers
    aren't forced to invent artificial layer tags for unlayered graphs.
    """
    if hasattr(edges, "tolist"):
        edges = edges.tolist()

    out: list[tuple[int, int, int]] = []
    for item in edges:
        if len(item) == 2:
            out.append((int(item[0]), int(item[1]), 0))
        elif len(item) == 3:
            out.append((int(item[0]), int(item[1]), int(item[2])))
        else:
            raise ValueError(f"Edge must be a 2-tuple or 3-tuple; got {item!r}")
    return out


def seed_layout(
    nodes: Union[int, Sequence[Sequence[float]], Any] = None,
    edges: Sequence[Sequence[int]] | Any = (),
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    seed: int = 0,
    iterations: int = 3,
    config: str = "ogdf-foxglove-202510-profile-1",
    layers_enabled: bool = True,
    time_exclude_score: bool = False,
    coordinate_type: str = "double",
    node_count: int | None = None,
) -> LayoutRecord:
    """Generate one GPU FMMM layout candidate and return its publication record.

    Parameters
    ----------
    nodes:
        Either an integer node count, or initial (x, y) node positions (N x 2).
    edges:
        Edge list as (source, target) or (source, target, layer).
    width, height:
        Layout bounding dimensions (must strictly exceed 1.0).
    seed:
        Deterministic PRNG seed for coarsest placement and perturbations.
    iterations:
        Base iteration multiplier per multilevel schedule (must be > 0).
    config:
        Pinned layout reference profile name.
    layers_enabled:
        Whether edge crossing predicates enforce layer separation.
    time_exclude_score:
        If True, layout_ms excludes post-layout GPU geometry admission and scoring.
    coordinate_type:
        Coordinate precision: 'double' or 'fp32'.
    node_count:
        Convenience keyword alias for passing node count as integer.

    Returns
    -------
    LayoutRecord:
        Immutable record carrying status, final coordinates, objective scores, and timing.
    """
    if width <= 1.0 or height <= 1.0:
        raise ValueError(f"width and height must exceed 1.0, got ({width}, {height})")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    target_nodes = node_count if node_count is not None else nodes
    if target_nodes is None:
        raise ValueError("Must specify either nodes or node_count")

    norm_nodes = _normalize_nodes(target_nodes, width=width, height=height)
    norm_edges = _normalize_edges(edges)

    backend = _backend()
    raw = backend.fmmm_seed(
        norm_nodes,
        norm_edges,
        max_width=int(width),
        max_height=int(height),
        seed=int(seed),
        config=str(config),
        iterations=int(iterations),
        layers_enabled=bool(layers_enabled),
        retain_device=False,
        time_exclude_score=bool(time_exclude_score),
        coordinate_type=str(coordinate_type),
    )

    if not isinstance(raw, dict):
        raise FmmmError("CUDA backend returned an invalid publication payload")

    score_dict = raw.get("score", {})
    score = ObjectiveScore(
        valid=bool(score_dict.get("valid", True)),
        exact_valid=bool(score_dict.get("exact_valid", True)),
        K=int(score_dict.get("K", 0)),
        nK=int(score_dict.get("nK", 0)),
        Phi=int(score_dict.get("Phi", 0)),
        C=int(score_dict.get("C", 0)),
        edge_crossings=list(score_dict.get("edge_crossings", [])),
        proposal_status=str(score_dict.get("proposal_status", "accepted")),
        fallback_required=bool(score_dict.get("fallback_required", False)),
        fallback_reason=str(score_dict.get("fallback_reason", "none")),
    )

    timing = TimingBreakdown(
        layout_ms=float(raw.get("layout_ms", -1.0)),
        device_ms=float(raw.get("device_ms", -1.0)),
        upload_ms=float(raw.get("upload_ms", -1.0)),
        score_ms=float(raw.get("score_admit_publish_ms", -1.0)),
        download_ms=float(raw.get("final_download_ms", -1.0)),
    )

    layer_indices = list(raw.get("layer_index", [0] * len(norm_nodes)))

    return LayoutRecord(
        status=str(raw.get("status", "converged")),
        coordinates=list(raw.get("coordinates", [])),
        score=score,
        timing=timing,
        admitted=bool(raw.get("admitted", True)),
        seed=int(seed),
        config=str(config),
        reason=str(raw.get("reason", "none")),
        width=float(width),
        height=float(height),
        layer_index=layer_indices,
        raw=raw,
    )


def layout_networkx(
    graph: Any,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    seed: int = 0,
    iterations: int = 3,
    config: str = "ogdf-foxglove-202510-profile-1",
    initial_positions: Mapping[Any, Sequence[float]] | None = None,
    return_dict: bool = True,
) -> Union[dict[Any, tuple[float, float]], LayoutRecord]:
    """Compute a GPU FMMM layout for a NetworkX graph.

    Parameters
    ----------
    graph:
        A NetworkX Graph, DiGraph, or MultiGraph.
    width, height:
        Canvas dimensions (> 1.0).
    seed:
        RNG seed for deterministic layout generation.
    iterations:
        Multilevel force calculation budget.
    config:
        FMMM profile name.
    initial_positions:
        Optional mapping of node labels to initial (x, y) coordinates.
    return_dict:
        If True (default), returns a `{node: (x, y)}` position dictionary ready
        for `networkx.draw(G, pos=pos)`. If False, returns the complete `LayoutRecord`.

    Returns
    -------
    dict or LayoutRecord:
        Position dictionary or full layout record.
    """
    node_list = list(graph.nodes())
    if not node_list:
        if return_dict:
            return {}
        raise ValueError("Cannot lay out an empty NetworkX graph")

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    nodes_coords: list[tuple[float, float]] = []
    if initial_positions:
        for node in node_list:
            pos = initial_positions.get(node, (width * 0.5, height * 0.5))
            nodes_coords.append((float(pos[0]), float(pos[1])))
    else:
        center_x, center_y = width * 0.5, height * 0.5
        radius = min(width, height) * 0.35
        n_count = len(node_list)
        for idx in range(n_count):
            theta = (2.0 * math.pi * idx) / max(1, n_count)
            nodes_coords.append((center_x + radius * math.cos(theta), center_y + radius * math.sin(theta)))

    edges_list: list[tuple[int, int, int]] = []
    for edge in graph.edges(data=True):
        u, v = edge[0], edge[1]
        data = edge[2] if len(edge) > 2 else {}
        layer = int(data.get("layer", 0)) if isinstance(data, dict) else 0
        if u in node_to_idx and v in node_to_idx:
            edges_list.append((node_to_idx[u], node_to_idx[v], layer))

    record = seed_layout(
        nodes_coords,
        edges_list,
        width=width,
        height=height,
        seed=seed,
        iterations=iterations,
        config=config,
    )

    if return_dict:
        return {node: record.coordinates[idx] for idx, node in enumerate(node_list)}
    return record
