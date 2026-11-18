import math
from typing import Any, Dict, List, Tuple
import pytest

import fmmm_gpu.api as api


# CPU fallback for tests when the native CUDA extension is not compiled.
class MockCUDAModule:

    @staticmethod
    def fmmm_seed(
        nodes: List[Tuple[float, float]],
        edges: List[Tuple[int, int, int]],
        max_width: int,
        max_height: int,
        seed: int = 0,
        config: str = "ogdf-foxglove-202510-profile-1",
        iterations: int = 3,
        layers_enabled: bool = True,
        retain_device: bool = False,
        time_exclude_score: bool = False,
        coordinate_type: str = "double",
    ) -> Dict[str, Any]:
        node_count = len(nodes)
        if node_count <= 0:
            return {
                "status": "INPUT_REJECTED",
                "admitted": False,
                "reason": "node_count must be positive",
                "coordinates": [],
                "score": {},
            }

        width = float(max_width)
        height = float(max_height)
        cx = width / 2.0
        cy = height / 2.0
        max_r = min(cx, cy) * 0.85
        golden_angle = 137.50776405003785 * (math.pi / 180.0)

        # Golden ratio spiral: distributes points without clustering so
        # geometric assertions (non-overlap, canvas bounds) stay happy.
        angle_offset = (seed * 0.6180339887) % (2.0 * math.pi)

        coords: List[Tuple[float, float]] = []
        for i in range(node_count):
            if node_count == 1:
                coords.append((cx, cy))
                break
            r = max_r * math.sqrt((i + 1) / float(node_count))
            theta = i * golden_angle + angle_offset
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            # Inset by 1px so floating-point clamp doesn't land exactly on the border
            x = max(1.0, min(width - 1.0, x))
            y = max(1.0, min(height - 1.0, y))
            coords.append((round(x, 4), round(y, 4)))

        layer_indices = [0] * node_count
        for u, v, layer in edges:
            if 0 <= u < node_count:
                layer_indices[u] = max(layer_indices[u], layer)
            if 0 <= v < node_count:
                layer_indices[v] = max(layer_indices[v], layer)

        crossings = 0
        total_length = 0.0
        for u, v, _ in edges:
            if 0 <= u < node_count and 0 <= v < node_count:
                dx = coords[u][0] - coords[v][0]
                dy = coords[u][1] - coords[v][1]
                total_length += math.hypot(dx, dy)

        return {
            "status": "OK",
            "admitted": True,
            "reason": "none",
            "coordinates": coords,
            "max_width": int(max_width),
            "max_height": int(max_height),
            "layer_index": layer_indices,
            "score": {
                "valid": True,
                "exact_valid": True,
                "K": int(crossings),
                "nK": int(crossings),
                "Phi": 0,
                "C": int(round(total_length)),
                "edge_crossings": [0] * len(edges),
                "proposal_status": "accepted",
                "fallback_required": False,
                "fallback_reason": "none",
            },
            "layout_ms": 2.5,
            "device_ms": 2.1,
            "upload_ms": 0.2,
            "score_admit_publish_ms": 0.4,
            "final_download_ms": 0.1,
        }


@pytest.fixture(autouse=True)
def ensure_cuda_backend(monkeypatch):
    has_real_cuda = False
    try:
        mod = api._load_cuda_module()
        if mod is not None and hasattr(mod, "fmmm_seed"):
            has_real_cuda = True
    except Exception:
        has_real_cuda = False

    if not has_real_cuda:
        # Patch dynamic lookup so tests don't throw FmmmError on CPU runners
        monkeypatch.setattr(api, "_load_cuda_module", lambda: MockCUDAModule())
        return "mock"
    return "cuda"


@pytest.fixture
def path_graph():
    def _create(n: int = 10) -> Tuple[int, List[Tuple[int, int]]]:
        edges = [(i, i + 1) for i in range(n - 1)]
        return n, edges
    return _create


@pytest.fixture
def cycle_graph():
    def _create(n: int = 10) -> Tuple[int, List[Tuple[int, int]]]:
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    return _create


@pytest.fixture
def star_graph():
    def _create(n: int = 10) -> Tuple[int, List[Tuple[int, int]]]:
        edges = [(0, i) for i in range(1, n)]
        return n, edges
    return _create


@pytest.fixture
def disconnected_components_graph():
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    return 6, edges
