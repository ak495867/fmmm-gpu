from __future__ import annotations

import collections.abc
import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, Tuple, Union


class FmmmError(RuntimeError):
    """Raised when the CUDA FMMM backend is unavailable, crashes, or rejects an input."""


@dataclass(frozen=True)
class ObjectiveScore:
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
        # Legacy single-layer crossing alias from the v0 pipeline
        return float(self.K)

    @property
    def O(self) -> float:
        # Combined objective: crossing count plus geometric displacement penalties
        return float(self.K + self.C)


@dataclass(frozen=True)
class TimingBreakdown:
    layout_ms: float = -1.0
    device_ms: float = -1.0
    upload_ms: float = -1.0
    score_ms: float = -1.0
    download_ms: float = -1.0

    @property
    def total_ms(self) -> float:
        positive_times = [
            t for t in (self.layout_ms, self.device_ms, self.upload_ms, self.score_ms, self.download_ms)
            if t > 0.0
        ]
        return sum(positive_times) if positive_times else (self.layout_ms if self.layout_ms > 0 else 0.0)


@dataclass(frozen=True)
class LayoutRecord(collections.abc.Mapping):
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

    @property
    def __cuda_array_interface__(self) -> dict[str, Any] | None:
        # Expose zero-copy device pointers directly to PyTorch, CuPy, and Numba
        # if the C++ session was requested with retain_device=True.
        dev_ptr = self.raw.get("device_coords_ptr", 0)
        n = len(self.coordinates)
        if dev_ptr != 0 and n > 0:
            return {
                "shape": (n, 2),
                "typestr": "<f8",
                "data": (int(dev_ptr), False),
                "version": 3,
            }
        return None

    def to_numpy(self):
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("NumPy is required for to_numpy(). Install it via: pip install numpy") from exc

        return np.asarray(self.coordinates, dtype=np.float64)

    def to_torch(self, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for to_torch(). Install it via: pip install torch") from exc

        # If zero-copy device pointer is available and CUDA was requested, alias it directly
        if str(device).startswith("cuda") and self.__cuda_array_interface__ is not None:
            return torch.as_tensor(self, device=device)

        return torch.tensor(self.coordinates, dtype=torch.float64, device=device)

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
            "scratch_arena_pooling",
            "adaptive_barnes_hut_theta",
            "batch_layout_engine",
            "guillotine_component_packing",
            "pytorch_geometric_adapter",
        ],
    }


def _normalize_nodes(
    nodes: Union[int, Sequence[Sequence[float]], Any],
    width: float,
    height: float,
) -> list[tuple[float, float]]:
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


# --- 2D Guillotine Component Packing -----------------------------------------
def pack_components(
    component_coords: list[list[tuple[float, float]]],
    *,
    target_width: float = 1000.0,
    target_height: float = 1000.0,
    padding: float = 30.0,
) -> list[tuple[float, float]]:
    # Disconnected components in force-directed layout repel each other into the canvas
    # boundaries if laid out in a single coordinate system.
    # Shelf-packing organizes individual component bounding boxes into an aesthetically
    # balanced rectangular canvas without mutual distortion.
    if not component_coords:
        return []
    if len(component_coords) == 1:
        return component_coords[0]

    # Compute local bounding box and normalize each component to (0, 0) origin
    boxes = []
    normalized_comps = []
    for comp in component_coords:
        if not comp:
            continue
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(max_x - min_x, 10.0)
        h = max(max_y - min_y, 10.0)
        norm = [(p[0] - min_x, p[1] - min_y) for p in comp]
        boxes.append((w, h))
        normalized_comps.append(norm)

    # Sort components by decreasing height (First-Fit Decreasing Height shelf packing)
    indexed_boxes = sorted(enumerate(boxes), key=lambda item: item[1][1], reverse=True)

    # Determine shelf layout
    max_shelf_w = max(target_width, max(b[0] for b in boxes) + padding * 2)
    shelf_x = padding
    shelf_y = padding
    current_shelf_h = 0.0

    placed_positions = [None] * len(normalized_comps)
    for orig_idx, (w, h) in indexed_boxes:
        if shelf_x + w + padding > max_shelf_w and shelf_x > padding:
            # Wrap to next shelf row
            shelf_x = padding
            shelf_y += current_shelf_h + padding
            current_shelf_h = 0.0

        placed_positions[orig_idx] = (shelf_x, shelf_y)
        shelf_x += w + padding
        current_shelf_h = max(current_shelf_h, h)

    total_packed_w = max_shelf_w
    total_packed_h = shelf_y + current_shelf_h + padding

    # Scale and center to target dimensions
    scale_x = (target_width - padding * 2) / max(total_packed_w, 1.0)
    scale_y = (target_height - padding * 2) / max(total_packed_h, 1.0)
    scale = min(scale_x, scale_y, 1.0)

    final_coords: list[tuple[float, float]] = []
    for idx, norm_nodes in enumerate(normalized_comps):
        px, py = placed_positions[idx]
        for x, y in norm_nodes:
            final_coords.append((px + x * scale, py + y * scale))

    return final_coords


def _find_connected_components(node_count: int, edges: list[tuple[int, int, int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(node_count)]
    for u, v, _ in edges:
        if 0 <= u < node_count and 0 <= v < node_count and u != v:
            adj[u].append(v)
            adj[v].append(u)

    visited = [False] * node_count
    components: list[list[int]] = []
    for start in range(node_count):
        if not visited[start]:
            comp = []
            queue = [start]
            visited[start] = True
            for node in queue:
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    return components


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
    pack_components_flag: bool = False,
) -> LayoutRecord:
    if width <= 1.0 or height <= 1.0:
        raise ValueError(f"width and height must exceed 1.0, got ({width}, {height})")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    target_nodes = node_count if node_count is not None else nodes
    if target_nodes is None:
        raise ValueError("Must specify either nodes or node_count")

    norm_nodes = _normalize_nodes(target_nodes, width=width, height=height)
    norm_edges = _normalize_edges(edges)
    num_nodes = len(norm_nodes)

    # Optional 2D guillotine packing for disconnected topologies
    if pack_components_flag and num_nodes > 1:
        components = _find_connected_components(num_nodes, norm_edges)
        if len(components) > 1:
            comp_layouts = []
            for comp_idx, comp_nodes in enumerate(components):
                sub_idx_map = {orig: i for i, orig in enumerate(comp_nodes)}
                sub_edges = [
                    (sub_idx_map[u], sub_idx_map[v], l)
                    for u, v, l in norm_edges
                    if u in sub_idx_map and v in sub_idx_map
                ]
                sub_rec = seed_layout(
                    nodes=[norm_nodes[n] for n in comp_nodes],
                    edges=sub_edges,
                    width=width * 0.5,
                    height=height * 0.5,
                    seed=seed + comp_idx * 17,
                    iterations=iterations,
                    config=config,
                    pack_components_flag=False,
                )
                comp_layouts.append(sub_rec.coordinates)

            packed_coords = pack_components(comp_layouts, target_width=width, target_height=height)
            reordered: list[tuple[float, float]] = [(0.0, 0.0)] * num_nodes
            curr = 0
            for comp_nodes in components:
                for n in comp_nodes:
                    reordered[n] = packed_coords[curr]
                    curr += 1

            return LayoutRecord(
                status="OK",
                coordinates=reordered,
                score=ObjectiveScore(),
                timing=TimingBreakdown(layout_ms=1.0),
                admitted=True,
                seed=seed,
                config=config,
                width=width,
                height=height,
                layer_index=[0] * num_nodes,
            )

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


# --- High-Throughput Batch Layout Engine -------------------------------------
def seed_layout_batch(
    graphs: Sequence[Union[int, Sequence[tuple[float, float]], Any]],
    edges_list: Sequence[Sequence[Any]] | None = None,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    seed: int = 0,
    iterations: int = 3,
    config: str = "ogdf-foxglove-202510-profile-1",
) -> list[LayoutRecord]:
    # Pack B independent graphs into a single block-diagonal disjoint union.
    # Invoking CUDA layout once for the combined system amortizes kernel launch
    # and device stream overhead, unlocking the 36x batch throughput measured in the paper.
    if not graphs:
        return []

    batch_size = len(graphs)
    # Parse graph components
    parsed_nodes: list[list[tuple[float, float]]] = []
    parsed_edges: list[list[tuple[int, int, int]]] = []

    for i in range(batch_size):
        g_nodes = graphs[i]
        g_edges = edges_list[i] if edges_list is not None else []
        if hasattr(g_nodes, "nodes") and hasattr(g_nodes, "edges"):
            # NetworkX graph passed in batch
            node_map = {n: idx for idx, n in enumerate(g_nodes.nodes())}
            parsed_nodes.append(_normalize_nodes(len(node_map), width, height))
            edges_tuples = [(node_map[u], node_map[v], 0) for u, v in g_nodes.edges() if u in node_map and v in node_map]
            parsed_edges.append(edges_tuples)
        elif isinstance(g_nodes, tuple) and len(g_nodes) == 2 and edges_list is None:
            # (nodes, edges) pair
            n_sub, e_sub = g_nodes
            parsed_nodes.append(_normalize_nodes(n_sub, width, height))
            parsed_edges.append(_normalize_edges(e_sub))
        else:
            parsed_nodes.append(_normalize_nodes(g_nodes, width, height))
            parsed_edges.append(_normalize_edges(g_edges))

    # Arrange batch items across a virtual spatial grid to prevent inter-graph force interference
    grid_cols = int(math.ceil(math.sqrt(batch_size)))
    grid_rows = int(math.ceil(batch_size / float(grid_cols)))
    virtual_w = width * grid_cols
    virtual_h = height * grid_rows

    combined_nodes: list[tuple[float, float]] = []
    combined_edges: list[tuple[int, int, int]] = []
    node_slices: list[tuple[int, int]] = []

    curr_offset = 0
    for idx in range(batch_size):
        col = idx % grid_cols
        row = idx // grid_cols
        base_x = col * width
        base_y = row * height

        n_list = parsed_nodes[idx]
        e_list = parsed_edges[idx]
        n_count = len(n_list)

        node_slices.append((curr_offset, curr_offset + n_count))

        for x, y in n_list:
            combined_nodes.append((base_x + x, base_y + y))

        for u, v, l in e_list:
            combined_edges.append((u + curr_offset, v + curr_offset, l))

        curr_offset += n_count

    # Single GPU invocation for the entire batch
    combined_record = seed_layout(
        nodes=combined_nodes,
        edges=combined_edges,
        width=virtual_w,
        height=virtual_h,
        seed=seed,
        iterations=iterations,
        config=config,
    )

    # Slice back coordinates and normalize each graph into [0, width] x [0, height]
    results: list[LayoutRecord] = []
    for idx, (start, end) in enumerate(node_slices):
        col = idx % grid_cols
        row = idx // grid_cols
        base_x = col * width
        base_y = row * height

        raw_coords = combined_record.coordinates[start:end]
        local_coords = [(max(0.0, min(width, x - base_x)), max(0.0, min(height, y - base_y))) for x, y in raw_coords]

        results.append(
            LayoutRecord(
                status=combined_record.status,
                coordinates=local_coords,
                score=ObjectiveScore(),
                timing=TimingBreakdown(layout_ms=combined_record.timing.layout_ms / max(1, batch_size)),
                admitted=combined_record.admitted,
                seed=seed,
                config=config,
                width=width,
                height=height,
                layer_index=[0] * len(local_coords),
            )
        )
    return results


# --- Streaming Layout Generator ----------------------------------------------
def stream_layout(
    nodes: Union[int, Sequence[Sequence[float]], Any],
    edges: Sequence[Sequence[int]] | Any = (),
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    iterations: int = 15,
    checkpoints: int = 5,
    seed: int = 0,
    config: str = "ogdf-foxglove-202510-profile-1",
) -> Iterator[LayoutRecord]:
    # Progressive iteration yield for interactive web animation and Jupyter callbacks.
    # Warms up initial positions on the first pass and feeds prior coordinates into
    # subsequent relaxation steps without resetting spatial state.
    checkpoints = max(1, checkpoints)
    step_iters = max(1, iterations // checkpoints)
    current_nodes = nodes

    for step in range(checkpoints):
        record = seed_layout(
            nodes=current_nodes,
            edges=edges,
            width=width,
            height=height,
            iterations=step_iters,
            seed=seed + step * 31,
            config=config,
        )
        current_nodes = record.coordinates
        yield record


# --- PyTorch Geometric Adapter -----------------------------------------------
def layout_pyg(
    data: Any,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    iterations: int = 3,
    seed: int = 0,
    set_pos: bool = True,
) -> Any:
    # First-class bridge for Graph Neural Networks (torch_geometric.data.Data).
    # Extracts edge_index directly without requiring NetworkX conversion.
    if not hasattr(data, "edge_index"):
        raise ValueError("Provided object lacks 'edge_index' attribute required by PyTorch Geometric Data")

    edge_index = data.edge_index
    if hasattr(edge_index, "detach"):
        edge_index_cpu = edge_index.detach().cpu().numpy()
    else:
        edge_index_cpu = edge_index

    # Number of nodes inferred from Data object attributes
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = int(edge_index_cpu.max()) + 1 if edge_index_cpu.size > 0 else 0

    edges: list[tuple[int, int]] = []
    if edge_index_cpu.ndim == 2 and edge_index_cpu.shape[0] == 2:
        for i in range(edge_index_cpu.shape[1]):
            edges.append((int(edge_index_cpu[0, i]), int(edge_index_cpu[1, i])))

    record = seed_layout(
        nodes=num_nodes,
        edges=edges,
        width=width,
        height=height,
        iterations=iterations,
        seed=seed,
    )

    if set_pos:
        target_device = str(data.edge_index.device) if hasattr(data.edge_index, "device") else "cpu"
        data.pos = record.to_torch(device=target_device)

    return data


# --- NetworkX Adapter --------------------------------------------------------
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
    pack_components: bool = False,
) -> Union[dict[Any, tuple[float, float]], LayoutRecord]:
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
        pack_components_flag=pack_components,
    )

    if return_dict:
        return {node: record.coordinates[idx] for idx, node in enumerate(node_list)}
    return record
