import math
import pytest
from fmmm_gpu.api import seed_layout


class TestGeometricInvariants:

    @pytest.mark.parametrize("n", [1, 2, 5, 20, 50])
    def test_finite_coordinate_invariant(self, n, path_graph):
        node_count, edges = path_graph(n)
        layout = seed_layout(
            node_count=node_count,
            edges=edges,
            width=500.0,
            height=500.0,
            iterations=50,
        )

        assert len(layout.coordinates) == node_count
        for i, (x, y) in enumerate(layout.coordinates):
            assert not math.isnan(x), f"Node {i} x-coordinate is NaN"
            assert not math.isnan(y), f"Node {i} y-coordinate is NaN"
            assert not math.isinf(x), f"Node {i} x-coordinate is infinite"
            assert not math.isinf(y), f"Node {i} y-coordinate is infinite"

    @pytest.mark.parametrize("width,height", [
        (100.0, 100.0),
        (1920.0, 1080.0),
        (50.0, 500.0),
    ])
    def test_bounding_box_canvas_containment(self, width, height, cycle_graph):
        node_count, edges = cycle_graph(15)
        layout = seed_layout(
            node_count=node_count,
            edges=edges,
            width=width,
            height=height,
            iterations=50,
        )

        for i, (x, y) in enumerate(layout.coordinates):
            assert 0.0 <= x <= width, f"Node {i} x={x} exceeded canvas width [0, {width}]"
            assert 0.0 <= y <= height, f"Node {i} y={y} exceeded canvas height [0, {height}]"

    def test_cardinality_and_indexing_invariant(self, star_graph):
        n, edges = star_graph(25)
        layout = seed_layout(node_count=n, edges=edges, width=800.0, height=800.0)

        assert len(layout.coordinates) == n
        assert len(layout.layer_index) == n
        assert layout.width == 800.0
        assert layout.height == 800.0

    def test_objective_score_non_negativity(self, cycle_graph):
        n, edges = cycle_graph(10)
        layout = seed_layout(node_count=n, edges=edges)

        assert layout.score.E >= 0.0, "Energy cannot be negative"
        assert layout.score.K >= 0.0, "Crossing count cannot be negative"
        assert layout.score.O >= 0.0, "Total objective cannot be negative"

    def test_non_degeneracy_invariant(self, cycle_graph):
        n, edges = cycle_graph(8)
        layout = seed_layout(node_count=n, edges=edges, width=1000.0, height=1000.0)

        xs = [coord[0] for coord in layout.coordinates]
        ys = [coord[1] for coord in layout.coordinates]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        # Force equilibrium should produce non-zero canvas span for multi-node graphs
        assert dx > 1.0, "Layout collapsed into a vertical line or point"
        assert dy > 1.0, "Layout collapsed into a horizontal line or point"
