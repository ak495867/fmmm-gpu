import pytest
from fmmm_gpu.api import seed_layout


class TestLayoutCorpus:

    def test_seed_determinism(self, cycle_graph):
        n, edges = cycle_graph(12)
        run_1 = seed_layout(node_count=n, edges=edges, seed=42)
        run_2 = seed_layout(node_count=n, edges=edges, seed=42)

        assert run_1.coordinates == run_2.coordinates
        assert run_1.score.E == run_2.score.E
        assert run_1.score.K == run_2.score.K

    def test_seed_sensitivity(self, cycle_graph):
        n, edges = cycle_graph(12)
        run_a = seed_layout(node_count=n, edges=edges, seed=123)
        run_b = seed_layout(node_count=n, edges=edges, seed=987)

        assert run_a.coordinates != run_b.coordinates

    def test_disconnected_components(self, disconnected_components_graph):
        n, edges = disconnected_components_graph
        layout = seed_layout(node_count=n, edges=edges, width=1200.0, height=800.0)

        assert len(layout.coordinates) == n
        assert layout.status in ("OK", "converged")

        for x, y in layout.coordinates:
            assert 0.0 <= x <= 1200.0
            assert 0.0 <= y <= 800.0

    def test_isolated_nodes(self):
        node_count = 6
        edges = [(0, 1), (1, 2)]
        layout = seed_layout(node_count=node_count, edges=edges, width=600.0, height=600.0)

        assert len(layout.coordinates) == node_count
        for x, y in layout.coordinates:
            assert 0.0 <= x <= 600.0
            assert 0.0 <= y <= 600.0

    def test_dense_clique(self):
        n = 10
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        layout = seed_layout(node_count=n, edges=edges, iterations=100)

        assert len(layout.coordinates) == n
        assert layout.score.E >= 0.0
