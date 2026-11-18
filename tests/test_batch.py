import pytest
from fmmm_gpu.api import seed_layout_batch, LayoutRecord


def test_seed_layout_batch_multiple_graphs(cycle_graph, path_graph):
    n1, e1 = cycle_graph(8)
    n2, e2 = path_graph(6)
    n3, e3 = cycle_graph(5)

    batch_inputs = [(n1, e1), (n2, e2), (n3, e3)]
    results = seed_layout_batch(batch_inputs, width=800.0, height=800.0)

    assert len(results) == 3
    assert all(isinstance(r, LayoutRecord) for r in results)
    assert len(results[0].coordinates) == n1
    assert len(results[1].coordinates) == n2
    assert len(results[2].coordinates) == n3

    for r in results:
        for x, y in r.coordinates:
            assert 0.0 <= x <= 800.0
            assert 0.0 <= y <= 800.0


def test_seed_layout_batch_empty():
    assert seed_layout_batch([]) == []


def test_seed_layout_batch_single_graph(star_graph):
    n, e = star_graph(10)
    results = seed_layout_batch([(n, e)], width=500.0, height=500.0)

    assert len(results) == 1
    assert len(results[0].coordinates) == n
