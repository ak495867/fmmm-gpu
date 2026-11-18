import pytest
from fmmm_gpu.api import stream_layout, LayoutRecord


def test_stream_layout_yields_expected_checkpoints(cycle_graph):
    n, edges = cycle_graph(10)
    snapshots = list(stream_layout(
        nodes=n,
        edges=edges,
        width=600.0,
        height=600.0,
        iterations=15,
        checkpoints=3,
        seed=42,
    ))

    assert len(snapshots) == 3
    for rec in snapshots:
        assert isinstance(rec, LayoutRecord)
        assert len(rec.coordinates) == n
        for x, y in rec.coordinates:
            assert 0.0 <= x <= 600.0
            assert 0.0 <= y <= 600.0


def test_stream_layout_progressive_updates(path_graph):
    n, edges = path_graph(8)
    iterator = stream_layout(nodes=n, edges=edges, checkpoints=2)

    first = next(iterator)
    second = next(iterator)

    assert len(first.coordinates) == n
    assert len(second.coordinates) == n
