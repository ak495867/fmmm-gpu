import pytest
from fmmm_gpu.api import pack_components, seed_layout


def test_pack_components_empty_and_single():
    assert pack_components([]) == []
    single = [(10.0, 20.0), (30.0, 40.0)]
    assert pack_components([single]) == single


def test_pack_components_multiple_subgraphs():
    comp_a = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)]
    comp_b = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]
    comp_c = [(0.0, 0.0), (80.0, 80.0)]

    packed = pack_components(
        [comp_a, comp_b, comp_c],
        target_width=800.0,
        target_height=600.0,
        padding=20.0,
    )

    total_nodes = len(comp_a) + len(comp_b) + len(comp_c)
    assert len(packed) == total_nodes

    for x, y in packed:
        assert 0.0 <= x <= 800.0
        assert 0.0 <= y <= 600.0


def test_seed_layout_with_pack_components_flag(disconnected_components_graph):
    n, edges = disconnected_components_graph
    layout = seed_layout(
        nodes=n,
        edges=edges,
        width=1000.0,
        height=800.0,
        pack_components_flag=True,
    )

    assert len(layout.coordinates) == n
    assert layout.status in ("OK", "converged")
    for x, y in layout.coordinates:
        assert 0.0 <= x <= 1000.0
        assert 0.0 <= y <= 800.0
