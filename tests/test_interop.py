import pytest
from fmmm_gpu.api import layout_networkx, LayoutRecord


def _has_networkx():
    try:
        import networkx
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_networkx(), reason="NetworkX is not installed in the test environment")
class TestNetworkXInterop:

    def test_basic_graph_layout_as_dict(self):
        import networkx as nx

        G = nx.cycle_graph(6)
        pos = layout_networkx(G, width=600.0, height=400.0, return_dict=True)

        assert isinstance(pos, dict)
        assert len(pos) == 6
        for node in G.nodes():
            assert node in pos
            x, y = pos[node]
            assert 0.0 <= x <= 600.0
            assert 0.0 <= y <= 400.0

    def test_string_node_identifiers(self):
        import networkx as nx

        G = nx.Graph()
        G.add_edge("source_db", "api_gateway")
        G.add_edge("api_gateway", "cache_redis")
        G.add_edge("api_gateway", "auth_service")

        pos = layout_networkx(G, width=1000.0, height=800.0, return_dict=True)

        assert len(pos) == 4
        assert "source_db" in pos
        assert "api_gateway" in pos
        assert "cache_redis" in pos
        assert "auth_service" in pos

        for name, coord in pos.items():
            assert len(coord) == 2
            assert all(isinstance(c, (float, int)) for c in coord)

    def test_return_full_layout_record(self):
        import networkx as nx

        G = nx.path_graph(5)
        record = layout_networkx(G, return_dict=False)

        assert isinstance(record, LayoutRecord)
        assert record.status in ("OK", "converged")
        assert len(record.coordinates) == 5
        assert record.score.E >= 0.0

    def test_directed_graph_and_multigraph(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_edges_from([(10, 20), (20, 30), (30, 10), (30, 40)])
        pos = layout_networkx(G, return_dict=True)

        assert len(pos) == 4
        assert 10 in pos
        assert 40 in pos

    def test_edge_layer_attributes(self):
        import networkx as nx

        G = nx.Graph()
        G.add_edge("A", "B", layer=0)
        G.add_edge("B", "C", layer=3)

        record = layout_networkx(G, return_dict=False)
        assert len(record.layer_index) == 3
        assert max(record.layer_index) >= 3

    def test_empty_graph_handling(self):
        import networkx as nx

        G = nx.Graph()
        with pytest.raises(ValueError, match="Cannot lay out an empty NetworkX graph"):
            layout_networkx(G)
