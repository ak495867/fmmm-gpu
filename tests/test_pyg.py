import pytest
from fmmm_gpu.api import layout_pyg


class MockPyGData:

    def __init__(self, edge_index, num_nodes=None):
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.pos = None


def test_layout_pyg_numpy_mock():
    try:
        import numpy as np
    except ImportError:
        pytest.skip("NumPy is required to test PyG adapter")

    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    data = MockPyGData(edge_index=edge_index, num_nodes=4)

    out = layout_pyg(data, width=500.0, height=500.0, set_pos=True)

    assert out is data
    assert out.pos is not None
    assert len(out.pos) == 4


def test_layout_pyg_missing_attribute():
    with pytest.raises(ValueError, match="lacks 'edge_index' attribute"):
        layout_pyg(object())
