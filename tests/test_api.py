import pytest
from fmmm_gpu.api import (
    LayoutRecord,
    ObjectiveScore,
    TimingBreakdown,
    seed_layout,
)


class TestDataModels:

    def test_objective_score_defaults_and_fields(self):
        score = ObjectiveScore(
            valid=True,
            exact_valid=True,
            K=3,
            nK=3,
            Phi=0,
            C=17,
            edge_crossings=[1, 2],
        )
        assert score.valid is True
        assert score.K == 3
        assert score.nK == 3
        assert score.C == 17
        assert score.edge_crossings == [1, 2]
        # E is the legacy single-layer crossing alias from the v0 pipeline
        assert score.E == 3.0

        empty_score = ObjectiveScore()
        assert empty_score.K == 0
        assert empty_score.valid is True

    def test_timing_breakdown_fields(self):
        timings = TimingBreakdown(
            layout_ms=10.2,
            device_ms=8.5,
            upload_ms=0.5,
            score_ms=1.1,
            download_ms=0.2,
        )
        assert timings.layout_ms == 10.2
        assert timings.device_ms == 8.5
        assert timings.total_ms > 10.0

    def test_layout_record_mapping_protocol_backward_compatibility(self):
        # Downstream visualization code still indexes into result['coordinates']
        record = LayoutRecord(
            status="OK",
            coordinates=[(0.0, 0.0), (100.0, 100.0)],
            score=ObjectiveScore(K=0),
            timing=TimingBreakdown(layout_ms=5.0),
            admitted=True,
            seed=42,
            config="test-config",
            width=500.0,
            height=500.0,
            layer_index=[0, 1],
        )

        assert record.status == "OK"
        assert record.width == 500.0
        assert record.height == 500.0
        assert len(record.coordinates) == 2
        assert record.score.K == 0
        assert record.admitted is True

        assert record["status"] == "OK"
        assert record["width"] == 500.0
        assert record["coordinates"] == [(0.0, 0.0), (100.0, 100.0)]
        assert "coordinates" in record
        assert "score" in record
        assert "nonexistent" not in record

        keys = list(record.keys())
        assert "status" in keys
        assert "coordinates" in keys
        assert "score" in keys
        assert len(record) == len(keys)
        assert record.get("status") == "OK"
        assert record.get("nonexistent", "fallback") == "fallback"

        as_dict = dict(record)
        assert isinstance(as_dict, dict)
        assert as_dict["status"] == "OK"


class TestSeedLayoutAPI:

    def test_seed_layout_with_node_count_and_2tuple_edges(self):
        node_count = 4
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        res = seed_layout(nodes=node_count, edges=edges, width=800.0, height=600.0)

        assert isinstance(res, LayoutRecord)
        assert res.status == "OK"
        assert len(res.coordinates) == node_count
        assert res.width == 800.0
        assert res.height == 600.0
        assert len(res.layer_index) == node_count
        assert all(l == 0 for l in res.layer_index)

    def test_seed_layout_with_explicit_coordinates_and_3tuples(self):
        initial_coords = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)]
        edges = [(0, 1, 0), (1, 2, 2)]
        res = seed_layout(nodes=initial_coords, edges=edges, width=1000.0, height=1000.0)

        assert len(res.coordinates) == 3
        assert res.layer_index[2] >= 2

    def test_seed_layout_input_validation(self):
        with pytest.raises(ValueError, match="node_count must be positive"):
            seed_layout(nodes=0, edges=[])

        with pytest.raises(ValueError, match="width and height must exceed 1.0"):
            seed_layout(nodes=5, edges=[], width=0.5, height=500.0)

        with pytest.raises(ValueError, match="width and height must exceed 1.0"):
            seed_layout(nodes=5, edges=[], width=500.0, height=-10.0)

        with pytest.raises(ValueError, match="iterations must be positive"):
            seed_layout(nodes=5, edges=[], iterations=0)

        with pytest.raises(ValueError, match="Edge must be a 2-tuple or 3-tuple"):
            seed_layout(nodes=5, edges=[(0,)])  # type: ignore

    def test_tensor_conversion_helpers(self):
        record = LayoutRecord(
            status="OK",
            coordinates=[(10.0, 20.0), (30.0, 40.0)],
            score=ObjectiveScore(),
            timing=TimingBreakdown(),
            admitted=True,
            seed=0,
            config="test",
            width=500.0,
            height=500.0,
        )

        try:
            import numpy as np
            arr = record.to_numpy()
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (2, 2)
            assert arr[0, 0] == 10.0
            assert arr[1, 1] == 40.0
        except ImportError:
            with pytest.raises(ImportError, match="NumPy is required"):
                record.to_numpy()

        try:
            import torch
            tensor = record.to_torch()
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (2, 2)
        except ImportError:
            with pytest.raises(ImportError, match="PyTorch is required"):
                record.to_torch()
