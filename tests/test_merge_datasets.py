"""Tests for merging PyTorch Geometric dataset files."""

import torch
from torch_geometric.data import Data

from conftest import run_cli


def test_merge_datasets_preserves_data_attributes(tmp_path):
    ds1 = tmp_path / "dataset1.pt"
    ds2 = tmp_path / "dataset2.pt"
    merged = tmp_path / "merged.pt"

    graph_a = Data(
        x=torch.ones((1, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([1.0]),
    )
    graph_a.metadata = {"candidate_id": "A"}

    graph_b = Data(
        x=torch.zeros((1, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.tensor([0.0]),
    )
    graph_b.metadata = {"candidate_id": "B"}

    torch.save([graph_a], ds1)
    torch.save([graph_b], ds2)

    run_cli("merge_datasets.py", "--inputs", ds1, ds2, "--out", merged)
    graphs = torch.load(merged, map_location="cpu", weights_only=False)

    assert len(graphs) == 2
    assert graphs[0].metadata["candidate_id"] == "A"
    assert graphs[1].metadata["candidate_id"] == "B"
    assert graphs[0].y.item() == 1.0
