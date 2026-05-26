"""Tests for graph dataset construction in unlabeled inference mode."""

import torch

from conftest import run_cli
from test_tabular_pipeline import write_demo_signals


def test_build_graph_dataset_without_labels(tmp_path):
    signals = tmp_path / "signals.tsv"
    candidates = tmp_path / "candidates.tsv"
    dataset = tmp_path / "dataset.pt"
    write_demo_signals(signals)

    candidates.write_text(
        "candidate_id\tchrom\tstart\tend\tsvtype\tmedian_svlen\tsupport_read_count\tmean_mapq\tstd_pos\tstd_svlen\tread_names\tsignal_indices\n"
        "CAND_000001\tchr1\t110\t171\tDEL\t61\t2\t55.0\t10.0\t1.0\tread1,read2\t0,1\n"
    )

    run_cli("build_graph_dataset.py", "--signals", signals, "--candidates", candidates, "--out", dataset)
    graphs = torch.load(dataset, map_location="cpu", weights_only=False)

    assert len(graphs) == 1
    graph = graphs[0]
    assert tuple(graph.x.shape) == (3, 16)
    assert graph.y.item() == 0
    assert graph.metadata["candidate_id"] == "CAND_000001"
    assert graph.metadata["median_svlen"] == 61
    assert graph.metadata["support_read_count"] == 2
