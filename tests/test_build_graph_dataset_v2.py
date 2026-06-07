"""Tests for v0.2 graph construction with extra evidence nodes."""

import torch

from build_graph_dataset_v2 import FEATURE_NAMES_V2
from conftest import run_cli


EXTRA_HEADER = (
    "read_name\tevidence_type\tsrc_chrom\tsrc_start\tsrc_end\tsrc_strand\t"
    "dst_chrom\tdst_start\tdst_end\tdst_strand\tevent_pos\tevent_len\t"
    "mapq\tdst_mapq\tchrom_change\torientation_change\tis_supplementary\t"
    "has_sa\tsoftclip_left\tsoftclip_right\tcigar\tsa_cigar\tnm\tsource\n"
)


def write_signals(path):
    path.write_text(
        "read_name\tchrom\tevent_pos\tevent_end\tsvtype\tsvlen\tmapq\tstrand\tread_len\t"
        "read_event_start\tread_event_end\tis_supplementary\thas_sa\tsoftclip_left\tsoftclip_right\tcigar\n"
        "read1\tchr1\t120\t220\tDEL\t100\t60\t+\t1000\t100\t100\t0\t1\t60\t0\t60S100M100D740M\n"
        "read2\tchr1\t160\t260\tDEL\t100\t50\t-\t1200\t120\t120\t1\t1\t0\t70\t160M100D870M70S\n"
    )


def write_candidates_with_labels(path):
    path.write_text(
        "candidate_id\tchrom\tstart\tend\tsvtype\tmedian_svlen\tsupport_read_count\t"
        "mean_mapq\tstd_pos\tstd_svlen\tread_names\tsignal_indices\tlabel\n"
        "CAND_000001\tchr1\t100\t200\tDEL\t100\t2\t55.0\t20.0\t0.0\tread1,read2\t0,1\t1\n"
        "CAND_000002\tchr2\t9000\t9100\tDEL\t100\t0\t0.0\t0.0\t0.0\t\t\t0\n"
    )


def write_candidates_without_labels(path):
    path.write_text(
        "candidate_id\tchrom\tstart\tend\tsvtype\tmedian_svlen\tsupport_read_count\t"
        "mean_mapq\tstd_pos\tstd_svlen\tread_names\tsignal_indices\n"
        "CAND_000001\tchr1\t100\t200\tDEL\t100\t2\t55.0\t20.0\t0.0\tread1,read2\t0,1\n"
    )


def write_extra(path):
    path.write_text(
        EXTRA_HEADER
        + "read3\tSOFTCLIP_LEFT\tchr1\t90\t190\t+\t\t\t\t\t100\t60\t42\t\t0\t0\t0\t1\t60\t0\t60S100M\t\t2\tCIGAR_SOFTCLIP\n"
        + "read4\tSOFTCLIP_RIGHT\tchr1\t100\t200\t+\t\t\t\t\t200\t70\t43\t\t0\t0\t0\t0\t0\t70\t100M70S\t\t1\tCIGAR_SOFTCLIP\n"
        + "read5\tSA_CONNECTION\tchr2\t5000\t5080\t+\tchr1\t180\t260\t-\t5000\t0\t30\t55\t1\t1\t0\t1\t0\t0\t80M\t80M\t3\tSA_TAG\n"
        + "read6\tSUPPLEMENTARY\tchr3\t100000\t100100\t+\t\t\t\t\t100000\t100\t20\t\t0\t0\t1\t0\t0\t0\t100M\t\t0\tBAM_SUPPLEMENTARY_FLAG\n"
    )


def test_build_graph_dataset_v2_with_extra_evidence(tmp_path):
    signals = tmp_path / "signals.tsv"
    candidates = tmp_path / "candidates_labeled.tsv"
    extra = tmp_path / "extra_signals.tsv"
    dataset = tmp_path / "dataset_v2.pt"

    write_signals(signals)
    write_candidates_with_labels(candidates)
    write_extra(extra)

    run_cli(
        "build_graph_dataset_v2.py",
        "--signals",
        signals,
        "--extra",
        extra,
        "--candidates",
        candidates,
        "--out",
        dataset,
        "--extra_window",
        "100",
        "--read_edge_window",
        "100",
    )

    graphs = torch.load(dataset, map_location="cpu", weights_only=False)
    assert len(graphs) == 2
    assert all(graph.x.shape[1] == len(FEATURE_NAMES_V2) for graph in graphs)

    graph = graphs[0]
    assert graph.y.item() == 1
    assert tuple(graph.x.shape) == (6, len(FEATURE_NAMES_V2))
    assert graph.metadata["candidate_id"] == "CAND_000001"
    assert graph.metadata["num_cigar_nodes"] == 2
    assert graph.metadata["num_extra_nodes"] == 3
    assert graph.metadata["support_read_count"] == 2
    assert graph.metadata["median_svlen"] == 100
    assert graph.feature_names == FEATURE_NAMES_V2

    graph_without_extra = graphs[1]
    assert graph_without_extra.metadata["num_cigar_nodes"] == 0
    assert graph_without_extra.metadata["num_extra_nodes"] == 0
    assert tuple(graph_without_extra.x.shape) == (1, len(FEATURE_NAMES_V2))


def test_build_graph_dataset_v2_without_extra_evidence(tmp_path):
    signals = tmp_path / "signals.tsv"
    candidates = tmp_path / "candidates.tsv"
    extra = tmp_path / "empty_extra_signals.tsv"
    dataset = tmp_path / "dataset_v2.pt"

    write_signals(signals)
    write_candidates_without_labels(candidates)
    extra.write_text(EXTRA_HEADER)

    run_cli(
        "build_graph_dataset_v2.py",
        "--signals",
        signals,
        "--extra",
        extra,
        "--candidates",
        candidates,
        "--out",
        dataset,
    )

    graphs = torch.load(dataset, map_location="cpu", weights_only=False)
    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.y.item() == 0
    assert graph.metadata["num_cigar_nodes"] == 2
    assert graph.metadata["num_extra_nodes"] == 0
    assert graph.x.shape[1] == len(FEATURE_NAMES_V2)
    assert all(node.shape[0] == len(FEATURE_NAMES_V2) for node in graph.x)
