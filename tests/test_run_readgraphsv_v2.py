"""Tests for the ReadGraphSV v0.2 one-click pipeline wrapper."""

import pandas as pd
import torch

from build_graph_dataset_v2 import FEATURE_NAMES_V2
from conftest import run_cli
from run_readgraphsv_v2 import parse_args
from utils_graph import ReadGraphSAGE


def make_sequence(length):
    return "A" * length


def write_demo_sam(path):
    path.write_text(
        "@HD\tVN:1.6\tSO:unknown\n"
        "@SQ\tSN:chr1\tLN:100000\n"
        "read1\t0\tchr1\t101\t60\t60S100M100D100M\t*\t0\t0\t"
        f"{make_sequence(260)}\t*\tSA:Z:chr1,201,+,100M,55,2;\tNM:i:1\n"
    )


def write_truth_vcf(path):
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t201\ttruth_del\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=300;SVLEN=-100\n"
    )


def write_dummy_model(path):
    model = ReadGraphSAGE(in_channels=len(FEATURE_NAMES_V2), hidden_channels=8)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": len(FEATURE_NAMES_V2),
            "hidden_channels": 8,
            "feature_names": FEATURE_NAMES_V2,
        },
        path,
    )


def assert_common_outputs(outdir):
    expected = [
        "data/signals.tsv",
        "data/candidates.tsv",
        "data/extra_signals.tsv",
        "data/candidates_for_graph.tsv",
        "graphs/dataset_v2.pt",
        "results/predictions_v2.tsv",
        "results/filtered_candidates.tsv",
        "vcf/filtered.vcf",
    ]
    for relative_path in expected:
        assert (outdir / relative_path).exists(), relative_path


def test_parse_args_for_v2_pipeline():
    args = parse_args(
        [
            "--bam",
            "input.bam",
            "--model",
            "model.pt",
            "--outdir",
            "runs/demo",
            "--threshold",
            "0.25",
            "--cluster_window",
            "750",
            "--extra_window",
            "1200",
        ]
    )
    assert args.bam == "input.bam"
    assert args.model == "model.pt"
    assert args.outdir == "runs/demo"
    assert args.threshold == 0.25
    assert args.cluster_window == 750
    assert args.extra_window == 1200
    assert args.min_support == 1
    assert args.min_size_sim == 0.5


def test_run_readgraphsv_v2_without_truth_generates_candidates_for_graph(tmp_path):
    sam = tmp_path / "demo.sam"
    model = tmp_path / "model.pt"
    outdir = tmp_path / "run_no_truth"
    write_demo_sam(sam)
    write_dummy_model(model)

    run_cli(
        "run_readgraphsv_v2.py",
        "--bam",
        sam,
        "--model",
        model,
        "--outdir",
        outdir,
        "--threshold",
        "0.0",
        "--min_support",
        "1",
    )

    assert_common_outputs(outdir)
    candidates_for_graph = pd.read_csv(outdir / "data/candidates_for_graph.tsv", sep="\t")
    assert "label" in candidates_for_graph.columns
    assert set(candidates_for_graph["label"].astype(int)) == {0}

    filtered = pd.read_csv(outdir / "results/filtered_candidates.tsv", sep="\t")
    assert len(filtered) == 1
    vcf_text = (outdir / "vcf/filtered.vcf").read_text()
    assert "##source=ReadGraphSV_v0.2" in vcf_text
    assert '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">' in vcf_text
    assert "SVTYPE=DEL" in vcf_text
    assert "\tGT\t./." in vcf_text


def test_run_readgraphsv_v2_with_truth_calls_label_flow(tmp_path):
    sam = tmp_path / "demo.sam"
    truth = tmp_path / "truth.vcf"
    model = tmp_path / "model.pt"
    outdir = tmp_path / "run_with_truth"
    write_demo_sam(sam)
    write_truth_vcf(truth)
    write_dummy_model(model)

    run_cli(
        "run_readgraphsv_v2.py",
        "--bam",
        sam,
        "--model",
        model,
        "--truth",
        truth,
        "--outdir",
        outdir,
        "--threshold",
        "0.0",
        "--min_support",
        "1",
        "--max_dist",
        "500",
        "--min_size_sim",
        "0.5",
    )

    assert_common_outputs(outdir)
    assert (outdir / "data/candidates_labeled.tsv").exists()
    assert (outdir / "results/evaluation_v2.txt").exists()

    labeled = pd.read_csv(outdir / "data/candidates_labeled.tsv", sep="\t")
    candidates_for_graph = pd.read_csv(outdir / "data/candidates_for_graph.tsv", sep="\t")
    assert int(labeled["label"].iloc[0]) == 1
    assert int(candidates_for_graph["label"].iloc[0]) == 1
    evaluation_text = (outdir / "results/evaluation_v2.txt").read_text()
    assert "Raw Precision" in evaluation_text
