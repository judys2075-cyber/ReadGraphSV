"""Tests for the ReadGraphSV v0.2 one-click pipeline wrapper."""

from pathlib import Path

import pandas as pd
import torch

import run_readgraphsv_v2
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
    assert args.contig_length is None
    assert args.use_extra_candidates is False
    assert args.extra_candidate_window == 500
    assert args.min_softclip_support == 10
    assert args.min_sa_support == 2
    assert args.min_supplementary_support == 2
    assert args.min_extra_only_support == 30
    assert args.extra_candidate_min_size == args.min_size
    assert args.use_dedup is False
    assert args.dedup_window == 500
    assert args.dedup_min_size_sim == 0.5
    assert args.dedup_score_col == "gnn_prob"


def test_parse_args_for_v3_extra_candidate_options():
    args = parse_args(
        [
            "--bam",
            "input.bam",
            "--model",
            "model.pt",
            "--outdir",
            "runs/demo",
            "--min_size",
            "80",
            "--use_extra_candidates",
            "--extra_candidate_window",
            "900",
            "--min_softclip_support",
            "12",
            "--min_sa_support",
            "3",
            "--min_supplementary_support",
            "4",
            "--min_extra_only_support",
            "40",
            "--extra_candidate_min_size",
            "100",
            "--use_dedup",
            "--dedup_window",
            "750",
            "--dedup_min_size_sim",
            "0.7",
            "--dedup_score_col",
            "custom_score",
        ]
    )
    assert args.use_extra_candidates is True
    assert args.extra_candidate_window == 900
    assert args.min_softclip_support == 12
    assert args.min_sa_support == 3
    assert args.min_supplementary_support == 4
    assert args.min_extra_only_support == 40
    assert args.extra_candidate_min_size == 100
    assert args.use_dedup is True
    assert args.dedup_window == 750
    assert args.dedup_min_size_sim == 0.7
    assert args.dedup_score_col == "custom_score"


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
    assert not (outdir / "data/extra_candidates.tsv").exists()
    assert not (outdir / "data/candidates_v3_merged.tsv").exists()
    assert not (outdir / "results/filtered_candidates_dedup.tsv").exists()
    assert not (outdir / "vcf/filtered_dedup.vcf").exists()
    candidates_for_graph = pd.read_csv(outdir / "data/candidates_for_graph.tsv", sep="\t")
    assert "label" in candidates_for_graph.columns
    assert set(candidates_for_graph["label"].astype(int)) == {0}

    filtered = pd.read_csv(outdir / "results/filtered_candidates.tsv", sep="\t")
    assert len(filtered) == 1
    vcf_text = (outdir / "vcf/filtered.vcf").read_text()
    assert "##source=ReadGraphSV_v0.2" in vcf_text
    assert "##contig=<ID=chr1,length=100000>" in vcf_text
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


def minimal_candidates_row(candidate_source="CIGAR_ONLY"):
    return {
        "candidate_id": "CAND_000001",
        "chrom": "chr1",
        "start": 200,
        "end": 300,
        "svtype": "DEL",
        "median_svlen": 100,
        "support_read_count": 2,
        "mean_mapq": 60,
        "std_pos": 0,
        "std_svlen": 0,
        "read_names": "read1,read2",
        "signal_indices": "0,1",
        "candidate_source": candidate_source,
        "extra_support": 0,
        "matched_extra_count": 0,
        "extra_source": "",
    }


def write_minimal_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def install_mocked_v2_pipeline(monkeypatch, commands, vcf_calls=None):
    def fake_run_step(_name, command):
        commands.append(command)
        script = command[1].split("/")[-1]
        out = command[command.index("--out") + 1] if "--out" in command else None
        if script == "cluster_events.py":
            write_minimal_tsv(Path(out), [minimal_candidates_row()])
        elif script == "extract_extra_events.py":
            write_minimal_tsv(Path(out), [{"read_name": "read1", "evidence_type": "SOFTCLIP_LEFT"}])
        elif script == "extra_candidate_proposer.py":
            write_minimal_tsv(
                Path(out),
                [
                    {
                        "chrom": "chr1",
                        "start": 500,
                        "end": 501,
                        "svtype": "INS",
                        "svlen": 80,
                        "median_svlen": 80,
                        "support": 30,
                        "source": "SOFTCLIP",
                        "mean_pos": 500,
                        "mean_size": 80,
                    }
                ],
            )
        elif script == "merge_candidates_v3.py":
            write_minimal_tsv(
                Path(out),
                [
                    minimal_candidates_row(),
                    {
                        **minimal_candidates_row(candidate_source="EXTRA_ONLY"),
                        "candidate_id": "EXTRA_000001",
                        "start": 500,
                        "end": 501,
                        "svtype": "INS",
                        "median_svlen": 80,
                        "support_read_count": 30,
                        "extra_support": 30,
                        "extra_source": "SOFTCLIP",
                    },
                ],
            )
        elif script == "dedup_filtered_candidates.py":
            write_minimal_tsv(
                Path(out),
                [
                    {
                        "candidate_id": "CAND_000001",
                        "chrom": "chr1",
                        "start": 200,
                        "end": 300,
                        "svtype": "DEL",
                        "median_svlen": 100,
                        "support_read_count": 2,
                        "label": 0,
                        "gnn_prob": 0.9,
                        "gnn_pred": 1,
                        "candidate_source": "CIGAR_ONLY",
                    }
                ],
            )
            (Path(out).parent / "dedup_summary.txt").write_text(
                "Input candidates: 2\nOutput candidates: 1\nRemoved candidates: 1\nRemoved by svtype:\nDEL\t1\n"
            )
        elif out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("")

    def fake_filter_predictions(_predictions_path, out_path, _threshold):
        write_minimal_tsv(
            Path(out_path),
            [
                {
                    "candidate_id": "CAND_000001",
                    "chrom": "chr1",
                    "start": 200,
                    "end": 300,
                    "svtype": "DEL",
                    "median_svlen": 100,
                    "support_read_count": 2,
                    "label": 0,
                    "gnn_prob": 0.9,
                    "gnn_pred": 1,
                    "candidate_source": "CIGAR_ONLY",
                },
                {
                    "candidate_id": "CAND_000002",
                    "chrom": "chr1",
                    "start": 220,
                    "end": 320,
                    "svtype": "DEL",
                    "median_svlen": 100,
                    "support_read_count": 1,
                    "label": 0,
                    "gnn_prob": 0.8,
                    "gnn_pred": 1,
                    "candidate_source": "CIGAR_ONLY",
                },
            ],
        )
        return 2

    def fake_write_filtered_vcf(*args, **kwargs):
        if vcf_calls is not None:
            vcf_calls.append((args, kwargs))
        return 1

    monkeypatch.setattr(run_readgraphsv_v2, "check_inputs", lambda _args: None)
    monkeypatch.setattr(run_readgraphsv_v2, "run_step", fake_run_step)
    monkeypatch.setattr(run_readgraphsv_v2, "filter_predictions", fake_filter_predictions)
    monkeypatch.setattr(run_readgraphsv_v2, "write_filtered_vcf", fake_write_filtered_vcf)


def command_script_names(commands):
    return [command[1].split("/")[-1] for command in commands]


def test_default_pipeline_does_not_call_v3_candidate_tools(monkeypatch, tmp_path):
    commands = []
    vcf_calls = []
    install_mocked_v2_pipeline(monkeypatch, commands, vcf_calls=vcf_calls)
    outdir = tmp_path / "default_pipeline"

    run_readgraphsv_v2.main(["--bam", "input.bam", "--model", "model.pt", "--outdir", str(outdir)])

    script_names = command_script_names(commands)
    assert "extra_candidate_proposer.py" not in script_names
    assert "merge_candidates_v3.py" not in script_names
    assert "dedup_filtered_candidates.py" not in script_names
    assert vcf_calls[0][0][0] == str(outdir / "results/filtered_candidates.tsv")
    assert vcf_calls[0][0][1] == str(outdir / "vcf/filtered.vcf")
    candidates_for_graph = pd.read_csv(outdir / "data/candidates_for_graph.tsv", sep="\t")
    assert set(candidates_for_graph["candidate_source"]) == {"CIGAR_ONLY"}


def test_use_extra_candidates_calls_v3_tools_and_uses_merged_candidates(monkeypatch, tmp_path):
    commands = []
    install_mocked_v2_pipeline(monkeypatch, commands)
    outdir = tmp_path / "v3_pipeline"

    run_readgraphsv_v2.main(
        [
            "--bam",
            "input.bam",
            "--model",
            "model.pt",
            "--outdir",
            str(outdir),
            "--use_extra_candidates",
            "--min_extra_only_support",
            "30",
        ]
    )

    script_names = command_script_names(commands)
    assert "extra_candidate_proposer.py" in script_names
    assert "merge_candidates_v3.py" in script_names
    assert (outdir / "data/extra_candidates.tsv").exists()
    assert (outdir / "data/candidates_v3_merged.tsv").exists()

    candidates_for_graph = pd.read_csv(outdir / "data/candidates_for_graph.tsv", sep="\t")
    assert "EXTRA_ONLY" in set(candidates_for_graph["candidate_source"])
    assert "label" in candidates_for_graph.columns


def test_use_dedup_calls_dedup_and_exports_dedup_vcf(monkeypatch, tmp_path):
    commands = []
    vcf_calls = []
    install_mocked_v2_pipeline(monkeypatch, commands, vcf_calls=vcf_calls)
    outdir = tmp_path / "dedup_pipeline"

    run_readgraphsv_v2.main(
        [
            "--bam",
            "input.bam",
            "--model",
            "model.pt",
            "--outdir",
            str(outdir),
            "--use_dedup",
            "--dedup_window",
            "500",
            "--dedup_min_size_sim",
            "0.5",
        ]
    )

    script_names = command_script_names(commands)
    assert "dedup_filtered_candidates.py" in script_names
    assert (outdir / "results/filtered_candidates_dedup.tsv").exists()
    assert (outdir / "results/dedup_summary.txt").exists()
    assert vcf_calls[0][0][0] == str(outdir / "results/filtered_candidates_dedup.tsv")
    assert vcf_calls[0][0][1] == str(outdir / "vcf/filtered_dedup.vcf")
