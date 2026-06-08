"""Tests for candidate clustering, labeling, VCF export, and evaluation."""

import pandas as pd

from conftest import run_cli


def write_demo_signals(path):
    path.write_text(
        "\t".join(
            [
                "read_name",
                "chrom",
                "event_pos",
                "event_end",
                "svtype",
                "svlen",
                "mapq",
                "strand",
                "read_len",
                "read_event_start",
                "read_event_end",
                "is_supplementary",
                "has_sa",
                "softclip_left",
                "softclip_right",
                "cigar",
            ]
        )
        + "\n"
        + "read1\tchr1\t100\t160\tDEL\t60\t60\t+\t1000\t100\t100\t0\t0\t0\t0\t100M60D840M\n"
        + "read2\tchr1\t120\t182\tDEL\t62\t50\t-\t1100\t120\t120\t0\t1\t5\t0\t120M62D918M\n"
        + "read3\tchr1\t1000\t1001\tINS\t80\t55\t+\t900\t400\t480\t1\t1\t0\t20\t400M80I420M\n"
    )


def write_demo_truth(path):
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t101\ttruth_del\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=160;SVLEN=-60\n"
    )


def test_cluster_label_export_and_evaluate(tmp_path):
    signals = tmp_path / "signals.tsv"
    candidates = tmp_path / "candidates.tsv"
    labeled = tmp_path / "candidates_labeled.tsv"
    truth = tmp_path / "truth.vcf"
    pred = tmp_path / "predictions.tsv"
    vcf = tmp_path / "filtered.vcf"
    evaluation = tmp_path / "evaluation.txt"

    write_demo_signals(signals)
    write_demo_truth(truth)

    run_cli("cluster_events.py", "--signals", signals, "--window", "500", "--min_support", "1", "--out", candidates)
    cand_df = pd.read_csv(candidates, sep="\t")
    assert len(cand_df) == 2
    assert set(cand_df["svtype"]) == {"DEL", "INS"}
    assert cand_df.loc[cand_df["svtype"] == "DEL", "support_read_count"].iloc[0] == 2

    run_cli("label_candidates.py", "--candidates", candidates, "--truth", truth, "--out", labeled)
    labeled_df = pd.read_csv(labeled, sep="\t")
    assert labeled_df["label"].sum() == 1
    assert labeled_df.loc[labeled_df["label"] == 1, "matched_truth_id"].iloc[0] == "truth_del"

    pred.write_text(
        "candidate_id\tchrom\tstart\tend\tsvtype\tmedian_svlen\tsupport_read_count\tlabel\tgnn_prob\tgnn_pred\n"
        "CAND_000001\tchr1\t110\t171\tDEL\t61\t2\t1\t0.91\t1\n"
        "CAND_000002\tchr1\t1000\t1001\tINS\t80\t1\t0\t0.30\t0\n"
    )
    run_cli("export_vcf.py", "--pred", pred, "--threshold", "0.5", "--out", vcf)
    vcf_text = vcf.read_text()
    assert "##source=ReadGraphSV_v0.1" in vcf_text
    assert '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">' in vcf_text
    assert "chr1\t111\tReadGraphSV_1\tN\t<DEL>" in vcf_text
    assert "SVLEN=-61" in vcf_text

    run_cli("evaluate_predictions.py", "--pred", pred, "--threshold", "0.5", "--out", evaluation)
    evaluation_text = evaluation.read_text()
    assert "Raw Precision: 0.500000" in evaluation_text
    assert "GNN F1: 1.000000" in evaluation_text
