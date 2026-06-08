"""Tests for ReadGraphSV real-data benchmark helper scripts."""

import csv
import json

import scripts.benchmark_truvari_delins as benchmark
from scripts.benchmark_truvari_delins import parse_args as parse_benchmark_args
from scripts.benchmark_truvari_delins import read_truvari_summary
from scripts.truvari_threshold_sweep import write_threshold_vcf
from scripts.write_final_hg002_results import parse_args as parse_final_args
from scripts.write_final_hg002_results import write_final_results


GT_HEADER = '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'


def write_mock_summary(path, precision=0.8, recall=0.7, f1=0.746, tp_comp=10, tp_base=9, fp=2, fn=3):
    path.write_text(
        json.dumps(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "TP-comp": tp_comp,
                "TP-base": tp_base,
                "FP": fp,
                "FN": fn,
            }
        )
    )


def test_truvari_summary_parser_reads_expected_keys(tmp_path):
    summary = tmp_path / "summary.json"
    write_mock_summary(summary, precision=0.75, recall=0.6666667, f1=0.7058823, tp_comp=3, tp_base=4, fp=1, fn=2)

    metrics = read_truvari_summary(summary)
    assert metrics["Precision"] == 0.75
    assert round(metrics["Recall"], 6) == 0.666667
    assert round(metrics["F1"], 6) == 0.705882
    assert metrics["TP-comp"] == 3
    assert metrics["TP-base"] == 4
    assert metrics["FP"] == 1
    assert metrics["FN"] == 2


def test_benchmark_script_command_line_parser():
    args = parse_benchmark_args(
        [
            "--truth",
            "truth.vcf.gz",
            "--bed",
            "tier1.bed",
            "--outdir",
            "benchmark",
            "--chrom",
            "chr21",
            "--readgraphsv-vcf",
            "readgraphsv.vcf",
            "--sniffles2-vcf",
            "sniffles2.vcf.gz",
            "--refdist",
            "700",
            "--pctsize",
            "0.6",
        ]
    )

    assert args.truth == "truth.vcf.gz"
    assert args.bed == "tier1.bed"
    assert args.outdir == "benchmark"
    assert args.chrom == "chr21"
    assert args.readgraphsv_vcf == "readgraphsv.vcf"
    assert args.sniffles2_vcf == "sniffles2.vcf.gz"
    assert args.refdist == 700
    assert args.pctsize == 0.6
    assert args.sizemin == 50


def test_benchmark_truvari_command_includes_passonly_and_matching_params(tmp_path, monkeypatch):
    captured = {}

    def fake_run_command(command):
        captured["command"] = command

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    args = parse_benchmark_args(
        [
            "--truth",
            "truth.vcf.gz",
            "--bed",
            "tier1.bed",
            "--outdir",
            str(tmp_path),
            "--refdist",
            "500",
            "--pctsize",
            "0.5",
            "--sizemin",
            "50",
            "--pctseq",
            "0",
        ]
    )

    summary_path = benchmark.run_truvari("readgraphsv", tmp_path / "candidate.vcf.gz", args)
    command = captured["command"]

    assert summary_path == tmp_path / "truvari_readgraphsv" / "summary.json"
    assert "--includebed" in command
    assert command[command.index("--includebed") + 1] == "tier1.bed"
    assert "--passonly" in command
    assert "--sizemin" in command
    assert command[command.index("--sizemin") + 1] == "50"
    assert "--refdist" in command
    assert command[command.index("--refdist") + 1] == "500"
    assert "--pctsize" in command
    assert command[command.index("--pctsize") + 1] == "0.5"
    assert "--pctseq" in command
    assert float(command[command.index("--pctseq") + 1]) == 0.0


def test_threshold_vcf_writer_includes_gt_format_header(tmp_path):
    predictions = tmp_path / "predictions_v2.tsv"
    output_vcf = tmp_path / "threshold.vcf"
    predictions.write_text(
        "candidate_id\tchrom\tstart\tend\tsvtype\tmedian_svlen\tsupport_read_count\tlabel\tgnn_prob\tgnn_pred\n"
        "CAND_1\tchr21\t100\t180\tDEL\t80\t3\t0\t0.90\t1\n"
        "CAND_2\tchr21\t300\t301\tINS\t70\t2\t0\t0.40\t0\n"
        "CAND_3\tchr20\t500\t560\tDEL\t60\t2\t0\t0.95\t1\n"
    )

    count = write_threshold_vcf(predictions, output_vcf, threshold=0.5, chrom="21", contig_length=46709983)
    text = output_vcf.read_text()

    assert count == 1
    assert GT_HEADER in text
    assert "##contig=<ID=chr21,length=46709983>" in text
    assert "chr21\t101\tReadGraphSV_t0.50_1\tN\t<DEL>" in text
    assert "SVTYPE=DEL;END=180;SVLEN=-80;SUPPORT=3;GNN_PROB=0.900000" in text
    assert "CAND_2" not in text
    assert "chr20" not in text


def test_final_result_writer_generates_table_and_readme(tmp_path):
    sim_summary = tmp_path / "sim_summary.json"
    finetuned_summary = tmp_path / "finetuned_summary.json"
    sniffles_summary = tmp_path / "sniffles_summary.json"
    cutesv_summary = tmp_path / "cutesv_summary.json"
    outdir = tmp_path / "final"
    write_mock_summary(sim_summary, precision=0.8, recall=0.7, f1=0.746, tp_comp=10, tp_base=9, fp=2, fn=3)
    write_mock_summary(finetuned_summary, precision=0.85, recall=0.75, f1=0.797, tp_comp=12, tp_base=11, fp=1, fn=2)
    write_mock_summary(sniffles_summary, precision=0.9, recall=0.8, f1=0.847, tp_comp=13, tp_base=12, fp=1, fn=1)
    write_mock_summary(cutesv_summary, precision=0.88, recall=0.82, f1=0.849, tp_comp=14, tp_base=13, fp=2, fn=5)

    args = parse_final_args(
        [
            "--readgraphsv-sim-summary",
            str(sim_summary),
            "--readgraphsv-finetuned-summary",
            str(finetuned_summary),
            "--sniffles2-summary",
            str(sniffles_summary),
            "--cutesv-summary",
            str(cutesv_summary),
            "--outdir",
            str(outdir),
        ]
    )
    table_path, readme_path = write_final_results(args)

    table = table_path.read_text()
    readme = readme_path.read_text()
    assert "ReadGraphSV simulation-trained\t0.800000\t0.700000\t0.746000\t10\t9\t2\t3" in table
    assert "ReadGraphSV real-finetuned\t0.850000\t0.750000\t0.797000\t12\t11\t1\t2" in table
    assert "Sniffles2\t0.900000\t0.800000\t0.847000\t13\t12\t1\t1" in table
    assert "cuteSV\t0.880000\t0.820000\t0.849000\t14\t13\t2\t5\t" in table
    assert "HG002 PacBio HiFi Revio GRCh37" in readme
    assert "GIAB/NIST HG002 Tier1 v0.6" in readme
    assert "trained on chr20+chr22" in readme
    assert "held-out chr21" in readme

    with open(table_path, newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0]
    assert header == ["Tool", "Precision", "Recall", "F1", "TP-comp", "TP-base", "FP", "FN", "Summary"]
    assert all(len(row) == len(header) for row in rows)

    cutesv_row = next(row for row in rows[1:] if row[0] == "cuteSV")
    assert cutesv_row[7] == "5"
    assert cutesv_row[8] == str(cutesv_summary)
    assert cutesv_row[7] != f"5{cutesv_summary}"
