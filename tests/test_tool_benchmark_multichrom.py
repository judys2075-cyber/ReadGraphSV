"""Tests for multi-chromosome mainstream tool benchmark helpers."""

import csv
import json

import pytest

import scripts.collect_tool_benchmark_multichrom as collect
import scripts.run_tool_benchmark_multichrom as runner


def write_summary(path, precision=0.9, recall=0.8, f1=0.847, tp_comp=8, tp_base=9, fp=1, fn=2):
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "base cnt": tp_base + fn,
                "comp cnt": tp_comp + fp,
            }
        )
    )


def test_run_tool_benchmark_parser_defaults_and_aliases():
    args = runner.parse_args(["--chroms", "chr19,20", "--tools", "sniffles,cuteSV,svim", "--threads", "12"])

    assert args.chrom_list == ["19", "20"]
    assert args.tool_list == ["sniffles2", "cutesv", "svim"]
    assert args.threads == 12
    assert args.refdist == 500
    assert args.pctsize == 0.5
    assert args.sizemin == 50
    assert args.pctseq == 0.0


def test_run_tool_benchmark_builds_tool_and_benchmark_commands(tmp_path):
    args = runner.parse_args(
        [
            "--chroms",
            "21",
            "--tools",
            "sniffles2,cutesv,svim",
            "--bam-template",
            str(tmp_path / "bam" / "HG002_chr{chrom}.bam"),
            "--truth-template",
            str(tmp_path / "truth" / "HG002_chr{chrom}.vcf.gz"),
            "--bed-template",
            str(tmp_path / "truth" / "HG002_chr{chrom}.bed"),
            "--reference",
            str(tmp_path / "ref.fa"),
            "--out-template",
            str(tmp_path / "results" / "tool_benchmark_chr{chrom}"),
            "--threads",
            "16",
            "--dry-run",
        ]
    )

    sniffles_cmd = runner.build_tool_command(args, "21", "sniffles2")
    cutesv_cmd = runner.build_tool_command(args, "21", "cutesv")
    svim_cmd = runner.build_tool_command(args, "21", "svim")
    benchmark_cmd = runner.build_benchmark_command(args, "21", "cutesv")

    assert sniffles_cmd[:2] == ["sniffles", "--input"]
    assert "--threads" in sniffles_cmd
    assert sniffles_cmd[sniffles_cmd.index("--threads") + 1] == "16"
    assert str(tmp_path / "results" / "tool_benchmark_chr21" / "sniffles2" / "sniffles2.vcf") in sniffles_cmd

    assert cutesv_cmd[0] == "cuteSV"
    assert str(tmp_path / "ref.fa") in cutesv_cmd
    assert str(tmp_path / "results" / "tool_benchmark_chr21" / "cutesv" / "cutesv.vcf") in cutesv_cmd
    assert "--min_size" in cutesv_cmd
    assert cutesv_cmd[cutesv_cmd.index("--min_size") + 1] == "50"

    assert svim_cmd[:2] == ["svim", "alignment"]
    assert "--min_sv_size" in svim_cmd
    assert str(tmp_path / "results" / "tool_benchmark_chr21" / "svim") in svim_cmd

    assert "scripts/benchmark_truvari_delins.py" in benchmark_cmd[1]
    assert "--cutesv-vcf" in benchmark_cmd
    assert benchmark_cmd[benchmark_cmd.index("--chrom") + 1] == "21"
    assert benchmark_cmd[benchmark_cmd.index("--refdist") + 1] == "500"
    assert benchmark_cmd[benchmark_cmd.index("--pctsize") + 1] == "0.5"
    assert benchmark_cmd[benchmark_cmd.index("--sizemin") + 1] == "50"
    assert float(benchmark_cmd[benchmark_cmd.index("--pctseq") + 1]) == 0.0


def test_prepare_tool_directories_creates_cutesv_work_dir(tmp_path):
    args = runner.parse_args(
        [
            "--chroms",
            "21",
            "--tools",
            "cutesv",
            "--out-template",
            str(tmp_path / "results" / "tool_benchmark_chr{chrom}"),
        ]
    )

    output_dir = runner.prepare_tool_directories(args, "21", "cutesv")

    assert output_dir == tmp_path / "results" / "tool_benchmark_chr21" / "cutesv"
    assert output_dir.exists()
    assert (output_dir / "work").exists()


def test_prepare_tool_directories_creates_output_dirs_for_all_tools(tmp_path):
    args = runner.parse_args(
        [
            "--chroms",
            "21",
            "--tools",
            "sniffles2,cutesv,svim",
            "--out-template",
            str(tmp_path / "results" / "tool_benchmark_chr{chrom}"),
        ]
    )

    for tool in ["sniffles2", "cutesv", "svim"]:
        runner.prepare_tool_directories(args, "21", tool)

    assert (tmp_path / "results" / "tool_benchmark_chr21" / "sniffles2").exists()
    assert (tmp_path / "results" / "tool_benchmark_chr21" / "cutesv").exists()
    assert (tmp_path / "results" / "tool_benchmark_chr21" / "cutesv" / "work").exists()
    assert (tmp_path / "results" / "tool_benchmark_chr21" / "svim").exists()


def test_tool_benchmark_role_assignment():
    assert collect.assign_role("19") == "validation"
    assert collect.assign_role("chr20") == "training"
    assert collect.assign_role("21") == "primary held-out test"
    assert collect.assign_role("chr22") == "training"
    assert collect.assign_role("chr1") == "additional held-out test"


def test_collect_tool_benchmark_multichrom_from_fake_summaries(tmp_path):
    output = tmp_path / "final" / "tool_comparison.tsv"
    readgraphsv_template = str(tmp_path / "readgraphsv" / "chr{chrom}" / "summary.json")
    tool_template = str(tmp_path / "tools" / "chr{chrom}" / "{tool}" / "summary.json")

    write_summary(tmp_path / "readgraphsv" / "chr19" / "summary.json", precision=0.91, recall=0.81)
    write_summary(tmp_path / "tools" / "chr19" / "sniffles2" / "summary.json", precision=0.92, recall=0.82)
    write_summary(tmp_path / "tools" / "chr19" / "cutesv" / "summary.json", precision=0.83, recall=0.93)
    write_summary(tmp_path / "tools" / "chr19" / "svim" / "summary.json", precision=0.51, recall=0.94)

    args = collect.parse_args(
        [
            "--chroms",
            "19",
            "--tools",
            "readgraphsv,sniffles2,cutesv,svim",
            "--readgraphsv-summary-template",
            readgraphsv_template,
            "--tool-summary-template",
            tool_template,
            "--out",
            str(output),
        ]
    )
    output_path = collect.collect_tool_benchmark_results(args)

    assert output_path == output
    with open(output, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert [row["Tool"] for row in rows] == ["ReadGraphSV v0.3 + dedup", "Sniffles2", "cuteSV", "SVIM"]
    assert all(row["Chromosome"] == "chr19" for row in rows)
    assert all(row["Role"] == "validation" for row in rows)
    assert rows[0]["Precision"] == "0.910000"
    assert rows[1]["Recall"] == "0.820000"
    assert rows[2]["Precision"] == "0.830000"
    assert rows[3]["Recall"] == "0.940000"
    assert rows[0]["BaseCount"] == "11"
    assert rows[0]["CompCount"] == "9"
    assert rows[0]["Summary"] == str(tmp_path / "readgraphsv" / "chr19" / "summary.json")


def test_collect_tool_benchmark_missing_summary_error_is_clear(tmp_path):
    args = collect.parse_args(
        [
            "--chroms",
            "19",
            "--tools",
            "sniffles2",
            "--tool-summary-template",
            str(tmp_path / "missing" / "chr{chrom}" / "{tool}" / "summary.json"),
            "--out",
            str(tmp_path / "out.tsv"),
        ]
    )

    with pytest.raises(FileNotFoundError, match="Missing Sniffles2 chr19 Truvari summary"):
        collect.collect_tool_benchmark_results(args)
