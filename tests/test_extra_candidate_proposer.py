"""Tests for v0.3 extra evidence candidate proposal."""

import pandas as pd

from conftest import run_cli
from extra_candidate_proposer import OUTPUT_FIELDS, parse_args


EXTRA_HEADER = (
    "read_name\tevidence_type\tsrc_chrom\tsrc_start\tsrc_end\tsrc_strand\t"
    "dst_chrom\tdst_start\tdst_end\tdst_strand\tevent_pos\tevent_len\t"
    "mapq\tdst_mapq\tchrom_change\torientation_change\tis_supplementary\t"
    "has_sa\tsoftclip_left\tsoftclip_right\tcigar\tsa_cigar\tnm\tsource\n"
)


def write_tiny_extra(path):
    path.write_text(
        EXTRA_HEADER
        + "read1\tSOFTCLIP_LEFT\tchr1\t100\t200\t+\t\t\t\t\t100\t60\t60\t\t0\t0\t0\t0\t60\t0\t60S100M\t\t1\tCIGAR_SOFTCLIP\n"
        + "read2\tSOFTCLIP_RIGHT\tchr1\t110\t210\t+\t\t\t\t\t120\t70\t55\t\t0\t0\t0\t0\t0\t70\t100M70S\t\t2\tCIGAR_SOFTCLIP\n"
        + "read3\tSA_CONNECTION\tchr1\t500\t600\t+\tchr1\t700\t800\t+\t500\t200\t50\t50\t0\t0\t0\t1\t0\t0\t100M\t100M\t3\tSA_TAG\n"
        + "read4\tSA_CONNECTION\tchr1\t510\t610\t+\tchr1\t710\t810\t+\t510\t200\t50\t50\t0\t0\t0\t1\t0\t0\t100M\t100M\t3\tSA_TAG\n"
    )


def test_parse_args_for_extra_candidate_proposer():
    args = parse_args(
        [
            "--extra",
            "extra_signals.tsv",
            "--out",
            "extra_candidates.tsv",
            "--window",
            "750",
            "--min_support",
            "3",
            "--min_size",
            "80",
        ]
    )
    assert args.extra == "extra_signals.tsv"
    assert args.out == "extra_candidates.tsv"
    assert args.window == 750
    assert args.min_support == 3
    assert args.min_softclip_support == 3
    assert args.min_sa_support == 3
    assert args.min_supplementary_support == 3
    assert args.min_size == 80


def test_parse_args_uses_conservative_source_defaults():
    args = parse_args(["--extra", "extra_signals.tsv", "--out", "extra_candidates.tsv"])
    assert args.min_support == 2
    assert args.min_softclip_support == 10
    assert args.min_sa_support == 2
    assert args.min_supplementary_support == 2


def test_extra_candidate_proposer_generates_del_and_ins(tmp_path):
    extra = tmp_path / "extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    write_tiny_extra(extra)

    run_cli(
        "extra_candidate_proposer.py",
        "--extra",
        extra,
        "--out",
        out,
        "--window",
        "50",
        "--min_support",
        "2",
        "--min_size",
        "50",
    )

    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == OUTPUT_FIELDS
    assert "median_svlen" in df.columns
    assert len(df) == 2
    assert set(df["svtype"]) == {"DEL", "INS"}
    assert set(df["source"]) == {"SA", "SOFTCLIP"}

    ins = df[df["svtype"] == "INS"].iloc[0]
    assert ins["chrom"] == "chr1"
    assert int(ins["support"]) == 2
    assert int(ins["svlen"]) > 0
    assert int(ins["median_svlen"]) == int(ins["svlen"])
    assert int(ins["start"]) == 110
    assert int(ins["end"]) == 111

    deletion = df[df["svtype"] == "DEL"].iloc[0]
    assert int(deletion["support"]) == 2
    assert int(deletion["svlen"]) < 0
    assert int(deletion["median_svlen"]) == int(deletion["svlen"])
    assert abs(int(deletion["svlen"])) == 100
    assert int(deletion["end"]) - int(deletion["start"]) == 100


def test_softclip_below_default_source_support_is_filtered(tmp_path):
    extra = tmp_path / "extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    write_tiny_extra(extra)

    run_cli("extra_candidate_proposer.py", "--extra", extra, "--out", out, "--window", "50", "--min_size", "50")

    df = pd.read_csv(out, sep="\t")
    assert set(df["source"]) == {"SA"}
    assert set(df["svtype"]) == {"DEL"}


def test_sa_support_can_be_controlled_independently(tmp_path):
    extra = tmp_path / "extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    write_tiny_extra(extra)

    run_cli(
        "extra_candidate_proposer.py",
        "--extra",
        extra,
        "--out",
        out,
        "--window",
        "50",
        "--min_support",
        "2",
        "--min_softclip_support",
        "2",
        "--min_sa_support",
        "3",
        "--min_size",
        "50",
    )

    df = pd.read_csv(out, sep="\t")
    assert set(df["source"]) == {"SOFTCLIP"}
    assert set(df["svtype"]) == {"INS"}


def test_old_min_support_fallback_keeps_softclip_compatible(tmp_path):
    extra = tmp_path / "extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    write_tiny_extra(extra)

    run_cli(
        "extra_candidate_proposer.py",
        "--extra",
        extra,
        "--out",
        out,
        "--window",
        "50",
        "--min_support",
        "2",
        "--min_size",
        "50",
    )

    df = pd.read_csv(out, sep="\t")
    assert set(df["source"]) == {"SA", "SOFTCLIP"}


def test_singleton_low_support_cluster_is_filtered(tmp_path):
    extra = tmp_path / "extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    extra.write_text(
        EXTRA_HEADER
        + "read1\tSOFTCLIP_LEFT\tchr1\t100\t200\t+\t\t\t\t\t100\t60\t60\t\t0\t0\t0\t0\t60\t0\t60S100M\t\t1\tCIGAR_SOFTCLIP\n"
    )

    run_cli(
        "extra_candidate_proposer.py",
        "--extra",
        extra,
        "--out",
        out,
        "--min_support",
        "2",
        "--min_size",
        "50",
    )

    df = pd.read_csv(out, sep="\t")
    assert df.empty
    assert list(df.columns) == OUTPUT_FIELDS


def test_missing_fields_do_not_crash(tmp_path):
    extra = tmp_path / "bad_extra_signals.tsv"
    out = tmp_path / "extra_candidates.tsv"
    extra.write_text("read_name\tevidence_type\nread1\tSOFTCLIP_LEFT\nread2\tSA_CONNECTION\n")

    run_cli("extra_candidate_proposer.py", "--extra", extra, "--out", out)

    df = pd.read_csv(out, sep="\t")
    assert df.empty
    assert list(df.columns) == OUTPUT_FIELDS


def test_missing_evidence_type_column_writes_empty_output(tmp_path):
    extra = tmp_path / "missing_type.tsv"
    out = tmp_path / "extra_candidates.tsv"
    extra.write_text("read_name\tsrc_chrom\tevent_pos\tevent_len\nread1\tchr1\t100\t60\n")

    run_cli("extra_candidate_proposer.py", "--extra", extra, "--out", out)

    df = pd.read_csv(out, sep="\t")
    assert df.empty
    assert list(df.columns) == OUTPUT_FIELDS
