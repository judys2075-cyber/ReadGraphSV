"""Tests for v0.2 extra evidence extraction."""

import pandas as pd

from conftest import run_cli


def make_sequence(length):
    return "A" * length


def write_demo_sam(path):
    path.write_text(
        "@HD\tVN:1.6\tSO:unknown\n"
        "@SQ\tSN:chr1\tLN:100000\n"
        "@SQ\tSN:chr2\tLN:100000\n"
        "readA\t0\tchr1\t101\t60\t60S100M40S\t*\t0\t0\t"
        f"{make_sequence(200)}\t*\tSA:Z:chr2,501,-,80M70S,55,3;\tNM:i:1\n"
        "readA\t2048\tchr2\t501\t55\t80M70S\t*\t0\t0\t"
        f"{make_sequence(150)}\t*\tNM:i:3\n"
        "readB\t256\tchr1\t1001\t20\t100S100M\t*\t0\t0\t"
        f"{make_sequence(200)}\t*\n"
        "readC\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n"
    )


def test_extract_extra_events_softclip_sa_and_supplementary(tmp_path):
    sam = tmp_path / "demo.sam"
    out = tmp_path / "extra_signals.tsv"
    write_demo_sam(sam)

    run_cli("extract_extra_events.py", "--bam", sam, "--min_clip", "50", "--out", out)
    df = pd.read_csv(out, sep="\t")

    assert set(df["evidence_type"]) == {"SOFTCLIP_LEFT", "SOFTCLIP_RIGHT", "SA_CONNECTION", "SUPPLEMENTARY"}
    assert len(df) == 4

    left_clip = df[df["evidence_type"] == "SOFTCLIP_LEFT"].iloc[0]
    assert left_clip["src_chrom"] == "chr1"
    assert left_clip["event_pos"] == 100
    assert left_clip["event_len"] == 60
    assert left_clip["source"] == "CIGAR_SOFTCLIP"

    sa = df[df["evidence_type"] == "SA_CONNECTION"].iloc[0]
    assert sa["src_chrom"] == "chr1"
    assert sa["src_start"] == 100
    assert sa["dst_chrom"] == "chr2"
    assert sa["dst_start"] == 500
    assert sa["dst_end"] == 580
    assert sa["dst_strand"] == "-"
    assert sa["dst_mapq"] == 55
    assert sa["chrom_change"] == 1
    assert sa["orientation_change"] == 1
    assert sa["sa_cigar"] == "80M70S"
    assert sa["nm"] == 3

    supplementary = df[df["evidence_type"] == "SUPPLEMENTARY"].iloc[0]
    assert supplementary["src_chrom"] == "chr2"
    assert supplementary["event_pos"] == 500
    assert supplementary["event_len"] == 80
    assert supplementary["is_supplementary"] == 1

    right_clip = df[df["evidence_type"] == "SOFTCLIP_RIGHT"].iloc[0]
    assert right_clip["src_chrom"] == "chr2"
    assert right_clip["event_pos"] == 580
    assert right_clip["event_len"] == 70


def test_extract_extra_events_writes_empty_header(tmp_path):
    sam = tmp_path / "empty_extra.sam"
    out = tmp_path / "extra_signals.tsv"
    sam.write_text(
        "@HD\tVN:1.6\tSO:unknown\n"
        "@SQ\tSN:chr1\tLN:100000\n"
        "readA\t0\tchr1\t101\t60\t100M\t*\t0\t0\t"
        f"{make_sequence(100)}\t*\n"
    )

    run_cli("extract_extra_events.py", "--bam", sam, "--min_clip", "50", "--out", out)
    df = pd.read_csv(out, sep="\t")
    assert df.empty
    assert list(df.columns)[0:4] == ["read_name", "evidence_type", "src_chrom", "src_start"]
