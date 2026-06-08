"""Tests for v0.3 CIGAR/extra candidate merging."""

import csv

import pandas as pd

from conftest import run_cli
from merge_candidates_v3 import CORE_COLUMNS, merge_candidates, parse_args


CIGAR_COLUMNS = [
    "candidate_id",
    "chrom",
    "start",
    "end",
    "svtype",
    "median_svlen",
    "support_read_count",
    "mean_mapq",
    "std_pos",
    "std_svlen",
    "read_names",
    "signal_indices",
]

EXTRA_COLUMNS = ["chrom", "start", "end", "svtype", "svlen", "median_svlen", "support", "source", "mean_pos", "mean_size"]


def write_tsv(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)


def cigar_row(candidate_id="CAND_000001", chrom="chr1", start=1000, end=1100, svtype="DEL", median_svlen=100):
    return {
        "candidate_id": candidate_id,
        "chrom": chrom,
        "start": start,
        "end": end,
        "svtype": svtype,
        "median_svlen": median_svlen,
        "support_read_count": 3,
        "mean_mapq": 60,
        "std_pos": 5,
        "std_svlen": 2,
        "read_names": "read1,read2,read3",
        "signal_indices": "0,1,2",
    }


def extra_row(chrom="chr1", start=1020, end=1120, svtype="DEL", svlen=-100, support=2, source="SA"):
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "svtype": svtype,
        "svlen": svlen,
        "median_svlen": svlen,
        "support": support,
        "source": source,
        "mean_pos": start,
        "mean_size": abs(svlen),
    }


def test_parse_args_for_merge_candidates_v3():
    args = parse_args(
        [
            "--cigar-candidates",
            "candidates.tsv",
            "--extra-candidates",
            "extra_candidates.tsv",
            "--out",
            "merged.tsv",
            "--window",
            "750",
            "--min-size-sim",
            "0.7",
            "--min-extra-only-support",
            "40",
        ]
    )
    assert args.cigar_candidates == "candidates.tsv"
    assert args.extra_candidates == "extra_candidates.tsv"
    assert args.out == "merged.tsv"
    assert args.window == 750
    assert args.min_size_sim == 0.7
    assert args.min_extra_only_support == 40


def test_cigar_only_candidate_is_preserved():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame(columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra)

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["candidate_id"] == "CAND_000001"
    assert row["candidate_source"] == "CIGAR_ONLY"
    assert int(row["extra_support"]) == 0
    assert int(row["matched_extra_count"]) == 0
    assert row["extra_source"] == ""


def test_cigar_extra_candidate_merges_when_position_and_size_match():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=1040, support=4)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5)

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["candidate_source"] == "CIGAR_EXTRA"
    assert int(row["extra_support"]) == 4
    assert int(row["matched_extra_count"]) == 1
    assert row["extra_source"] == "SA"
    assert int(row["start"]) == 1000
    assert int(row["median_svlen"]) == 100


def test_extra_only_candidate_is_added_when_unmatched():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=5000, end=5001, svtype="INS", svlen=80, support=5)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=0)

    assert len(merged) == 2
    extra_only = merged[merged["candidate_source"] == "EXTRA_ONLY"].iloc[0]
    assert extra_only["candidate_id"] == "EXTRA_000001"
    assert extra_only["svtype"] == "INS"
    assert int(extra_only["support_read_count"]) == 5
    assert int(extra_only["extra_support"]) == 5
    assert int(extra_only["median_svlen"]) == 80
    assert extra_only["extra_source"] == "SA"


def test_different_svtype_does_not_merge():
    cigar = pd.DataFrame([cigar_row(svtype="DEL", median_svlen=100)], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=1000, end=1001, svtype="INS", svlen=100, support=2)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=500, min_size_sim=0.5, min_extra_only_support=0)

    assert len(merged) == 2
    assert set(merged["candidate_source"]) == {"CIGAR_ONLY", "EXTRA_ONLY"}


def test_size_similarity_below_threshold_does_not_merge():
    cigar = pd.DataFrame([cigar_row(median_svlen=100)], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=1020, end=1320, svlen=-300, support=2)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=500, min_size_sim=0.5, min_extra_only_support=0)

    assert len(merged) == 2
    assert set(merged["candidate_source"]) == {"CIGAR_ONLY", "EXTRA_ONLY"}


def test_output_contains_required_and_added_columns():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=5000, end=5001, svtype="INS", svlen=70)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra)

    for column in [*CORE_COLUMNS, "candidate_source", "extra_support", "matched_extra_count", "extra_source"]:
        assert column in merged.columns
    assert {"chrom", "start", "svtype", "median_svlen"}.issubset(merged.columns)


def test_cli_writes_label_candidates_compatible_output(tmp_path):
    cigar_path = tmp_path / "candidates.tsv"
    extra_path = tmp_path / "extra_candidates.tsv"
    out_path = tmp_path / "merged.tsv"
    write_tsv(cigar_path, CIGAR_COLUMNS, [cigar_row()])
    write_tsv(extra_path, EXTRA_COLUMNS, [extra_row(start=5000, end=5001, svtype="INS", svlen=90, support=3)])

    run_cli(
        "merge_candidates_v3.py",
        "--cigar-candidates",
        cigar_path,
        "--extra-candidates",
        extra_path,
        "--out",
        out_path,
        "--window",
        "100",
        "--min-size-sim",
        "0.5",
        "--min-extra-only-support",
        "0",
    )

    with out_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert len(rows) == 2
    for row in rows:
        for column in ["chrom", "start", "svtype", "median_svlen"]:
            assert row[column] != ""
    assert {row["candidate_source"] for row in rows} == {"CIGAR_ONLY", "EXTRA_ONLY"}


def test_extra_only_below_min_extra_only_support_is_filtered():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=5000, end=5001, svtype="INS", svlen=90, support=29)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=30)

    assert len(merged) == 1
    assert set(merged["candidate_source"]) == {"CIGAR_ONLY"}


def test_extra_only_at_min_extra_only_support_is_kept():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=5000, end=5001, svtype="INS", svlen=90, support=30)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=30)

    assert len(merged) == 2
    extra_only = merged[merged["candidate_source"] == "EXTRA_ONLY"].iloc[0]
    assert int(extra_only["extra_support"]) == 30
    assert extra_only["extra_source"] == "SA"


def test_cigar_only_is_not_affected_by_min_extra_only_support():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame(columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, min_extra_only_support=999)

    assert len(merged) == 1
    assert merged.iloc[0]["candidate_source"] == "CIGAR_ONLY"


def test_cigar_extra_is_not_affected_by_min_extra_only_support():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=1000, support=2, source="SA")], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=999)

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["candidate_source"] == "CIGAR_EXTRA"
    assert int(row["extra_support"]) == 2
    assert row["extra_source"] == "SA"


def test_min_extra_only_support_zero_keeps_all_extra_only():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame([extra_row(start=5000, end=5001, svtype="INS", svlen=90, support=1)], columns=EXTRA_COLUMNS)

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=0)

    assert len(merged) == 2
    assert "EXTRA_ONLY" in set(merged["candidate_source"])


def test_multiple_matched_extra_sources_are_combined():
    cigar = pd.DataFrame([cigar_row()], columns=CIGAR_COLUMNS)
    extra = pd.DataFrame(
        [
            extra_row(start=1000, support=2, source="SA"),
            extra_row(start=1020, support=3, source="SOFTCLIP"),
            extra_row(start=1030, support=4, source="SA"),
        ],
        columns=EXTRA_COLUMNS,
    )

    merged = merge_candidates(cigar, extra, window=100, min_size_sim=0.5, min_extra_only_support=30)

    row = merged.iloc[0]
    assert row["candidate_source"] == "CIGAR_EXTRA"
    assert int(row["extra_support"]) == 9
    assert int(row["matched_extra_count"]) == 3
    assert row["extra_source"] == "SA,SOFTCLIP"
