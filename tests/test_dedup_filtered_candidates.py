"""Tests for filtered candidate deduplication."""

import pandas as pd

from conftest import run_cli
from dedup_filtered_candidates import deduplicate_candidates, parse_args


FILTERED_COLUMNS = [
    "candidate_id",
    "chrom",
    "start",
    "end",
    "svtype",
    "median_svlen",
    "support_read_count",
    "label",
    "gnn_prob",
    "gnn_pred",
    "candidate_source",
]


def candidate(
    candidate_id,
    chrom="chr21",
    start=100,
    end=200,
    svtype="DEL",
    median_svlen=100,
    support_read_count=3,
    gnn_prob=0.8,
    candidate_source="CIGAR_ONLY",
):
    return {
        "candidate_id": candidate_id,
        "chrom": chrom,
        "start": start,
        "end": end,
        "svtype": svtype,
        "median_svlen": median_svlen,
        "support_read_count": support_read_count,
        "label": 0,
        "gnn_prob": gnn_prob,
        "gnn_pred": 1,
        "candidate_source": candidate_source,
    }


def frame(rows):
    return pd.DataFrame(rows, columns=FILTERED_COLUMNS)


def test_parse_args_for_dedup_filtered_candidates():
    args = parse_args(
        [
            "--filtered",
            "filtered_candidates.tsv",
            "--labeled",
            "candidates_labeled.tsv",
            "--out",
            "filtered_candidates_dedup.tsv",
            "--window",
            "750",
            "--min-size-sim",
            "0.7",
            "--score-col",
            "custom_score",
        ]
    )
    assert args.filtered == "filtered_candidates.tsv"
    assert args.labeled == "candidates_labeled.tsv"
    assert args.out == "filtered_candidates_dedup.tsv"
    assert args.window == 750
    assert args.min_size_sim == 0.7
    assert args.score_col == "custom_score"


def test_nearby_same_type_same_size_keeps_highest_gnn_prob():
    df = frame(
        [
            candidate("low", start=100, median_svlen=100, gnn_prob=0.80),
            candidate("high", start=120, median_svlen=105, gnn_prob=0.95),
        ]
    )

    dedup, removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 1
    assert dedup.iloc[0]["candidate_id"] == "high"
    assert len(removed) == 1


def test_different_svtype_does_not_merge():
    df = frame(
        [
            candidate("del", start=100, svtype="DEL", median_svlen=100, gnn_prob=0.80),
            candidate("ins", start=120, end=121, svtype="INS", median_svlen=100, gnn_prob=0.95),
        ]
    )

    dedup, removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 2
    assert removed == []


def test_distance_over_window_does_not_merge():
    df = frame(
        [
            candidate("left", start=100, median_svlen=100),
            candidate("right", start=1000, median_svlen=100, gnn_prob=0.95),
        ]
    )

    dedup, removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 2
    assert removed == []


def test_size_similarity_below_threshold_does_not_merge():
    df = frame(
        [
            candidate("small", start=100, end=200, median_svlen=100),
            candidate("large", start=120, end=420, median_svlen=300, gnn_prob=0.95),
        ]
    )

    dedup, removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 2
    assert removed == []


def test_tie_break_prefers_higher_support_read_count():
    df = frame(
        [
            candidate("low_support", start=100, support_read_count=3, gnn_prob=0.90),
            candidate("high_support", start=120, support_read_count=8, gnn_prob=0.90),
        ]
    )

    dedup, _removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 1
    assert dedup.iloc[0]["candidate_id"] == "high_support"


def test_tie_break_prefers_candidate_source_order():
    df = frame(
        [
            candidate("extra_only", start=100, support_read_count=5, gnn_prob=0.90, candidate_source="EXTRA_ONLY"),
            candidate("cigar_only", start=120, support_read_count=5, gnn_prob=0.90, candidate_source="CIGAR_ONLY"),
            candidate("cigar_extra", start=140, support_read_count=5, gnn_prob=0.90, candidate_source="CIGAR_EXTRA"),
        ]
    )

    dedup, _removed = deduplicate_candidates(df, window=500, min_size_sim=0.5)

    assert len(dedup) == 1
    assert dedup.iloc[0]["candidate_id"] == "cigar_extra"


def test_cli_writes_dedup_output_summary_and_labeled_metadata(tmp_path):
    filtered = tmp_path / "filtered_candidates.tsv"
    labeled = tmp_path / "candidates_labeled.tsv"
    out = tmp_path / "filtered_candidates_dedup.tsv"
    frame(
        [
            {**candidate("first", start=100, gnn_prob=0.90), "candidate_source": ""},
            {**candidate("second", start=120, gnn_prob=0.80), "candidate_source": ""},
        ]
    ).to_csv(filtered, sep="\t", index=False)
    pd.DataFrame(
        [
            {"candidate_id": "first", "candidate_source": "CIGAR_EXTRA", "extra_support": 2},
            {"candidate_id": "second", "candidate_source": "CIGAR_ONLY", "extra_support": 0},
        ]
    ).to_csv(labeled, sep="\t", index=False)

    run_cli(
        "dedup_filtered_candidates.py",
        "--filtered",
        filtered,
        "--labeled",
        labeled,
        "--out",
        out,
        "--window",
        "500",
        "--min-size-sim",
        "0.5",
    )

    dedup = pd.read_csv(out, sep="\t")
    summary = (tmp_path / "dedup_summary.txt").read_text()
    assert len(dedup) == 1
    assert dedup.iloc[0]["candidate_id"] == "first"
    assert dedup.iloc[0]["candidate_source"] == "CIGAR_EXTRA"
    assert int(dedup.iloc[0]["extra_support"]) == 2
    assert "Input candidates: 2" in summary
    assert "Output candidates: 1" in summary
    assert "Removed candidates: 1" in summary
    assert "DEL\t1" in summary
