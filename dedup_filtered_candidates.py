#!/usr/bin/env python3
"""Deduplicate filtered ReadGraphSV candidates with NMS-style clustering."""

import argparse
import logging
import math
import os
from collections import defaultdict

import pandas as pd


SUMMARY_NAME = "dedup_summary.txt"
SOURCE_PRIORITY = {"CIGAR_EXTRA": 3, "CIGAR_ONLY": 2, "EXTRA_ONLY": 1}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", required=True, help="Input filtered_candidates.tsv")
    parser.add_argument("--labeled", default=None, help="Optional candidates_labeled.tsv for candidate metadata")
    parser.add_argument("--out", required=True, help="Output deduplicated filtered_candidates TSV")
    parser.add_argument("--window", type=int, default=500, help="Maximum start distance inside a duplicate cluster")
    parser.add_argument("--min-size-sim", type=float, default=0.5, help="Minimum size similarity for duplicate calls")
    parser.add_argument("--score-col", default="gnn_prob", help="Score column used to choose the representative")
    return parser.parse_args(argv)


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_tsv(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.warning("Input table is missing or empty: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        logging.warning("Input table has no readable rows: %s", path)
        return pd.DataFrame()


def numeric(value, default=0.0):
    try:
        if value is None or value == "" or pd.isna(value):
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def row_start(row):
    return int(round(numeric(row.get("start", 0), 0)))


def row_size(row):
    for column in ["median_svlen", "svlen", "size"]:
        if column not in row:
            continue
        value = numeric(row.get(column), 0)
        if abs(value) > 0:
            return abs(value)

    start = numeric(row.get("start", 0), 0)
    end = numeric(row.get("end", start), start)
    return abs(end - start)


def size_similarity(size_a, size_b):
    size_a = abs(float(size_a))
    size_b = abs(float(size_b))
    if size_a <= 0 or size_b <= 0:
        return 0.0
    return min(size_a, size_b) / max(size_a, size_b)


def empty_like(value):
    if value is None or pd.isna(value):
        return True
    return str(value).strip() in {"", "NA", "nan"}


def supplement_with_labeled(filtered, labeled_path):
    if not labeled_path:
        return filtered

    labeled = read_tsv(labeled_path)
    if filtered.empty or labeled.empty:
        return filtered
    if "candidate_id" not in filtered.columns or "candidate_id" not in labeled.columns:
        logging.warning("Cannot merge labeled metadata without candidate_id in both tables")
        return filtered

    filtered = filtered.copy()
    labeled = labeled.drop_duplicates("candidate_id", keep="first").copy()
    labeled.index = labeled["candidate_id"].astype(str)
    keys = filtered["candidate_id"].astype(str)

    for column in labeled.columns:
        if column == "candidate_id":
            continue
        mapped = keys.map(labeled[column])
        if column not in filtered.columns:
            filtered[column] = mapped
            continue

        missing = filtered[column].map(empty_like)
        filtered.loc[missing, column] = mapped[missing]

    return filtered


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def duplicate_compatible(row_a, row_b, window, min_size_sim):
    if abs(row_start(row_a) - row_start(row_b)) > window:
        return False
    return size_similarity(row_size(row_a), row_size(row_b)) >= min_size_sim


def source_priority(value):
    return SOURCE_PRIORITY.get(str(value), 0)


def winner_key(row, original_index, score_col):
    return (
        numeric(row.get(score_col, 0), 0),
        numeric(row.get("support_read_count", 0), 0),
        source_priority(row.get("candidate_source", "")),
        -row_start(row),
        -int(original_index),
    )


def cluster_indices(group, window, min_size_sim):
    indices = list(group.index)
    uf = UnionFind(indices)
    ordered = sorted(indices, key=lambda idx: row_start(group.loc[idx]))

    for left_pos, left_idx in enumerate(ordered):
        left_row = group.loc[left_idx]
        left_start = row_start(left_row)
        for right_idx in ordered[left_pos + 1 :]:
            right_row = group.loc[right_idx]
            if row_start(right_row) - left_start > window:
                break
            if duplicate_compatible(left_row, right_row, window, min_size_sim):
                uf.union(left_idx, right_idx)

    clusters = defaultdict(list)
    for idx in indices:
        clusters[uf.find(idx)].append(idx)
    return list(clusters.values())


def required_columns_present(df, score_col):
    required = {"chrom", "start", "svtype", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"filtered candidates are missing required columns: {sorted(missing)}")


def deduplicate_candidates(filtered, window=500, min_size_sim=0.5, score_col="gnn_prob"):
    if filtered.empty:
        return filtered.copy(), []

    required_columns_present(filtered, score_col)
    filtered = filtered.copy()
    keep_indices = []

    for (_chrom, _svtype), group in filtered.groupby(["chrom", "svtype"], sort=False):
        for cluster in cluster_indices(group, window, min_size_sim):
            winner = max(cluster, key=lambda idx: winner_key(filtered.loc[idx], idx, score_col))
            keep_indices.append(winner)

    keep_set = set(keep_indices)
    removed_indices = [idx for idx in filtered.index if idx not in keep_set]
    dedup = filtered.loc[[idx for idx in filtered.index if idx in keep_set]].copy()
    return dedup.reset_index(drop=True), removed_indices


def write_summary(input_df, output_df, removed_indices, summary_path):
    removed_df = input_df.loc[removed_indices] if removed_indices else input_df.iloc[0:0]
    removed_by_svtype = removed_df["svtype"].value_counts().to_dict() if "svtype" in removed_df.columns else {}

    with open(summary_path, "w") as handle:
        print(f"Input candidates: {len(input_df)}", file=handle)
        print(f"Output candidates: {len(output_df)}", file=handle)
        print(f"Removed candidates: {len(removed_indices)}", file=handle)
        print("Removed by svtype:", file=handle)
        if removed_by_svtype:
            for svtype in sorted(removed_by_svtype):
                print(f"{svtype}\t{int(removed_by_svtype[svtype])}", file=handle)
        else:
            print("None\t0", file=handle)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    filtered = read_tsv(args.filtered)
    filtered = supplement_with_labeled(filtered, args.labeled)
    dedup, removed_indices = deduplicate_candidates(
        filtered,
        window=args.window,
        min_size_sim=args.min_size_sim,
        score_col=args.score_col,
    )
    dedup.to_csv(args.out, sep="\t", index=False)

    summary_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), SUMMARY_NAME)
    write_summary(filtered, dedup, removed_indices, summary_path)
    logging.info("Deduplicated %d candidates to %d; removed %d", len(filtered), len(dedup), len(removed_indices))
    logging.info("Wrote deduplicated candidates to %s", args.out)
    logging.info("Wrote deduplication summary to %s", summary_path)


if __name__ == "__main__":
    main()
