#!/usr/bin/env python3
"""Merge CIGAR-derived and extra-evidence DEL/INS candidate proposals."""

import argparse
import logging
import math
import os

import pandas as pd


CORE_COLUMNS = ["chrom", "start", "end", "svtype", "median_svlen"]
COMPAT_COLUMNS = [
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
ADDED_COLUMNS = ["candidate_source", "extra_support", "matched_extra_count", "extra_source"]
VALID_SVTYPES = {"DEL", "INS"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cigar-candidates", required=True, help="Input CIGAR candidates.tsv")
    parser.add_argument("--extra-candidates", required=True, help="Input extra_candidates.tsv")
    parser.add_argument("--out", required=True, help="Output merged candidates TSV")
    parser.add_argument("--window", type=int, default=500, help="Maximum start distance for merging")
    parser.add_argument("--min-size-sim", type=float, default=0.5, help="Minimum candidate size similarity")
    parser.add_argument(
        "--min-extra-only-support",
        type=int,
        default=30,
        help="Minimum extra support required for EXTRA_ONLY candidates; use 0 to keep all",
    )
    return parser.parse_args(argv)


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_table(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.warning("Input file is missing or empty: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        logging.warning("Input file has no readable rows: %s", path)
        return pd.DataFrame()


def parse_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def parse_int(value, default=0):
    parsed = parse_float(value, default=None)
    if parsed is None:
        return default
    return int(round(parsed))


def row_start(row):
    return parse_int(row.get("start"), default=0)


def row_size(row):
    for column in ["median_svlen", "svlen", "size"]:
        if column not in row:
            continue
        value = parse_float(row.get(column), default=None)
        if value is not None and abs(value) > 0:
            return abs(value)

    start = parse_float(row.get("start"), default=None)
    end = parse_float(row.get("end"), default=None)
    if start is not None and end is not None and abs(end - start) > 0:
        return abs(end - start)
    return 0.0


def signed_median_svlen(row):
    svtype = str(row.get("svtype", "")).upper()
    for column in ["median_svlen", "svlen", "size"]:
        if column not in row:
            continue
        value = parse_float(row.get(column), default=None)
        if value is not None and abs(value) > 0:
            if column == "size" and svtype == "DEL":
                return -abs(int(round(value)))
            return int(round(value))

    size = int(round(row_size(row)))
    if svtype == "DEL":
        return -abs(size)
    return abs(size)


def size_similarity(size_a, size_b):
    size_a = abs(float(size_a))
    size_b = abs(float(size_b))
    if size_a <= 0 or size_b <= 0:
        return 0.0
    return min(size_a, size_b) / max(size_a, size_b)


def chrom_sort_key(chrom):
    text = str(chrom)
    stripped = text[3:] if text.lower().startswith("chr") else text
    aliases = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    upper = stripped.upper()
    if upper in aliases:
        return (0, aliases[upper], text)
    try:
        return (0, int(stripped), text)
    except ValueError:
        return (1, stripped, text)


def normalize_svtype(value):
    return str(value).upper()


def normalize_cigar_candidates(cigar):
    cigar = cigar.copy()
    for column in COMPAT_COLUMNS:
        if column not in cigar.columns:
            cigar[column] = ""

    for idx, row in cigar.iterrows():
        if not str(row.get("candidate_id", "")).strip() or str(row.get("candidate_id", "")) == "nan":
            cigar.at[idx, "candidate_id"] = f"CIGAR_{idx + 1:06d}"
        cigar.at[idx, "svtype"] = normalize_svtype(row.get("svtype", ""))
        if parse_float(row.get("median_svlen"), default=None) is None:
            cigar.at[idx, "median_svlen"] = signed_median_svlen(row)
        if parse_float(row.get("end"), default=None) is None:
            start = row_start(row)
            svtype = normalize_svtype(row.get("svtype", ""))
            size = int(round(row_size(row)))
            cigar.at[idx, "end"] = start + size if svtype == "DEL" else start + 1
        if parse_float(row.get("support_read_count"), default=None) is None:
            cigar.at[idx, "support_read_count"] = 0
        for column in ["mean_mapq", "std_pos", "std_svlen"]:
            if parse_float(row.get(column), default=None) is None:
                cigar.at[idx, column] = 0

    return cigar


def normalize_extra_candidates(extra):
    extra = extra.copy()
    for column in CORE_COLUMNS:
        if column not in extra.columns:
            extra[column] = ""
    if "support" not in extra.columns:
        extra["support"] = 1

    normalized_rows = []
    for idx, row in extra.iterrows():
        svtype = normalize_svtype(row.get("svtype", ""))
        if svtype not in VALID_SVTYPES:
            logging.warning("Skipping extra candidate with unsupported svtype: %s", row.get("svtype", ""))
            continue
        start = row_start(row)
        median_svlen = signed_median_svlen(row)
        end = parse_int(row.get("end"), default=None)
        if end is None:
            size = abs(median_svlen)
            end = start + size if svtype == "DEL" else start + 1

        normalized = row.to_dict()
        normalized["chrom"] = str(row.get("chrom", ""))
        normalized["start"] = int(start)
        normalized["end"] = int(end)
        normalized["svtype"] = svtype
        normalized["median_svlen"] = int(median_svlen)
        normalized["_extra_index"] = idx
        normalized_rows.append(normalized)

    return pd.DataFrame(normalized_rows)


def extra_support(row):
    support = parse_int(row.get("support"), default=1)
    return max(1, support)


def extra_source(row):
    source = str(row.get("source", "")).strip()
    if not source or source.lower() == "nan":
        return ""
    return source


def combine_extra_sources(extra, matched_indices):
    sources = sorted({extra_source(extra.loc[extra_idx]) for extra_idx in matched_indices})
    return ",".join(source for source in sources if source)


def find_best_cigar_match(extra_row, cigar, window, min_size_sim):
    best = None
    extra_chrom = str(extra_row.get("chrom", ""))
    extra_svtype = normalize_svtype(extra_row.get("svtype", ""))
    extra_start = row_start(extra_row)
    extra_size = row_size(extra_row)

    for cigar_idx, cigar_row in cigar.iterrows():
        if str(cigar_row.get("chrom", "")) != extra_chrom:
            continue
        if normalize_svtype(cigar_row.get("svtype", "")) != extra_svtype:
            continue

        distance = abs(row_start(cigar_row) - extra_start)
        if distance > window:
            continue

        similarity = size_similarity(row_size(cigar_row), extra_size)
        if similarity < min_size_sim:
            continue

        key = (distance, -similarity, cigar_idx)
        if best is None or key < best[0]:
            best = (key, cigar_idx)

    return None if best is None else best[1]


def output_columns(cigar):
    columns = list(cigar.columns)
    for column in COMPAT_COLUMNS + ADDED_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def make_extra_only_row(extra_row, columns, extra_number):
    row = {column: "" for column in columns}
    for column in CORE_COLUMNS:
        row[column] = extra_row.get(column, "")

    row["candidate_id"] = f"EXTRA_{extra_number:06d}"
    row["chrom"] = str(extra_row.get("chrom", ""))
    row["start"] = int(row_start(extra_row))
    row["end"] = int(parse_int(extra_row.get("end"), default=row["start"] + 1))
    row["svtype"] = normalize_svtype(extra_row.get("svtype", ""))
    row["median_svlen"] = int(signed_median_svlen(extra_row))
    row["support_read_count"] = extra_support(extra_row)
    row["mean_mapq"] = 0
    row["std_pos"] = 0
    row["std_svlen"] = 0
    row["read_names"] = ""
    row["signal_indices"] = ""
    row["candidate_source"] = "EXTRA_ONLY"
    row["extra_support"] = extra_support(extra_row)
    row["matched_extra_count"] = 0
    row["extra_source"] = extra_source(extra_row)
    return row


def merge_candidates(cigar, extra, window=500, min_size_sim=0.5, min_extra_only_support=30):
    cigar = normalize_cigar_candidates(cigar)
    extra = normalize_extra_candidates(extra)
    columns = output_columns(cigar)

    matches_by_cigar = {idx: [] for idx in cigar.index}
    matched_extra = set()
    for extra_idx, extra_row in extra.iterrows():
        cigar_idx = find_best_cigar_match(extra_row, cigar, window, min_size_sim)
        if cigar_idx is None:
            continue
        matches_by_cigar[cigar_idx].append(extra_idx)
        matched_extra.add(extra_idx)

    rows = []
    for cigar_idx, cigar_row in cigar.iterrows():
        row = {column: cigar_row.get(column, "") for column in columns}
        matched_indices = matches_by_cigar.get(cigar_idx, [])
        support = sum(extra_support(extra.loc[extra_idx]) for extra_idx in matched_indices)
        row["candidate_source"] = "CIGAR_EXTRA" if matched_indices else "CIGAR_ONLY"
        row["extra_support"] = int(support)
        row["matched_extra_count"] = int(len(matched_indices))
        row["extra_source"] = combine_extra_sources(extra, matched_indices)
        rows.append(row)

    extra_number = 1
    for extra_idx, extra_row in extra.iterrows():
        if extra_idx in matched_extra:
            continue
        if extra_support(extra_row) < min_extra_only_support:
            continue
        rows.append(make_extra_only_row(extra_row, columns, extra_number))
        extra_number += 1

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return pd.DataFrame(columns=columns)

    for column in CORE_COLUMNS:
        if column not in out.columns:
            out[column] = ""

    out["_chrom_sort"] = out["chrom"].map(chrom_sort_key)
    out["_start_sort"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(int)
    out["_svtype_sort"] = out["svtype"].astype(str)
    out = out.sort_values(["_chrom_sort", "_start_sort", "_svtype_sort"], kind="mergesort")
    out = out.drop(columns=["_chrom_sort", "_start_sort", "_svtype_sort"]).reset_index(drop=True)
    return out


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    cigar = read_table(args.cigar_candidates)
    extra = read_table(args.extra_candidates)
    merged = merge_candidates(
        cigar,
        extra,
        window=args.window,
        min_size_sim=args.min_size_sim,
        min_extra_only_support=args.min_extra_only_support,
    )
    merged.to_csv(args.out, sep="\t", index=False)

    source_counts = merged["candidate_source"].value_counts().to_dict() if "candidate_source" in merged.columns else {}
    logging.info("Merged %d CIGAR candidates and %d extra candidates into %d rows", len(cigar), len(extra), len(merged))
    logging.info("Candidate source counts: %s", source_counts)
    logging.info("Wrote merged candidates to %s", args.out)


if __name__ == "__main__":
    main()
