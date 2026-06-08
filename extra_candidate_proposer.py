#!/usr/bin/env python3
"""Propose conservative DEL/INS candidates from ReadGraphSV extra evidence."""

import argparse
import csv
import logging
import math
import os
import statistics
import sys
from collections import defaultdict


OUTPUT_FIELDS = [
    "chrom",
    "start",
    "end",
    "svtype",
    "svlen",
    "median_svlen",
    "support",
    "source",
    "mean_pos",
    "mean_size",
]

SOFTCLIP_TYPES = {"SOFTCLIP_LEFT", "SOFTCLIP_RIGHT"}
VALID_SVTYPES = {"DEL", "INS"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", required=True, help="Input extra_signals.tsv")
    parser.add_argument("--out", required=True, help="Output extra_candidates.tsv")
    parser.add_argument("--window", type=int, default=500, help="Maximum adjacent evidence distance in a cluster")
    parser.add_argument("--min_support", type=int, default=2, help="Minimum evidence count per cluster")
    parser.add_argument(
        "--min_softclip_support",
        type=int,
        default=None,
        help="Minimum evidence count for SOFTCLIP-derived clusters; default 10 unless --min_support is explicit",
    )
    parser.add_argument(
        "--min_sa_support",
        type=int,
        default=None,
        help="Minimum evidence count for SA_CONNECTION-derived clusters; default 2 unless --min_support is explicit",
    )
    parser.add_argument(
        "--min_supplementary_support",
        type=int,
        default=None,
        help="Minimum evidence count for SUPPLEMENTARY-derived clusters; default 2 unless --min_support is explicit",
    )
    parser.add_argument("--min_size", type=int, default=50, help="Minimum absolute candidate size")
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    min_support_explicit = "--min_support" in argv_list
    fallback_support = args.min_support if min_support_explicit else None
    if args.min_softclip_support is None:
        args.min_softclip_support = fallback_support if fallback_support is not None else 10
    if args.min_sa_support is None:
        args.min_sa_support = fallback_support if fallback_support is not None else 2
    if args.min_supplementary_support is None:
        args.min_supplementary_support = fallback_support if fallback_support is not None else 2
    return args


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


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


def parse_int(value, default=None):
    parsed = parse_float(value, default=None)
    if parsed is None:
        return default
    return int(round(parsed))


def first_value(row, names, default=""):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def row_chrom(row):
    return str(first_value(row, ["src_chrom", "chrom", "CHROM"]))


def row_event_pos(row):
    return parse_int(first_value(row, ["event_pos", "pos", "start", "src_start"], default=""))


def row_event_len(row):
    return parse_int(first_value(row, ["event_len", "svlen", "SVLEN", "size"], default=""))


def softclip_length(row, evidence_type):
    length = row_event_len(row)
    if length is not None and length > 0:
        return length
    if evidence_type == "SOFTCLIP_LEFT":
        return parse_int(row.get("softclip_left"))
    if evidence_type == "SOFTCLIP_RIGHT":
        return parse_int(row.get("softclip_right"))
    return None


def make_proposal(chrom, position, svtype, size, source):
    if not chrom or position is None or size is None or size <= 0:
        return None
    return {
        "chrom": str(chrom),
        "position": int(position),
        "svtype": str(svtype),
        "size": int(size),
        "source": str(source),
    }


def proposal_from_softclip(row, min_size):
    evidence_type = str(row.get("evidence_type", ""))
    if evidence_type not in SOFTCLIP_TYPES:
        return None
    chrom = row_chrom(row)
    position = row_event_pos(row)
    size = softclip_length(row, evidence_type)
    if size is None or size < min_size:
        return None
    return make_proposal(chrom, position, "INS", size, "SOFTCLIP")


def proposal_from_sa(row, min_size):
    if str(row.get("evidence_type", "")) != "SA_CONNECTION":
        return None

    src_chrom = str(first_value(row, ["src_chrom", "chrom"], default=""))
    dst_chrom = str(row.get("dst_chrom", ""))
    if not src_chrom or not dst_chrom or src_chrom != dst_chrom:
        return None
    if parse_int(row.get("orientation_change"), default=0) != 0:
        return None

    src_start = parse_int(row.get("src_start"))
    src_end = parse_int(row.get("src_end"))
    dst_start = parse_int(row.get("dst_start"))
    dst_end = parse_int(row.get("dst_end"))
    if None in {src_start, src_end, dst_start, dst_end}:
        logging.warning("Skipping SA_CONNECTION with incomplete breakpoints: %s", row.get("read_name", ""))
        return None

    forward_gap = dst_start - src_end
    reverse_gap = src_start - dst_end
    if forward_gap >= min_size:
        return make_proposal(src_chrom, src_end, "DEL", forward_gap, "SA")
    if reverse_gap >= min_size:
        return make_proposal(src_chrom, dst_end, "DEL", reverse_gap, "SA")
    return None


def proposal_from_supplementary(row, min_size):
    if str(row.get("evidence_type", "")) != "SUPPLEMENTARY":
        return None

    svtype = str(first_value(row, ["svtype", "SVTYPE"], default="")).upper()
    if svtype not in VALID_SVTYPES:
        return None
    chrom = row_chrom(row)
    position = row_event_pos(row)
    size = row_event_len(row)
    if size is None or abs(size) < min_size:
        return None
    return make_proposal(chrom, position, svtype, abs(size), "SUPPLEMENTARY")


def proposal_from_row(row, min_size):
    evidence_type = str(row.get("evidence_type", ""))
    if evidence_type in SOFTCLIP_TYPES:
        return proposal_from_softclip(row, min_size)
    if evidence_type == "SA_CONNECTION":
        return proposal_from_sa(row, min_size)
    if evidence_type == "SUPPLEMENTARY":
        return proposal_from_supplementary(row, min_size)
    return None


def read_extra_rows(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.warning("Extra evidence file is empty or missing: %s", path)
        return [], []

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def collect_proposals(path, min_size):
    fieldnames, rows = read_extra_rows(path)
    if not fieldnames:
        logging.warning("Extra evidence file has no header: %s", path)
        return []
    if "evidence_type" not in fieldnames:
        logging.warning("Extra evidence file lacks evidence_type column; no proposals can be made")
        return []

    proposals = []
    skipped = 0
    for row in rows:
        proposal = proposal_from_row(row, min_size)
        if proposal is None:
            skipped += 1
            continue
        proposals.append(proposal)
    logging.info("Converted %d extra evidence rows into %d proposals; skipped %d", len(rows), len(proposals), skipped)
    return proposals


def cluster_proposals(proposals, window):
    groups = defaultdict(list)
    for proposal in proposals:
        groups[(proposal["chrom"], proposal["svtype"])].append(proposal)

    clusters = []
    for (_chrom, _svtype), group in groups.items():
        group = sorted(group, key=lambda item: (item["position"], item["size"], item["source"]))
        current = []
        last_pos = None
        for proposal in group:
            pos = proposal["position"]
            if current and abs(pos - last_pos) > window:
                clusters.append(current)
                current = []
            current.append(proposal)
            last_pos = pos
        if current:
            clusters.append(current)
    return clusters


def round_int(value):
    return int(round(float(value)))


def summarize_cluster(cluster):
    chrom = cluster[0]["chrom"]
    svtype = cluster[0]["svtype"]
    positions = [proposal["position"] for proposal in cluster]
    sizes = [proposal["size"] for proposal in cluster]
    sources = sorted({proposal["source"] for proposal in cluster})

    start = round_int(statistics.median(positions))
    size = max(1, abs(round_int(statistics.median(sizes))))
    svlen = -size if svtype == "DEL" else size
    end = start + size if svtype == "DEL" else start + 1
    source = sources[0] if len(sources) == 1 else "MIXED_EXTRA"
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "svtype": svtype,
        "svlen": svlen,
        "median_svlen": svlen,
        "support": len(cluster),
        "source": source,
        "mean_pos": round(sum(positions) / len(positions), 4),
        "mean_size": round(sum(sizes) / len(sizes), 4),
    }


def cluster_support_threshold(cluster, min_support, min_softclip_support, min_sa_support, min_supplementary_support):
    thresholds = []
    for source in {proposal["source"] for proposal in cluster}:
        if source == "SOFTCLIP":
            thresholds.append(min_softclip_support)
        elif source == "SA":
            thresholds.append(min_sa_support)
        elif source == "SUPPLEMENTARY":
            thresholds.append(min_supplementary_support)
        else:
            thresholds.append(min_support)
    return max(thresholds) if thresholds else min_support


def build_candidates(
    proposals,
    window,
    min_support,
    min_size,
    min_softclip_support,
    min_sa_support,
    min_supplementary_support,
):
    candidates = []
    for cluster in cluster_proposals(proposals, window):
        required_support = cluster_support_threshold(
            cluster,
            min_support,
            min_softclip_support,
            min_sa_support,
            min_supplementary_support,
        )
        if len(cluster) < required_support:
            continue
        candidate = summarize_cluster(cluster)
        if abs(int(candidate["svlen"])) < min_size:
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda row: (row["chrom"], int(row["start"]), row["svtype"], abs(int(row["svlen"]))))
    return candidates


def write_candidates(candidates, path):
    ensure_parent_dir(path)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    proposals = collect_proposals(args.extra, args.min_size)
    candidates = build_candidates(
        proposals,
        args.window,
        args.min_support,
        args.min_size,
        args.min_softclip_support,
        args.min_sa_support,
        args.min_supplementary_support,
    )
    write_candidates(candidates, args.out)
    logging.info("Wrote %d extra evidence candidate proposals to %s", len(candidates), args.out)


if __name__ == "__main__":
    main()
