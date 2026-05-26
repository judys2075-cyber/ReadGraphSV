#!/usr/bin/env python3
"""Extract large CIGAR DEL/INS signals directly from a long-read BAM."""

import argparse
import csv
import logging

from tqdm import tqdm

from utils_bam import (
    chrom_sort_key,
    ensure_parent_dir,
    extract_cigar_indel_events,
    iter_bam_records,
)


FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True, help="Input BAM/CRAM/SAM file")
    parser.add_argument("--min_size", type=int, default=50, help="Minimum CIGAR I/D length")
    parser.add_argument("--out", required=True, help="Output signals.tsv")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    events = []
    total_records = 0
    usable_records = 0
    logging.info("Scanning BAM without requiring an index: %s", args.bam)
    for record in tqdm(iter_bam_records(args.bam), desc="records", unit="rec"):
        total_records += 1
        if record.is_unmapped or record.is_secondary:
            continue
        usable_records += 1
        events.extend(extract_cigar_indel_events(record, args.min_size))

    events.sort(key=lambda row: (chrom_sort_key(row["chrom"]), int(row["event_pos"]), row["svtype"], row["read_name"]))

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for event in events:
            writer.writerow(event)

    logging.info("Scanned %d records, kept %d primary/supplementary records", total_records, usable_records)
    logging.info("Wrote %d CIGAR DEL/INS signals to %s", len(events), args.out)


if __name__ == "__main__":
    main()
