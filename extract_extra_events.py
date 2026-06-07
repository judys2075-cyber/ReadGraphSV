#!/usr/bin/env python3
"""Extract soft-clip, SA-tag, and supplementary-alignment SV evidence from BAM."""

import argparse
import csv
import logging
import re

from tqdm import tqdm

from utils_bam import (
    chrom_sort_key,
    ensure_parent_dir,
    has_sa_tag,
    iter_bam_records,
)


FIELDS = [
    "read_name",
    "evidence_type",
    "src_chrom",
    "src_start",
    "src_end",
    "src_strand",
    "dst_chrom",
    "dst_start",
    "dst_end",
    "dst_strand",
    "event_pos",
    "event_len",
    "mapq",
    "dst_mapq",
    "chrom_change",
    "orientation_change",
    "is_supplementary",
    "has_sa",
    "softclip_left",
    "softclip_right",
    "cigar",
    "sa_cigar",
    "nm",
    "source",
]

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
REF_CONSUMING_OPS = {"M", "D", "N", "=", "X"}
CIGAR_S = 4
CIGAR_H = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", required=True, help="Input BAM/CRAM/SAM file")
    parser.add_argument("--min_clip", type=int, default=50, help="Minimum soft-clip length")
    parser.add_argument("--out", required=True, help="Output extra_signals.tsv")
    return parser.parse_args()


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def reference_span_from_cigar(cigar):
    """Return reference-consuming span for a SAM CIGAR string."""
    if not cigar or cigar == "*":
        return 0
    span = 0
    consumed = 0
    for length, op in CIGAR_RE.findall(cigar):
        consumed += len(length) + len(op)
        if op in REF_CONSUMING_OPS:
            span += int(length)
    if consumed != len(cigar):
        logging.debug("Could not fully parse CIGAR string: %s", cigar)
    return span


def get_terminal_softclip_lengths(record):
    """Return terminal soft clips, allowing outer hard clips."""
    if not record.cigartuples:
        return 0, 0

    left_index = 1 if record.cigartuples[0][0] == CIGAR_H and len(record.cigartuples) > 1 else 0
    right_index = -2 if record.cigartuples[-1][0] == CIGAR_H and len(record.cigartuples) > 1 else -1

    left = record.cigartuples[left_index][1] if record.cigartuples[left_index][0] == CIGAR_S else 0
    right = record.cigartuples[right_index][1] if record.cigartuples[right_index][0] == CIGAR_S else 0
    return int(left), int(right)


def record_nm(record):
    try:
        return record.get_tag("NM") if record.has_tag("NM") else ""
    except (KeyError, ValueError):
        return ""


def record_context(record):
    start = safe_int(record.reference_start)
    end = safe_int(record.reference_end, start)
    softclip_left, softclip_right = get_terminal_softclip_lengths(record)
    return {
        "read_name": record.query_name or "",
        "src_chrom": record.reference_name or "",
        "src_start": start,
        "src_end": end,
        "src_strand": "-" if record.is_reverse else "+",
        "mapq": safe_int(record.mapping_quality),
        "is_supplementary": 1 if record.is_supplementary else 0,
        "has_sa": has_sa_tag(record),
        "softclip_left": softclip_left,
        "softclip_right": softclip_right,
        "cigar": record.cigarstring or "",
        "nm": record_nm(record),
    }


def base_row(record, evidence_type, event_pos, event_len, source):
    ctx = record_context(record)
    return {
        "read_name": ctx["read_name"],
        "evidence_type": evidence_type,
        "src_chrom": ctx["src_chrom"],
        "src_start": ctx["src_start"],
        "src_end": ctx["src_end"],
        "src_strand": ctx["src_strand"],
        "dst_chrom": "",
        "dst_start": "",
        "dst_end": "",
        "dst_strand": "",
        "event_pos": safe_int(event_pos),
        "event_len": safe_int(event_len),
        "mapq": ctx["mapq"],
        "dst_mapq": "",
        "chrom_change": 0,
        "orientation_change": 0,
        "is_supplementary": ctx["is_supplementary"],
        "has_sa": ctx["has_sa"],
        "softclip_left": ctx["softclip_left"],
        "softclip_right": ctx["softclip_right"],
        "cigar": ctx["cigar"],
        "sa_cigar": "",
        "nm": ctx["nm"],
        "source": source,
    }


def extract_softclip_events(record, min_clip):
    if not record.cigartuples or record.reference_start is None:
        return []

    ctx = record_context(record)
    events = []
    if ctx["softclip_left"] >= min_clip:
        events.append(
            base_row(
                record,
                evidence_type="SOFTCLIP_LEFT",
                event_pos=ctx["src_start"],
                event_len=ctx["softclip_left"],
                source="CIGAR_SOFTCLIP",
            )
        )
    if ctx["softclip_right"] >= min_clip:
        events.append(
            base_row(
                record,
                evidence_type="SOFTCLIP_RIGHT",
                event_pos=ctx["src_end"],
                event_len=ctx["softclip_right"],
                source="CIGAR_SOFTCLIP",
            )
        )
    return events


def parse_sa_entries(record):
    try:
        sa_tag = record.get_tag("SA")
    except (KeyError, ValueError):
        return []

    entries = []
    for raw_entry in str(sa_tag).strip().strip(";").split(";"):
        if not raw_entry:
            continue
        parts = raw_entry.split(",")
        if len(parts) < 6:
            logging.debug("Skipping malformed SA entry for %s: %s", record.query_name, raw_entry)
            continue
        chrom, pos, strand, cigar, mapq, nm = parts[:6]
        dst_start = max(0, safe_int(pos, 1) - 1)
        dst_end = dst_start + reference_span_from_cigar(cigar)
        entries.append(
            {
                "chrom": chrom,
                "start": dst_start,
                "end": dst_end,
                "strand": strand,
                "cigar": cigar,
                "mapq": safe_int(mapq),
                "nm": safe_int(nm),
            }
        )
    return entries


def extract_sa_connection_events(record):
    """Extract SA connections from primary alignments to supplementary segments."""
    if record.is_supplementary or not has_sa_tag(record):
        return []

    ctx = record_context(record)
    events = []
    for entry in parse_sa_entries(record):
        chrom_change = 1 if ctx["src_chrom"] != entry["chrom"] else 0
        orientation_change = 1 if ctx["src_strand"] != entry["strand"] else 0
        event_len = 0 if chrom_change else abs(entry["start"] - ctx["src_start"])
        row = base_row(
            record,
            evidence_type="SA_CONNECTION",
            event_pos=ctx["src_start"],
            event_len=event_len,
            source="SA_TAG",
        )
        row.update(
            {
                "dst_chrom": entry["chrom"],
                "dst_start": entry["start"],
                "dst_end": entry["end"],
                "dst_strand": entry["strand"],
                "dst_mapq": entry["mapq"],
                "chrom_change": chrom_change,
                "orientation_change": orientation_change,
                "sa_cigar": entry["cigar"],
                "nm": entry["nm"],
            }
        )
        events.append(row)
    return events


def extract_supplementary_event(record):
    if not record.is_supplementary:
        return []
    ctx = record_context(record)
    return [
        base_row(
            record,
            evidence_type="SUPPLEMENTARY",
            event_pos=ctx["src_start"],
            event_len=max(0, ctx["src_end"] - ctx["src_start"]),
            source="BAM_SUPPLEMENTARY_FLAG",
        )
    ]


def extract_extra_events(record, min_clip):
    if record.is_unmapped or record.is_secondary:
        return []
    if record.reference_start is None or record.reference_name is None:
        return []
    events = []
    events.extend(extract_softclip_events(record, min_clip))
    events.extend(extract_sa_connection_events(record))
    events.extend(extract_supplementary_event(record))
    return events


def sort_key(row):
    return (
        chrom_sort_key(row.get("src_chrom", "")),
        safe_int(row.get("event_pos")),
        str(row.get("evidence_type", "")),
        str(row.get("read_name", "")),
    )


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
        events.extend(extract_extra_events(record, args.min_clip))

    events.sort(key=sort_key)

    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for event in events:
            writer.writerow(event)

    logging.info("Scanned %d records, kept %d primary/supplementary records", total_records, usable_records)
    logging.info("Wrote %d extra evidence signals to %s", len(events), args.out)


if __name__ == "__main__":
    main()
