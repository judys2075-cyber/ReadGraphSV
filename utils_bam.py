#!/usr/bin/env python3
"""BAM and CIGAR helpers for ReadGraphSV."""

import logging
import os

import pysam


CIGAR_M = 0
CIGAR_I = 1
CIGAR_D = 2
CIGAR_N = 3
CIGAR_S = 4
CIGAR_H = 5
CIGAR_P = 6
CIGAR_EQ = 7
CIGAR_X = 8

QUERY_AND_REF_OPS = {CIGAR_M, CIGAR_EQ, CIGAR_X}


def open_alignment_file(path):
    """Open BAM/CRAM/SAM with a small amount of format tolerance."""
    mode = "r" if path.lower().endswith(".sam") else "rb"
    try:
        return pysam.AlignmentFile(path, mode)
    except ValueError:
        logging.warning("Falling back to text SAM mode for %s", path)
        return pysam.AlignmentFile(path, "r")


def iter_bam_records(path):
    """Yield records in file order without requiring a BAM index."""
    with open_alignment_file(path) as bam:
        for record in bam.fetch(until_eof=True):
            yield record


def get_read_length(record):
    """Return query length with robust fallbacks."""
    if record.query_length is not None:
        return int(record.query_length)
    inferred = record.infer_query_length(always=True)
    if inferred is not None:
        return int(inferred)
    if record.query_sequence is not None:
        return len(record.query_sequence)
    return 0


def get_softclip_lengths(record):
    """Return left and right soft-clip lengths from CIGAR."""
    if not record.cigartuples:
        return 0, 0
    left = record.cigartuples[0][1] if record.cigartuples[0][0] == CIGAR_S else 0
    right = record.cigartuples[-1][1] if record.cigartuples[-1][0] == CIGAR_S else 0
    return int(left), int(right)


def has_sa_tag(record):
    """Return 1 if the alignment carries an SA tag, else 0."""
    try:
        return 1 if record.has_tag("SA") else 0
    except (KeyError, ValueError):
        return 0


def chrom_sort_key(chrom):
    """Sort chromosomes in a human-friendly order while remaining generic."""
    name = str(chrom)
    clean = name[3:] if name.lower().startswith("chr") else name
    special = {"x": 23, "y": 24, "m": 25, "mt": 25}
    lower = clean.lower()
    if clean.isdigit():
        return 0, int(clean), name
    if lower in special:
        return 0, special[lower], name
    return 1, clean, name


def extract_cigar_indel_events(record, min_size):
    """Extract large CIGAR DEL/INS events from one alignment record.

    Coordinates are 0-based. Deletions are represented as half-open
    reference intervals [event_pos, event_end). Insertions are represented
    at the current reference position as [event_pos, event_pos + 1).
    """
    if record.is_unmapped or record.is_secondary:
        return []
    if not record.cigartuples or record.reference_start is None:
        return []

    chrom = record.reference_name
    if chrom is None:
        return []

    read_name = record.query_name or ""
    read_len = get_read_length(record)
    softclip_left, softclip_right = get_softclip_lengths(record)
    has_sa = has_sa_tag(record)
    strand = "-" if record.is_reverse else "+"
    mapq = int(record.mapping_quality or 0)
    cigar = record.cigarstring or ""

    read_pos = 0
    ref_pos = int(record.reference_start)
    events = []

    for op, length in record.cigartuples:
        length = int(length)
        if op in QUERY_AND_REF_OPS:
            read_pos += length
            ref_pos += length
        elif op == CIGAR_I:
            if length >= min_size:
                events.append(
                    {
                        "read_name": read_name,
                        "chrom": chrom,
                        "event_pos": ref_pos,
                        "event_end": ref_pos + 1,
                        "svtype": "INS",
                        "svlen": length,
                        "mapq": mapq,
                        "strand": strand,
                        "read_len": read_len,
                        "read_event_start": read_pos,
                        "read_event_end": read_pos + length,
                        "is_supplementary": 1 if record.is_supplementary else 0,
                        "has_sa": has_sa,
                        "softclip_left": softclip_left,
                        "softclip_right": softclip_right,
                        "cigar": cigar,
                    }
                )
            read_pos += length
        elif op == CIGAR_D:
            if length >= min_size:
                events.append(
                    {
                        "read_name": read_name,
                        "chrom": chrom,
                        "event_pos": ref_pos,
                        "event_end": ref_pos + length,
                        "svtype": "DEL",
                        "svlen": length,
                        "mapq": mapq,
                        "strand": strand,
                        "read_len": read_len,
                        "read_event_start": read_pos,
                        "read_event_end": read_pos,
                        "is_supplementary": 1 if record.is_supplementary else 0,
                        "has_sa": has_sa,
                        "softclip_left": softclip_left,
                        "softclip_right": softclip_right,
                        "cigar": cigar,
                    }
                )
            ref_pos += length
        elif op == CIGAR_N:
            ref_pos += length
        elif op == CIGAR_S:
            read_pos += length
        elif op in {CIGAR_H, CIGAR_P}:
            continue
        else:
            logging.debug("Ignoring unsupported CIGAR op %s in %s", op, read_name)

    return events


def ensure_parent_dir(path):
    """Create the parent directory for a file path if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
