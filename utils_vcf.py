#!/usr/bin/env python3
"""Small VCF parser utilities for ReadGraphSV truth labeling."""

import gzip
import logging


def open_text_auto(path):
    """Open plain text or gzip-compressed text files."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_info(info_text):
    """Parse a VCF INFO field into a dictionary."""
    info = {}
    if not info_text or info_text == ".":
        return info
    for item in info_text.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = True
    return info


def first_int(value, default=0):
    """Parse the first integer from a scalar or comma-separated INFO value."""
    if value is None or value is True:
        return default
    text = str(value).split(",")[0]
    try:
        return int(float(text))
    except ValueError:
        return default


def parse_truth_vcf(path, allowed_types=("DEL", "INS")):
    """Read DEL/INS truth records from a VCF.

    Returned positions are 0-based to match BAM-derived event coordinates.
    """
    truth = []
    allowed = set(allowed_types)
    try:
        handle = open_text_auto(path)
    except OSError as exc:
        logging.error("Could not open truth VCF %s: %s", path, exc)
        return truth

    with handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos_text, record_id, _ref, alt, _qual, _filter, info_text = fields[:8]
            info = parse_info(info_text)
            svtype = str(info.get("SVTYPE", "")).split(",")[0]
            if not svtype and alt.startswith("<") and alt.endswith(">"):
                svtype = alt.strip("<>")
            if svtype not in allowed:
                continue

            try:
                pos0 = int(pos_text) - 1
            except ValueError:
                continue

            svlen = abs(first_int(info.get("SVLEN"), 0))
            end = first_int(info.get("END"), 0)
            if svlen == 0 and end > 0:
                # VCF END is 1-based inclusive; as a 0-based half-open end it is
                # already numerically equal to END.
                svlen = abs(end - pos0)
            if svtype == "INS" and svlen == 0 and len(alt) > 1 and not alt.startswith("<"):
                svlen = max(1, len(alt) - 1)
            if svlen <= 0:
                logging.debug("Skipping truth record without usable SVLEN: %s", line.strip())
                continue

            truth.append(
                {
                    "truth_id": record_id if record_id and record_id != "." else f"{chrom}:{pos_text}:{svtype}",
                    "chrom": chrom,
                    "pos": pos0,
                    "svtype": svtype,
                    "svlen": svlen,
                }
            )

    logging.info("Loaded %d DEL/INS truth records from %s", len(truth), path)
    return truth


def size_similarity(size_a, size_b):
    """Return min(size)/max(size), or 0 for invalid sizes."""
    a = abs(float(size_a))
    b = abs(float(size_b))
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)
