#!/usr/bin/env python3
"""Cluster nearby CIGAR DEL/INS signals into candidate SVs."""

import argparse
import logging
import os

import numpy as np
import pandas as pd


FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", required=True, help="Input signals.tsv")
    parser.add_argument("--window", type=int, default=500, help="Maximum position distance inside a cluster")
    parser.add_argument("--min_support", type=int, default=2, help="Minimum unique supporting reads")
    parser.add_argument("--out", required=True, help="Output candidates.tsv")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def empty_output(path):
    pd.DataFrame(columns=FIELDS).to_csv(path, sep="\t", index=False)


def len_similarity_ok(length, cluster_lengths, min_ratio=0.5):
    if len(cluster_lengths) == 0:
        return True
    med = abs(float(np.median(cluster_lengths)))
    cur = abs(float(length))
    if med <= 0 or cur <= 0:
        return True
    return min(med, cur) / max(med, cur) >= min_ratio


def summarize_cluster(cluster, candidate_id):
    df = pd.DataFrame(cluster)
    start = int(round(float(np.median(df["event_pos"]))))
    median_svlen = int(round(float(np.median(df["svlen"]))))
    svtype = str(df["svtype"].iloc[0])
    end = start + median_svlen if svtype == "DEL" else start + 1
    read_names = sorted(str(x) for x in df["read_name"].dropna().unique())
    return {
        "candidate_id": candidate_id,
        "chrom": str(df["chrom"].iloc[0]),
        "start": start,
        "end": int(end),
        "svtype": svtype,
        "median_svlen": median_svlen,
        "support_read_count": len(read_names),
        "mean_mapq": round(float(df["mapq"].mean()), 4) if len(df) else 0.0,
        "std_pos": round(float(df["event_pos"].std(ddof=0)), 4) if len(df) > 1 else 0.0,
        "std_svlen": round(float(df["svlen"].std(ddof=0)), 4) if len(df) > 1 else 0.0,
        "read_names": ",".join(read_names),
        "signal_indices": ",".join(str(int(x)) for x in df["signal_index"].tolist()),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    if not os.path.exists(args.signals) or os.path.getsize(args.signals) == 0:
        logging.warning("Signal file is empty or missing: %s", args.signals)
        empty_output(args.out)
        return

    try:
        signals = pd.read_csv(args.signals, sep="\t")
    except pd.errors.EmptyDataError:
        logging.warning("Signal file has no readable rows: %s", args.signals)
        empty_output(args.out)
        return
    if signals.empty:
        logging.warning("No signals found in %s", args.signals)
        empty_output(args.out)
        return

    required = {"chrom", "event_pos", "svtype", "svlen", "read_name", "mapq"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals file is missing required columns: {sorted(missing)}")

    signals = signals.copy()
    signals["signal_index"] = np.arange(len(signals), dtype=int)
    signals["event_pos"] = pd.to_numeric(signals["event_pos"], errors="coerce")
    signals["svlen"] = pd.to_numeric(signals["svlen"], errors="coerce")
    signals["mapq"] = pd.to_numeric(signals["mapq"], errors="coerce").fillna(0)
    signals = signals.dropna(subset=["event_pos", "svlen"])
    signals["event_pos"] = signals["event_pos"].astype(int)
    signals["svlen"] = signals["svlen"].astype(int)

    candidates = []
    candidate_counter = 1
    for (chrom, svtype), group in signals.groupby(["chrom", "svtype"], sort=False):
        group = group.sort_values(["event_pos", "svlen", "read_name"])
        current = []
        current_lengths = []
        last_pos = None
        for row in group.to_dict("records"):
            pos = int(row["event_pos"])
            length = int(row["svlen"])
            starts_new = (
                len(current) > 0
                and (abs(pos - int(last_pos)) > args.window or not len_similarity_ok(length, current_lengths))
            )
            if starts_new:
                read_count = len({str(x["read_name"]) for x in current})
                if read_count >= args.min_support:
                    candidates.append(summarize_cluster(current, f"CAND_{candidate_counter:06d}"))
                    candidate_counter += 1
                current = []
                current_lengths = []

            current.append(row)
            current_lengths.append(length)
            last_pos = pos

        if current:
            read_count = len({str(x["read_name"]) for x in current})
            if read_count >= args.min_support:
                candidates.append(summarize_cluster(current, f"CAND_{candidate_counter:06d}"))
                candidate_counter += 1

    out_df = pd.DataFrame(candidates, columns=FIELDS)
    out_df.to_csv(args.out, sep="\t", index=False)
    logging.info("Clustered %d signals into %d candidates at min_support=%d", len(signals), len(out_df), args.min_support)
    logging.info("Wrote candidates to %s", args.out)


if __name__ == "__main__":
    main()
