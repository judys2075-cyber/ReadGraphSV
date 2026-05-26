#!/usr/bin/env python3
"""Label candidate DEL/INS events by matching them to a truth VCF."""

import argparse
import logging
import os

import pandas as pd

from utils_vcf import parse_truth_vcf, size_similarity


ADDED_FIELDS = [
    "label",
    "matched_truth_id",
    "matched_truth_pos",
    "matched_truth_svlen",
    "size_similarity",
    "pos_distance",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Input candidates.tsv")
    parser.add_argument("--truth", required=True, help="Truth VCF with DEL/INS records")
    parser.add_argument("--max_dist", type=int, default=500, help="Maximum breakpoint distance")
    parser.add_argument("--min_size_sim", type=float, default=0.7, help="Minimum size similarity")
    parser.add_argument("--out", required=True, help="Output candidates_labeled.tsv")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def label_one_candidate(row, truth_by_key, max_dist, min_size_sim):
    chrom = str(row["chrom"])
    svtype = str(row["svtype"])
    start = int(row["start"])
    candidate_svlen = abs(float(row["median_svlen"]))

    best = None
    for truth in truth_by_key.get((chrom, svtype), []):
        pos_distance = abs(start - int(truth["pos"]))
        if pos_distance > max_dist:
            continue
        sim = size_similarity(candidate_svlen, truth["svlen"])
        if sim < min_size_sim:
            continue
        key = (pos_distance, -sim)
        if best is None or key < best[0]:
            best = (key, truth, sim, pos_distance)

    if best is None:
        return {
            "label": 0,
            "matched_truth_id": ".",
            "matched_truth_pos": -1,
            "matched_truth_svlen": -1,
            "size_similarity": 0.0,
            "pos_distance": -1,
        }

    _key, truth, sim, pos_distance = best
    return {
        "label": 1,
        "matched_truth_id": truth["truth_id"],
        "matched_truth_pos": int(truth["pos"]),
        "matched_truth_svlen": int(truth["svlen"]),
        "size_similarity": round(float(sim), 6),
        "pos_distance": int(pos_distance),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    if os.path.exists(args.candidates):
        try:
            candidates = pd.read_csv(args.candidates, sep="\t")
        except pd.errors.EmptyDataError:
            candidates = pd.DataFrame()
    else:
        candidates = pd.DataFrame()
    if candidates.empty:
        logging.warning("No candidates found in %s", args.candidates)
        for field in ADDED_FIELDS:
            candidates[field] = []
        candidates.to_csv(args.out, sep="\t", index=False)
        return

    required = {"chrom", "start", "svtype", "median_svlen"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate file is missing required columns: {sorted(missing)}")

    truth = parse_truth_vcf(args.truth, allowed_types=("DEL", "INS"))
    truth_by_key = {}
    for record in truth:
        truth_by_key.setdefault((record["chrom"], record["svtype"]), []).append(record)

    labels = [
        label_one_candidate(row, truth_by_key, args.max_dist, args.min_size_sim)
        for _, row in candidates.iterrows()
    ]
    label_df = pd.DataFrame(labels)
    out_df = pd.concat([candidates.reset_index(drop=True), label_df], axis=1)
    out_df.to_csv(args.out, sep="\t", index=False)

    positives = int(out_df["label"].sum()) if not out_df.empty else 0
    logging.info("Labeled %d candidates: %d positive, %d negative", len(out_df), positives, len(out_df) - positives)
    logging.info("Wrote labeled candidates to %s", args.out)


if __name__ == "__main__":
    main()
