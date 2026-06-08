#!/usr/bin/env python3
"""Export GNN-filtered ReadGraphSV predictions to a simple DEL/INS VCF."""

import argparse
import logging
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Input predictions.tsv")
    parser.add_argument("--out", required=True, help="Output filtered.vcf")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum GNN probability")
    parser.add_argument("--sample", default="ReadGraphSV", help="Sample name in the VCF FORMAT column")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def chrom_sort_key(chrom):
    name = str(chrom)
    clean = name[3:] if name.lower().startswith("chr") else name
    special = {"x": 23, "y": 24, "m": 25, "mt": 25}
    lower = clean.lower()
    if clean.isdigit():
        return 0, int(clean), name
    if lower in special:
        return 0, special[lower], name
    return 1, clean, name


def numeric(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_predictions(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def estimate_svlen(row):
    if "median_svlen" in row and not pd.isna(row["median_svlen"]):
        value = abs(int(round(numeric(row["median_svlen"], 0))))
        if value > 0:
            return value

    start = int(round(numeric(row.get("start", 0), 0)))
    end = int(round(numeric(row.get("end", start + 1), start + 1)))
    svtype = str(row.get("svtype", ""))
    if svtype == "DEL":
        return max(1, abs(end - start))
    if svtype == "INS":
        return max(1, abs(end - start))
    return max(1, abs(end - start))


def write_header(handle, sample):
    print("##fileformat=VCFv4.2", file=handle)
    print("##source=ReadGraphSV_v0.1", file=handle)
    print('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">', file=handle)
    print('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">', file=handle)
    print('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">', file=handle)
    print('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Supporting read count">', file=handle)
    print('##INFO=<ID=GNN_PROB,Number=1,Type=Float,Description="GNN confidence score">', file=handle)
    print('##ALT=<ID=DEL,Description="Deletion">', file=handle)
    print('##ALT=<ID=INS,Description="Insertion">', file=handle)
    print('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', file=handle)
    print(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}", file=handle)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    pred = read_predictions(args.pred)
    with open(args.out, "w") as handle:
        write_header(handle, args.sample)

        if pred.empty:
            logging.warning("Prediction file is empty; wrote VCF header only: %s", args.out)
            return

        required = {"chrom", "start", "end", "svtype", "support_read_count", "gnn_prob"}
        missing = required - set(pred.columns)
        if missing:
            raise ValueError(f"prediction file is missing required columns: {sorted(missing)}")

        pred = pred.copy()
        pred["gnn_prob"] = pd.to_numeric(pred["gnn_prob"], errors="coerce").fillna(0.0)
        pred["start"] = pd.to_numeric(pred["start"], errors="coerce")
        pred["end"] = pd.to_numeric(pred["end"], errors="coerce")
        pred = pred.dropna(subset=["start", "end"])
        pred = pred[(pred["gnn_prob"] >= args.threshold) & (pred["svtype"].isin(["DEL", "INS"]))]
        if pred.empty:
            logging.info("No predictions passed threshold %.3f; wrote VCF header only", args.threshold)
            return

        pred["_chrom_key"] = pred["chrom"].map(chrom_sort_key)
        pred = pred.sort_values(["_chrom_key", "start", "end", "svtype"])

        record_count = 0
        for _, row in pred.iterrows():
            svtype = str(row["svtype"])
            start0 = int(round(numeric(row["start"], 0)))
            end0 = int(round(numeric(row["end"], start0 + 1)))
            svlen_abs = estimate_svlen(row)
            prob = numeric(row["gnn_prob"], 0.0)
            support = int(round(numeric(row["support_read_count"], 0)))

            if svtype == "DEL":
                alt = "<DEL>"
                vcf_end = max(end0, start0 + svlen_abs)
                svlen = -abs(svlen_abs)
            elif svtype == "INS":
                alt = "<INS>"
                vcf_end = start0 + 1
                svlen = abs(svlen_abs)
            else:
                continue

            record_count += 1
            chrom = str(row["chrom"])
            pos1 = start0 + 1
            record_id = f"ReadGraphSV_{record_count}"
            qual = f"{prob * 100:.2f}"
            info = (
                f"SVTYPE={svtype};END={vcf_end};SVLEN={svlen};"
                f"SUPPORT={support};GNN_PROB={prob:.6f}"
            )
            print(
                f"{chrom}\t{pos1}\t{record_id}\tN\t{alt}\t{qual}\tPASS\t{info}\tGT\t0/1",
                file=handle,
            )

    logging.info("Wrote %d VCF records to %s", record_count, args.out)


if __name__ == "__main__":
    main()
