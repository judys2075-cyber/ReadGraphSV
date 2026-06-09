#!/usr/bin/env python3
"""Export threshold-specific ReadGraphSV VCFs and benchmark them with Truvari."""

import argparse
import csv
import logging
import os
import shutil
import subprocess
from pathlib import Path

try:
    from benchmark_truvari_delins import GT_FORMAT_HEADER, read_truvari_summary, require_file, require_tool
except ImportError:  # pragma: no cover - used when imported as scripts.truvari_threshold_sweep
    from scripts.benchmark_truvari_delins import (
        GT_FORMAT_HEADER,
        read_truvari_summary,
        require_file,
        require_tool,
    )


SWEEP_FIELDS = ["Threshold", "Precision", "Recall", "F1", "TP-comp", "TP-base", "FP", "FN", "VCF"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="ReadGraphSV predictions_v2.tsv")
    parser.add_argument("--truth", required=True, help="Truth VCF for Truvari bench")
    parser.add_argument("--bed", required=True, help="Confident-region BED for Truvari --includebed")
    parser.add_argument("--outdir", required=True, help="Threshold sweep output directory")
    parser.add_argument("--chrom", default="21", help="Chromosome to export, e.g. 21 or chr21")
    parser.add_argument("--contig-length", type=int, default=None, help="Optional VCF contig length")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated probability thresholds",
    )
    parser.add_argument("--refdist", type=int, default=500, help="Truvari --refdist")
    parser.add_argument("--pctsize", type=float, default=0.5, help="Truvari --pctsize")
    parser.add_argument("--sizemin", type=int, default=50, help="Truvari --sizemin")
    parser.add_argument("--pctseq", type=float, default=0.0, help="Truvari --pctseq")
    return parser.parse_args(argv)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_command(command):
    logging.info("Command: %s", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def parse_thresholds(text):
    thresholds = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("No thresholds were provided")
    return thresholds


def chrom_aliases(chrom):
    text = str(chrom)
    clean = text[3:] if text.lower().startswith("chr") else text
    return {clean, f"chr{clean}"}


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
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_predictions(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def estimate_svlen(row):
    median_svlen = abs(int(round(numeric(row.get("median_svlen", 0), 0))))
    if median_svlen > 0:
        return median_svlen
    start = int(round(numeric(row.get("start", 0), 0)))
    end = int(round(numeric(row.get("end", start + 1), start + 1)))
    return max(1, abs(end - start))


def write_vcf_header(handle, chrom, contig_length=None):
    print("##fileformat=VCFv4.2", file=handle)
    print("##source=ReadGraphSV_v0.2_threshold_sweep", file=handle)
    if contig_length:
        print(f"##contig=<ID={chrom},length={int(contig_length)}>", file=handle)
    print('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">', file=handle)
    print('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the variant">', file=handle)
    print('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">', file=handle)
    print('##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Supporting read count">', file=handle)
    print('##INFO=<ID=GNN_PROB,Number=1,Type=Float,Description="GNN confidence score">', file=handle)
    print('##ALT=<ID=DEL,Description="Deletion">', file=handle)
    print('##ALT=<ID=INS,Description="Insertion">', file=handle)
    print(GT_FORMAT_HEADER, file=handle)
    print("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tReadGraphSV", file=handle)


def write_threshold_vcf(predictions, output_vcf, threshold, chrom="21", contig_length=None):
    rows = read_predictions(predictions)
    allowed_chroms = chrom_aliases(chrom)
    selected = []
    for row in rows:
        svtype = str(row.get("svtype", ""))
        probability = numeric(row.get("gnn_prob", 0), 0)
        if probability < threshold or svtype not in {"DEL", "INS"}:
            continue
        if str(row.get("chrom", "")) not in allowed_chroms:
            continue
        start = int(round(numeric(row.get("start", 0), 0)))
        end = int(round(numeric(row.get("end", start + 1), start + 1)))
        selected.append((chrom_sort_key(row.get("chrom", "")), start, end, svtype, row))

    selected.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    ensure_dir(Path(output_vcf).parent)
    header_chrom = chrom if str(chrom).lower().startswith("chr") else f"chr{chrom}"
    record_count = 0
    with open(output_vcf, "w") as handle:
        write_vcf_header(handle, header_chrom, contig_length=contig_length)
        for _chrom_key, start, end, svtype, row in selected:
            probability = numeric(row.get("gnn_prob", 0), 0)
            svlen_abs = estimate_svlen(row)
            support = int(round(numeric(row.get("support_read_count", 0), 0)))
            if svtype == "DEL":
                alt = "<DEL>"
                vcf_end = max(end, start + svlen_abs)
                svlen = -abs(svlen_abs)
            else:
                alt = "<INS>"
                vcf_end = start + 1
                svlen = abs(svlen_abs)

            record_count += 1
            info = (
                f"SVTYPE={svtype};END={vcf_end};SVLEN={svlen};"
                f"SUPPORT={support};GNN_PROB={probability:.6f}"
            )
            print(
                f"{row.get('chrom')}\t{start + 1}\tReadGraphSV_t{threshold:.2f}_{record_count}\t"
                f"N\t{alt}\t{probability * 100:.2f}\tPASS\t{info}\tGT\t./.",
                file=handle,
            )
    logging.info("Wrote %d records at threshold %.2f to %s", record_count, threshold, output_vcf)
    return record_count


def bgzip_and_index(vcf_path):
    gz_path = f"{vcf_path}.gz"
    with open(gz_path, "wb") as output_handle:
        subprocess.run(["bgzip", "-c", str(vcf_path)], check=True, stdout=output_handle)
    run_command(["tabix", "-f", "-p", "vcf", gz_path])
    return gz_path


def run_truvari(args, threshold, candidate_vcf):
    output_dir = Path(args.outdir) / f"truvari_t{threshold:.2f}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    command = [
        "truvari",
        "bench",
        "-b",
        args.truth,
        "-c",
        str(candidate_vcf),
        "-o",
        str(output_dir),
        "--includebed",
        args.bed,
        "--passonly",
        "--refdist",
        str(args.refdist),
        "--pctsize",
        str(args.pctsize),
        "--sizemin",
        str(args.sizemin),
        "--pctseq",
        str(args.pctseq),
    ]
    run_command(command)
    return output_dir / "summary.json"


def write_sweep_tsv(rows, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_best_threshold(rows, output_path):
    best = max(rows, key=lambda row: (float(row["F1"]), float(row["Recall"]), float(row["Precision"])))
    with open(output_path, "w") as handle:
        print(f"Best threshold: {float(best['Threshold']):.2f}", file=handle)
        print(f"Precision: {float(best['Precision']):.6f}", file=handle)
        print(f"Recall: {float(best['Recall']):.6f}", file=handle)
        print(f"F1: {float(best['F1']):.6f}", file=handle)
        print(f"TP-comp: {best['TP-comp']}", file=handle)
        print(f"TP-base: {best['TP-base']}", file=handle)
        print(f"FP: {best['FP']}", file=handle)
        print(f"FN: {best['FN']}", file=handle)
    logging.info("Best threshold is %.2f", float(best["Threshold"]))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    require_file(args.predictions, "predictions TSV")
    require_file(args.truth, "truth VCF")
    require_file(args.bed, "BED")
    for command in ["bgzip", "tabix", "truvari"]:
        require_tool(command)

    outdir = Path(args.outdir)
    vcf_dir = outdir / "vcfs"
    ensure_dir(vcf_dir)
    rows = []
    for threshold in parse_thresholds(args.thresholds):
        vcf_path = vcf_dir / f"readgraphsv_t{threshold:.2f}.vcf"
        write_threshold_vcf(
            args.predictions,
            vcf_path,
            threshold=threshold,
            chrom=args.chrom,
            contig_length=args.contig_length,
        )
        vcf_gz = bgzip_and_index(vcf_path)
        summary_path = run_truvari(args, threshold, vcf_gz)
        metrics = read_truvari_summary(summary_path)
        rows.append(
            {
                "Threshold": f"{threshold:.2f}",
                "Precision": f"{metrics['Precision']:.6f}",
                "Recall": f"{metrics['Recall']:.6f}",
                "F1": f"{metrics['F1']:.6f}",
                "TP-comp": metrics["TP-comp"],
                "TP-base": metrics["TP-base"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "VCF": str(vcf_gz),
            }
        )

    write_sweep_tsv(rows, outdir / "threshold_sweep.tsv")
    write_best_threshold(rows, outdir / "best_threshold.txt")


if __name__ == "__main__":
    main()
