#!/usr/bin/env python3
"""Prepare DEL/INS caller VCFs, run Truvari, and summarize benchmark metrics."""

import argparse
import csv
import gzip
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path


GT_FORMAT_HEADER = '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
SUMMARY_FIELDS = ["Tool", "Precision", "Recall", "F1", "TP-comp", "TP-base", "FP", "FN"]
CALLER_ARGS = [
    ("readgraphsv", "ReadGraphSV", "readgraphsv_vcf"),
    ("sniffles2", "Sniffles2", "sniffles2_vcf"),
    ("cutesv", "cuteSV", "cutesv_vcf"),
    ("svim", "SVIM", "svim_vcf"),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Truth VCF for Truvari bench")
    parser.add_argument("--bed", required=True, help="Confident-region BED for Truvari --includebed")
    parser.add_argument("--outdir", required=True, help="Benchmark output directory")
    parser.add_argument("--chrom", default="21", help="Chromosome to evaluate, e.g. 21 or chr21")
    parser.add_argument("--refdist", type=int, default=500, help="Truvari --refdist")
    parser.add_argument("--pctsize", type=float, default=0.5, help="Truvari --pctsize")
    parser.add_argument("--sizemin", type=int, default=50, help="Minimum absolute SVLEN")
    parser.add_argument("--pctseq", type=float, default=0.0, help="Truvari --pctseq")
    parser.add_argument("--readgraphsv-vcf", dest="readgraphsv_vcf", help="ReadGraphSV VCF")
    parser.add_argument("--sniffles2-vcf", dest="sniffles2_vcf", help="Sniffles2 VCF")
    parser.add_argument("--cutesv-vcf", dest="cutesv_vcf", help="cuteSV VCF")
    parser.add_argument("--svim-vcf", dest="svim_vcf", help="SVIM VCF")
    return parser.parse_args(argv)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def require_file(path, label):
    if not path:
        raise ValueError(f"Missing required {label}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_tool(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found on PATH: {name}")


def run_command(command):
    logging.info("Command: %s", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def open_text(path):
    text_path = str(path)
    if text_path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def parse_info(info_text):
    info = {}
    for item in str(info_text).split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = True
    return info


def infer_svtype(alt, info):
    svtype = str(info.get("SVTYPE", "")).upper()
    if svtype:
        return svtype
    alt_text = str(alt).upper()
    if alt_text in {"<DEL>", "DEL"}:
        return "DEL"
    if alt_text in {"<INS>", "INS"}:
        return "INS"
    return ""


def parse_svlen(info, pos):
    svlen_text = str(info.get("SVLEN", "")).split(",")[0]
    try:
        return abs(int(float(svlen_text)))
    except ValueError:
        pass
    try:
        return abs(int(float(info.get("END", pos))) - int(float(pos)))
    except (TypeError, ValueError):
        return 0


def chrom_aliases(chrom):
    text = str(chrom)
    clean = text[3:] if text.lower().startswith("chr") else text
    return {clean, f"chr{clean}"}


def has_gt_header(header_lines):
    return any(line.startswith("##FORMAT=<ID=GT,") for line in header_lines)


def write_header_with_gt(header_lines, handle):
    needs_gt = not has_gt_header(header_lines)
    inserted = False
    for line in header_lines:
        if needs_gt and not inserted and line.startswith("#CHROM"):
            print(GT_FORMAT_HEADER, file=handle)
            inserted = True
        handle.write(line)
    if needs_gt and not inserted:
        print(GT_FORMAT_HEADER, file=handle)


def bgzip_file(input_path, output_path):
    ensure_dir(Path(output_path).parent)
    with open(output_path, "wb") as output_handle:
        subprocess.run(["bgzip", "-c", str(input_path)], check=True, stdout=output_handle)


def tabix_vcf(path):
    run_command(["tabix", "-f", "-p", "vcf", str(path)])


def filter_delins_vcf(input_vcf, output_vcf, chrom, sizemin):
    allowed_chroms = chrom_aliases(chrom)
    header_lines = []
    with open_text(input_vcf) as handle:
        for line in handle:
            if line.startswith("#"):
                header_lines.append(line)

    kept = 0
    with open(output_vcf, "w") as output_handle:
        write_header_with_gt(header_lines, output_handle)
        with open_text(input_vcf) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 8:
                    continue
                chrom_value, pos, _record_id, _ref, alt, _qual, _filt, info_text = fields[:8]
                info = parse_info(info_text)
                svtype = infer_svtype(alt, info)
                svlen = parse_svlen(info, pos)
                if chrom_value in allowed_chroms and svtype in {"DEL", "INS"} and svlen >= sizemin:
                    output_handle.write(line)
                    kept += 1

    logging.info(
        "Filtered %d %s DEL/INS records with SVLEN >= %d into %s",
        kept,
        chrom,
        sizemin,
        output_vcf,
    )
    return kept


def prepare_caller_vcf(tool_name, input_vcf, outdir, chrom, sizemin):
    benchmark_inputs = Path(outdir) / "benchmark_inputs"
    ensure_dir(benchmark_inputs)
    sorted_vcf = benchmark_inputs / f"{tool_name}.sorted.tmp.vcf.gz"
    filtered_vcf = benchmark_inputs / f"{tool_name}_DELINS_{sizemin}.vcf"
    filtered_gz = benchmark_inputs / f"{tool_name}_DELINS_{sizemin}.vcf.gz"

    run_command(["bcftools", "sort", "-Oz", "-o", str(sorted_vcf), str(input_vcf)])
    tabix_vcf(sorted_vcf)
    filter_delins_vcf(sorted_vcf, filtered_vcf, chrom=chrom, sizemin=sizemin)
    bgzip_file(filtered_vcf, filtered_gz)
    tabix_vcf(filtered_gz)

    for temporary in [sorted_vcf, Path(f"{sorted_vcf}.tbi"), filtered_vcf]:
        if temporary.exists():
            temporary.unlink()
    return filtered_gz


def run_truvari(tool_name, candidate_vcf, args):
    output_dir = Path(args.outdir) / f"truvari_{tool_name}"
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


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def read_truvari_summary(path):
    with open(path) as handle:
        data = json.load(handle)
    return {
        "Precision": to_float(data.get("precision")),
        "Recall": to_float(data.get("recall")),
        "F1": to_float(data.get("f1")),
        "TP-comp": to_int(data.get("TP-comp")),
        "TP-base": to_int(data.get("TP-base")),
        "FP": to_int(data.get("FP")),
        "FN": to_int(data.get("FN")),
    }


def write_summary_tsv(rows, output_path):
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_input_callers(args):
    callers = []
    for tool_name, label, attr_name in CALLER_ARGS:
        path = getattr(args, attr_name)
        if path:
            callers.append((tool_name, label, path))
    return callers


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    require_file(args.truth, "truth VCF")
    require_file(args.bed, "BED")
    for command in ["bcftools", "bgzip", "tabix", "truvari"]:
        require_tool(command)
    ensure_dir(args.outdir)

    callers = collect_input_callers(args)
    if not callers:
        raise ValueError("At least one caller VCF must be provided")

    rows = []
    for tool_name, label, input_vcf in callers:
        require_file(input_vcf, f"{label} VCF")
        logging.info("Benchmarking %s", label)
        prepared_vcf = prepare_caller_vcf(tool_name, input_vcf, args.outdir, args.chrom, args.sizemin)
        summary_path = run_truvari(tool_name, prepared_vcf, args)
        rows.append({"Tool": label, **read_truvari_summary(summary_path)})

    output_path = Path(args.outdir) / "summary.tsv"
    write_summary_tsv(rows, output_path)
    logging.info("Wrote benchmark summary to %s", output_path)


if __name__ == "__main__":
    main()
