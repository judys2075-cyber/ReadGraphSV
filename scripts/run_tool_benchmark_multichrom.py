#!/usr/bin/env python3
"""Run Sniffles2/cuteSV/SVIM and Truvari benchmarks across HG002 chromosomes."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMS = "19,20,21,22"
DEFAULT_TOOLS = "sniffles2,cutesv,svim"
DEFAULT_BAM_TEMPLATE = "real_data/HG002_chr{chrom}/bam/HG002_chr{chrom}.bam"
DEFAULT_TRUTH_TEMPLATE = "real_data/HG002_chr{chrom}/truth_chr{chrom}/HG002_chr{chrom}_DELINS_50.vcf.gz"
DEFAULT_BED_TEMPLATE = "real_data/HG002_chr{chrom}/truth_chr{chrom}/HG002_chr{chrom}_Tier1.bed"
DEFAULT_REFERENCE = "real_data/reference/GRCh37_hs37d5/hs37d5.fa"
DEFAULT_OUT_TEMPLATE = "results/tool_benchmark_chr{chrom}"
TOOL_LABELS = {"sniffles2": "Sniffles2", "cutesv": "cuteSV", "svim": "SVIM"}


def parse_chroms(text):
    chroms = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        chroms.append(item[3:] if item.lower().startswith("chr") else item)
    if not chroms:
        raise ValueError("No chromosomes were provided")
    return chroms


def parse_tools(text):
    tools = []
    aliases = {"sniffles": "sniffles2", "sniffles2": "sniffles2", "cutesv": "cutesv", "svim": "svim"}
    for item in str(text).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unsupported tool {item!r}; supported tools are sniffles2,cutesv,svim")
        tools.append(aliases[key])
    if not tools:
        raise ValueError("No tools were provided")
    return tools


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroms", default=DEFAULT_CHROMS, help="Comma-separated chromosomes, e.g. 19,20,21,22")
    parser.add_argument("--tools", default=DEFAULT_TOOLS, help="Comma-separated tools: sniffles2,cutesv,svim")
    parser.add_argument("--threads", type=int, default=16, help="Threads for tools that support threading")
    parser.add_argument("--bam-template", default=DEFAULT_BAM_TEMPLATE, help="BAM path template with {chrom}")
    parser.add_argument("--truth-template", default=DEFAULT_TRUTH_TEMPLATE, help="Truth VCF template with {chrom}")
    parser.add_argument("--bed-template", default=DEFAULT_BED_TEMPLATE, help="Tier1 BED template with {chrom}")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Reference FASTA for cuteSV and SVIM")
    parser.add_argument("--out-template", default=DEFAULT_OUT_TEMPLATE, help="Per-chromosome output root with {chrom}")
    parser.add_argument("--sniffles-bin", default="sniffles", help="Sniffles2 executable")
    parser.add_argument("--cutesv-bin", default="cuteSV", help="cuteSV executable")
    parser.add_argument("--svim-bin", default="svim", help="SVIM executable")
    parser.add_argument("--sample", default="HG002", help="Sample name used by tools when supported")
    parser.add_argument("--refdist", type=int, default=500, help="Truvari --refdist")
    parser.add_argument("--pctsize", type=float, default=0.5, help="Truvari --pctsize")
    parser.add_argument("--sizemin", type=int, default=50, help="Truvari --sizemin and tool minimum size where supported")
    parser.add_argument("--pctseq", type=float, default=0.0, help="Truvari --pctseq")
    parser.add_argument("--skip-existing", action="store_true", help="Skip caller execution when the output VCF exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running tools")
    args = parser.parse_args(argv)
    args.chrom_list = parse_chroms(args.chroms)
    args.tool_list = parse_tools(args.tools)
    return args


def render_template(template, chrom):
    return str(template).format(chrom=chrom)


def script_path(relative_path):
    return str(PROJECT_ROOT / relative_path)


def chromosome_paths(args, chrom):
    return {
        "bam": render_template(args.bam_template, chrom),
        "truth": render_template(args.truth_template, chrom),
        "bed": render_template(args.bed_template, chrom),
        "out_root": render_template(args.out_template, chrom),
    }


def tool_dir(args, chrom, tool):
    return Path(chromosome_paths(args, chrom)["out_root"]) / tool


def caller_vcf_path(args, chrom, tool):
    base = tool_dir(args, chrom, tool)
    if tool == "sniffles2":
        return base / "sniffles2.vcf"
    if tool == "cutesv":
        return base / "cutesv.vcf"
    if tool == "svim":
        return base / "variants.vcf"
    raise ValueError(f"Unsupported tool: {tool}")


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_executable(path_or_name, label):
    if Path(path_or_name).exists():
        return
    if shutil.which(path_or_name) is None:
        raise RuntimeError(f"Required {label} executable not found on PATH: {path_or_name}")


def check_inputs(args, chrom, tool):
    paths = chromosome_paths(args, chrom)
    require_file(paths["bam"], f"chr{chrom} BAM")
    require_file(paths["truth"], f"chr{chrom} truth VCF")
    require_file(paths["bed"], f"chr{chrom} Tier1 BED")
    if tool in {"cutesv", "svim"}:
        require_file(args.reference, "reference FASTA")
    if tool == "sniffles2":
        require_executable(args.sniffles_bin, "Sniffles2")
    elif tool == "cutesv":
        require_executable(args.cutesv_bin, "cuteSV")
    elif tool == "svim":
        require_executable(args.svim_bin, "SVIM")
    for executable in ["bcftools", "bgzip", "tabix", "truvari"]:
        require_executable(executable, executable)


def build_sniffles2_command(args, chrom):
    paths = chromosome_paths(args, chrom)
    return [
        args.sniffles_bin,
        "--input",
        paths["bam"],
        "--vcf",
        str(caller_vcf_path(args, chrom, "sniffles2")),
        "--threads",
        str(args.threads),
    ]


def build_cutesv_command(args, chrom):
    paths = chromosome_paths(args, chrom)
    output_dir = tool_dir(args, chrom, "cutesv")
    return [
        args.cutesv_bin,
        paths["bam"],
        args.reference,
        str(caller_vcf_path(args, chrom, "cutesv")),
        str(output_dir / "work"),
        "--threads",
        str(args.threads),
        "--min_size",
        str(args.sizemin),
    ]


def build_svim_command(args, chrom):
    paths = chromosome_paths(args, chrom)
    output_dir = tool_dir(args, chrom, "svim")
    return [
        args.svim_bin,
        "alignment",
        "--min_sv_size",
        str(args.sizemin),
        "--sample",
        args.sample,
        str(output_dir),
        paths["bam"],
        args.reference,
    ]


def build_tool_command(args, chrom, tool):
    if tool == "sniffles2":
        return build_sniffles2_command(args, chrom)
    if tool == "cutesv":
        return build_cutesv_command(args, chrom)
    if tool == "svim":
        return build_svim_command(args, chrom)
    raise ValueError(f"Unsupported tool: {tool}")


def benchmark_vcf_arg(tool):
    return f"--{tool}-vcf"


def build_benchmark_command(args, chrom, tool):
    paths = chromosome_paths(args, chrom)
    return [
        sys.executable,
        script_path("scripts/benchmark_truvari_delins.py"),
        "--truth",
        paths["truth"],
        "--bed",
        paths["bed"],
        "--outdir",
        str(tool_dir(args, chrom, tool)),
        "--chrom",
        chrom,
        "--refdist",
        str(args.refdist),
        "--pctsize",
        str(args.pctsize),
        "--sizemin",
        str(args.sizemin),
        "--pctseq",
        str(args.pctseq),
        benchmark_vcf_arg(tool),
        str(caller_vcf_path(args, chrom, tool)),
    ]


def prepare_tool_directories(args, chrom, tool):
    output_dir = tool_dir(args, chrom, tool)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tool == "cutesv":
        (output_dir / "work").mkdir(parents=True, exist_ok=True)
    return output_dir


def run_command(command, dry_run=False):
    logging.info("Command: %s", " ".join(map(str, command)))
    if not dry_run:
        subprocess.run(command, check=True)


def run_tool_for_chrom(args, chrom, tool):
    prepare_tool_directories(args, chrom, tool)
    vcf_path = caller_vcf_path(args, chrom, tool)

    if not args.dry_run:
        check_inputs(args, chrom, tool)

    if args.skip_existing and vcf_path.exists():
        logging.info("Skipping %s chr%s; VCF already exists: %s", TOOL_LABELS[tool], chrom, vcf_path)
    else:
        run_command(build_tool_command(args, chrom, tool), dry_run=args.dry_run)

    run_command(build_benchmark_command(args, chrom, tool), dry_run=args.dry_run)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Chromosomes: %s", ",".join(args.chrom_list))
    logging.info("Tools: %s", ",".join(args.tool_list))
    logging.info("Threads: %d", args.threads)
    logging.info("Dry run: %s", args.dry_run)
    for chrom in args.chrom_list:
        for tool in args.tool_list:
            logging.info("==== chr%s %s ====", chrom, TOOL_LABELS[tool])
            run_tool_for_chrom(args, chrom, tool)


if __name__ == "__main__":
    main()
