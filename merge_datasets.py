#!/usr/bin/env python3
"""Merge multiple PyTorch Geometric graph dataset files."""

import argparse
import logging
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more input dataset.pt files saved as list[Data]",
    )
    parser.add_argument("--out", required=True, help="Output merged dataset.pt")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_torch_load(path):
    """Load PyTorch objects while allowing PyG Data pickles."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset does not exist: {path}")

    dataset = safe_torch_load(path)
    if dataset is None:
        dataset = []
    if not isinstance(dataset, list):
        raise TypeError(f"Expected list[Data] in {path}, got {type(dataset).__name__}")
    return dataset


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_parent_dir(args.out)

    merged = []
    for path in args.inputs:
        dataset = load_dataset(path)
        logging.info("%s: %d graphs", path, len(dataset))
        merged.extend(dataset)

    torch.save(merged, args.out)
    logging.info("Merged total: %d graphs", len(merged))
    logging.info("Wrote merged dataset to %s", args.out)


if __name__ == "__main__":
    main()
