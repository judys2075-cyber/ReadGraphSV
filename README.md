# ReadGraphSV

ReadGraphSV is a prototype graph neural network framework for structural
variant discovery from long-read alignments. The project focuses on turning
alignment evidence from BAM files into explicit region-level graphs, so that a
GNN can learn whether a candidate event is likely to represent a real
structural variant.

The first release, ReadGraphSV v0.1, targets deletion (DEL) and insertion (INS)
candidate scoring. It scans primary and supplementary long-read alignments,
extracts large CIGAR `D` and `I` operations, clusters nearby read-level signals
into candidate SV regions, converts each candidate into a graph of read
evidence, and applies a GraphSAGE model for graph-level binary classification.

The central idea is simple: structural variants are not just isolated positions,
but patterns of agreement among multiple read alignments. ReadGraphSV therefore
represents each candidate as a small evidence graph:

- one candidate node summarizes the putative SV region;
- read evidence nodes represent supporting CIGAR-derived events;
- graph edges connect reads to the candidate and connect similar read evidence
  events to each other.

This design keeps the evidence interpretable and modular. The current
implementation is intentionally compact, making it suitable for method
development, benchmarking, and rapid experimentation with graph-based SV
classification.

## Current Capabilities

- Extract CIGAR-derived DEL and INS signals directly from long-read BAM files.
- Extract v0.2 extra evidence from soft clips, SA tags, and supplementary
  alignment records.
- Cluster read-level signals into candidate SV regions.
- Optionally label candidates against a truth VCF for supervised training and
  evaluation.
- Build PyTorch Geometric graph datasets from candidate regions.
- Train a GraphSAGE model for true/false candidate classification.
- Run one-command inference from BAM to filtered TSV and VCF output.
- Evaluate raw candidate calls against GNN-filtered calls.
- Merge multiple graph datasets for larger training runs.

## Benchmark Results

ReadGraphSV evaluates two stages:

- **Raw CIGAR candidates**: all clustered candidates are treated as positive
  calls.
- **ReadGraphSV GNN-filtered calls**: candidates are retained only when their
  GNN probability is above the selected threshold.

Example held-out chr21 evaluation:

| Method | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|
| Raw CIGAR candidates | 0.3589 | 1.0000 | 0.5282 | N/A |
| ReadGraphSV GNN-filtered | 0.7628 | 0.9369 | 0.8410 | N/A |

In this run, GNN filtering reduced false positives from 368 to 60 while
retaining 193 of 206 true positives. A reusable reporting template is available
at `results/benchmark_template.md`.

## Coordinate Convention

ReadGraphSV uses 0-based BAM-style coordinates internally. Deletions are
represented as half-open intervals. Insertions are represented at the current
reference position with `event_end = event_pos + 1`.

Truth VCF positions are converted from 1-based VCF POS to 0-based coordinates
during labeling.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, create a conda environment:

```bash
conda env create -f environment.yml
conda activate readgraphsv
```

PyTorch Geometric installation can depend on your CUDA/PyTorch version. If the
plain install fails, follow the official PyG wheel instructions for your
machine.

For development and tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

## Reproducibility

ReadGraphSV is designed as a reproducible method prototype. Recommended
practice:

- Use `environment.yml` or pinned `requirements.txt` versions to create the
  software environment.
- Keep raw BAM files, truth VCFs, graph datasets, model checkpoints, and
  prediction outputs outside version control.
- Record candidate extraction parameters: `--min_size`, `--window`, and
  `--min_support`.
- For supervised experiments, split data into train, validation, and test sets.
- Use validation data for model selection and threshold selection; reserve the
  test set for final reporting.
- Save `results/train_metrics.csv`, `results/test_metrics.txt`, and
  `results/evaluation.txt` for each experiment.

## One-Command Inference

After training a model, run the full inference pipeline from BAM to filtered
TSV and VCF:

```bash
python run_readgraphsv.py \
  --bam aligned.bam \
  --model models/readgraph_gnn.pt \
  --outdir output \
  --threshold 0.5
```

This creates:

- `output/signals.tsv`
- `output/candidates.tsv`
- `output/candidates_for_graph.tsv`
- `output/graph_dataset.pt`
- `output/predictions.tsv`
- `output/filtered_candidates.tsv`
- `output/filtered.vcf`

For optional evaluation against a truth VCF:

```bash
python run_readgraphsv.py \
  --bam aligned.bam \
  --model models/readgraph_gnn.pt \
  --truth truth.vcf \
  --outdir output \
  --threshold 0.5
```

Evaluation mode additionally writes:

- `output/candidates_labeled.tsv`
- `output/evaluation.txt`

ReadGraphSV v0.1 currently supports CIGAR-derived DEL/INS candidate scoring.
Future versions are planned to add split-read signals, soft-clip rescue,
edge-level prediction, and richer complex-SV graph representations.

## Step-By-Step Pipeline

Create output directories:

```bash
mkdir -p data graphs models results
```

Extract CIGAR DEL/INS signals:

```bash
python extract_cigar_events.py \
  --bam aligned_chr21_10x_with_rg.bam \
  --min_size 50 \
  --out data/signals.tsv
```

Extract v0.2 soft-clip, SA-tag, and supplementary-alignment evidence:

```bash
python extract_extra_events.py \
  --bam aligned_chr21_10x_with_rg.bam \
  --min_clip 50 \
  --out data/extra_signals.tsv
```

Cluster signals into candidate SVs:

```bash
python cluster_events.py \
  --signals data/signals.tsv \
  --window 500 \
  --min_support 2 \
  --out data/candidates.tsv
```

Label candidates with a truth VCF:

```bash
python label_candidates.py \
  --candidates data/candidates.tsv \
  --truth truth.vcf \
  --max_dist 500 \
  --min_size_sim 0.7 \
  --out data/candidates_labeled.tsv
```

Build a PyTorch Geometric graph dataset:

```bash
python build_graph_dataset.py \
  --signals data/signals.tsv \
  --candidates data/candidates_labeled.tsv \
  --out graphs/dataset.pt
```

Train the GNN:

```bash
python train_gnn.py \
  --dataset graphs/dataset.pt \
  --model_out models/readgraph_gnn.pt \
  --epochs 100 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --auto_threshold
```

Predict candidate probabilities:

```bash
python predict_gnn.py \
  --dataset graphs/dataset.pt \
  --model models/readgraph_gnn.pt \
  --out results/predictions.tsv
```

Evaluate raw candidate calls versus GNN-filtered calls:

```bash
python evaluate_predictions.py \
  --pred results/predictions.tsv \
  --threshold 0.5 \
  --out results/evaluation.txt
```

Merge multiple graph datasets:

```bash
python merge_datasets.py \
  --inputs graphs/dataset.pt runs/chr21_002/graphs/dataset.pt runs/chr21_003/graphs/dataset.pt \
  --out graphs/combined_chr21_001_002_003.pt
```

Export filtered predictions to VCF:

```bash
python export_vcf.py \
  --pred results/predictions.tsv \
  --threshold 0.5 \
  --out results/filtered.vcf
```

## Files

- `extract_cigar_events.py`: scans primary and supplementary alignments and
  emits large CIGAR DEL/INS events.
- `extract_extra_events.py`: extracts v0.2 soft-clip, SA-tag connection, and
  supplementary-alignment evidence from BAM/SAM records.
- `cluster_events.py`: groups nearby same-type signals into candidate SVs.
- `label_candidates.py`: labels candidates by matching DEL/INS truth VCF
  records.
- `build_graph_dataset.py`: builds one region-level PyG graph per candidate.
- `train_gnn.py`: trains a GraphSAGE binary graph classifier.
- `predict_gnn.py`: runs a trained model on candidate graphs.
- `evaluate_predictions.py`: compares raw candidates with GNN-filtered
  candidates.
- `merge_datasets.py`: merges multiple PyG `dataset.pt` files.
- `run_readgraphsv.py`: runs the v0.1 inference pipeline end to end.
- `export_vcf.py`: exports filtered DEL/INS predictions as a simple VCF.

## v0.1 Scope

The first version is intentionally narrow:

- DEL and INS only.
- CIGAR-derived evidence only.
- Candidate node plus read evidence nodes.
- Graph-level binary classification.

## v0.2 Extra Evidence

ReadGraphSV v0.2 adds a standalone extra evidence extractor for signals that
are useful for split-read and complex-SV modeling:

- `SOFTCLIP_LEFT` and `SOFTCLIP_RIGHT` evidence from large terminal soft clips.
- `SA_CONNECTION` evidence from primary-alignment `SA:Z` tags, with destination
  segment coordinates, orientation changes, chromosome changes, destination
  MAPQ, SA CIGAR, and NM values.
- `SUPPLEMENTARY` evidence from records carrying the supplementary alignment
  flag.

All extra evidence coordinates are 0-based. The extractor scans BAM/SAM records
in file order and does not require an index. If a BAM contains no qualifying
soft clips, SA tags, or supplementary records, it still writes a valid TSV with
only the header.

The v0.2 extra evidence file is currently independent from the v0.1 graph
builder. The next integration step is to turn these signals into additional
read-node features and edge attributes for richer evidence graphs.

## Development Roadmap

Planned development path:

- **v0.2**: extract SA tag, soft-clip, and supplementary-alignment evidence,
  then integrate these signals into graph features.
- **v0.3**: add edge attributes such as position distance, SV length
  similarity, same-read links, same-strand links, and orientation-change
  indicators.
- **v0.4**: extend from binary candidate filtering to multi-task learning:
  true/false classification, SV type classification, and breakpoint refinement.
- **v0.5**: expand beyond DEL/INS toward inversion, duplication,
  translocation, and complex-SV graph representations.
