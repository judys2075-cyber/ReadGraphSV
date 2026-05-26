# ReadGraphSV v0.1

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
- Cluster read-level signals into candidate SV regions.
- Optionally label candidates against a truth VCF for supervised training and
  evaluation.
- Build PyTorch Geometric graph datasets from candidate regions.
- Train a GraphSAGE model for true/false candidate classification.
- Run one-command inference from BAM to filtered TSV and VCF output.
- Evaluate raw candidate calls against GNN-filtered calls.
- Merge multiple graph datasets for larger training runs.

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

PyTorch Geometric installation can depend on your CUDA/PyTorch version. If the
plain install fails, follow the official PyG wheel instructions for your
machine.

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
  --epochs 100
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

Natural next steps are adding SA-tag/split-read evidence, soft-clip rescue,
edge features, multi-task SV type prediction, and breakpoint refinement.
