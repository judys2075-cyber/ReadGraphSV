# ReadGraphSV Workflow

ReadGraphSV converts long-read alignment evidence into candidate-level graph
classification problems.

## Evidence Extraction

The pipeline starts from a BAM/SAM file.

- `extract_cigar_events.py` extracts large CIGAR `D` and `I` events.
- `extract_extra_events.py` extracts softclip, SA-tag, and supplementary
  alignment evidence.

Coordinates are kept in 0-based BAM-style form internally.

## Candidate Generation

The v0.2 path clusters CIGAR DEL/INS events with `cluster_events.py`.

The optional v0.3 path adds:

- `extra_candidate_proposer.py` for conservative extra-evidence DEL/INS
  proposal.
- `merge_candidates_v3.py` for CIGAR/extra candidate merging.

This optional mode is enabled by:

```bash
--use_extra_candidates
```

## Graph Construction and Scoring

`build_graph_dataset_v2.py` builds one PyTorch Geometric graph per candidate.
Each graph contains:

- one candidate node;
- CIGAR evidence nodes;
- extra evidence nodes from softclip, SA tag, and supplementary alignments;
- candidate-evidence and nearby evidence-evidence edges.

`predict_gnn.py` loads a trained GraphSAGE model and assigns each candidate a
probability score.

## Deduplication and VCF Export

The optional deduplication step is enabled by:

```bash
--use_dedup
```

It runs `dedup_filtered_candidates.py` after GNN filtering and before VCF
export. Nearby same-type calls with similar sizes are clustered, and only the
best-scoring representative is retained.

The final output is a symbolic DEL/INS VCF suitable for Truvari evaluation.
