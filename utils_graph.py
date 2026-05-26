#!/usr/bin/env python3
"""Graph utilities and the baseline GraphSAGE model."""

import math

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


FEATURE_NAMES = [
    "svtype_DEL",
    "svtype_INS",
    "log_svlen",
    "log_support",
    "mean_mapq_norm",
    "std_pos_norm",
    "std_svlen_norm",
    "log_read_len",
    "read_mapq_norm",
    "strand_numeric",
    "distance_to_candidate_center_norm",
    "relative_svlen_diff",
    "has_sa",
    "is_supplementary",
    "is_candidate_node",
    "is_read_node",
]


def log1p_norm(value, scale=1000.0):
    """Log-normalize a positive numeric value."""
    try:
        value = max(0.0, float(value))
    except (TypeError, ValueError):
        value = 0.0
    return math.log1p(value) / math.log1p(scale)


def numeric(value, default=0.0):
    """Convert a value to float with a safe fallback."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_numeric(value):
    """Convert common boolean encodings to 0.0/1.0."""
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1.0
    return 0.0


def strand_numeric(value):
    """Encode strand as +1/-1/0."""
    if str(value) == "+":
        return 1.0
    if str(value) == "-":
        return -1.0
    return 0.0


class ReadGraphSAGE(nn.Module):
    """Small GraphSAGE model for graph-level binary classification."""

    def __init__(self, in_channels, hidden_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = float(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        pooled = global_mean_pool(x, batch)
        return self.classifier(pooled).view(-1)
