"""Tests for GNN training and fine-tuning workflows."""

import torch
from torch_geometric.data import Data

from conftest import PROJECT_ROOT, run_cli
from train_gnn import parse_args
from utils_graph import ReadGraphSAGE


FEATURE_NAMES = ["f0", "f1", "f2", "f3"]


def make_graph(value, label):
    x = torch.tensor([[value, 0.0, 1.0, 0.0], [0.0, value, 0.0, 1.0]], dtype=torch.float32)
    graph = Data(
        x=x,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([float(label)], dtype=torch.float32),
    )
    graph.feature_names = FEATURE_NAMES
    return graph


def write_dataset(path, values_and_labels):
    torch.save([make_graph(value, label) for value, label in values_and_labels], path)


def write_init_model(path):
    model = ReadGraphSAGE(in_channels=len(FEATURE_NAMES), hidden_channels=8)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_channels": len(FEATURE_NAMES),
            "hidden_channels": 8,
            "feature_names": FEATURE_NAMES,
        },
        path,
    )


def test_parse_args_supports_finetuning_options():
    args = parse_args(
        [
            "--dataset",
            "train.pt",
            "--val_dataset",
            "val.pt",
            "--test_dataset",
            "test.pt",
            "--init_model",
            "init.pt",
            "--model_out",
            "out.pt",
            "--class_weight",
            "auto",
            "--patience",
            "3",
            "--freeze_encoder",
        ]
    )

    assert args.val_dataset == "val.pt"
    assert args.test_dataset == "test.pt"
    assert args.init_model == "init.pt"
    assert args.class_weight == "auto"
    assert args.patience == 3
    assert args.freeze_encoder is True


def test_train_gnn_finetunes_with_independent_val_and_test(tmp_path):
    train = tmp_path / "train.pt"
    val = tmp_path / "val.pt"
    test = tmp_path / "test.pt"
    init_model = tmp_path / "init.pt"
    model_out = tmp_path / "finetuned.pt"

    write_dataset(train, [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)])
    write_dataset(val, [(0.15, 0), (0.85, 1)])
    write_dataset(test, [(0.12, 0), (0.88, 1)])
    write_init_model(init_model)

    run_cli(
        "train_gnn.py",
        "--dataset",
        train,
        "--val_dataset",
        val,
        "--test_dataset",
        test,
        "--init_model",
        init_model,
        "--model_out",
        model_out,
        "--epochs",
        "3",
        "--hidden",
        "8",
        "--lr",
        "0.001",
        "--class_weight",
        "auto",
        "--patience",
        "2",
        "--freeze_encoder",
        cwd=PROJECT_ROOT,
    )

    checkpoint = torch.load(model_out, map_location="cpu", weights_only=False)
    assert checkpoint["in_channels"] == len(FEATURE_NAMES)
    assert checkpoint["hidden_channels"] == 8
    assert checkpoint["feature_names"] == FEATURE_NAMES
    assert checkpoint["split_mode"] == "external_validation"
    assert checkpoint["init_model"] == str(init_model)
    assert checkpoint["val_dataset"] == str(val)
    assert checkpoint["test_dataset"] == str(test)
    assert checkpoint["class_weight"] == "auto"
    assert checkpoint["freeze_encoder"] is True
    assert 1 <= checkpoint["epoch"] <= 3


def test_train_gnn_random_split_remains_supported(tmp_path):
    dataset = tmp_path / "dataset.pt"
    model_out = tmp_path / "model.pt"
    write_dataset(dataset, [(0.1, 0), (0.2, 0), (0.3, 0), (0.8, 1), (0.9, 1), (1.0, 1)])

    run_cli(
        "train_gnn.py",
        "--dataset",
        dataset,
        "--model_out",
        model_out,
        "--epochs",
        "2",
        "--hidden",
        "8",
        "--class_weight",
        "none",
        "--patience",
        "0",
        cwd=PROJECT_ROOT,
    )

    checkpoint = torch.load(model_out, map_location="cpu", weights_only=False)
    assert checkpoint["split_mode"] == "random_train_val_test"
    assert checkpoint["class_weight"] == "none"
