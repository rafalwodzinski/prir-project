import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from main import (
    parse_args,
    save_model,
    load_model,
    save_history,
    load_history,
    calculate_time_to_quality,
    run_training,
)


def test_parse_args_defaults():
    """Weryfikuje domyslne wartosci argumentow CLI."""
    args = parse_args([])
    assert args.mode == "seq"
    assert args.lr == 0.01
    assert args.batch_size == 256
    assert args.epochs == 20
    assert args.seed == 42
    assert args.target_acc == 0.75
    assert args.save_model is None
    assert args.save_history is None


def test_parse_args_custom():
    """Weryfikuje parsowanie niestandardowych wartosci argumentow CLI."""
    custom_args = [
        "--mode", "numba",
        "--lr", "0.05",
        "--batch-size", "128",
        "--epochs", "10",
        "--seed", "123",
        "--target-acc", "0.80",
        "--save-model", "test_model.npz",
        "--save-history", "test_hist.json",
        "--quiet",
    ]
    args = parse_args(custom_args)
    assert args.mode == "numba"
    assert args.lr == 0.05
    assert args.batch_size == 128
    assert args.epochs == 10
    assert args.seed == 123
    assert args.target_acc == 0.80
    assert args.save_model == "test_model.npz"
    assert args.save_history == "test_hist.json"
    assert args.quiet is True


def test_save_and_load_model_npy_and_npz(tmp_path: Path):
    """Sprawdza zapis i wczytanie wag modelu w formacie .npy oraz .npz z metadanymi."""
    weights = np.array([0.1, -0.2, 0.5, 1.2], dtype=np.float32)

    # 1. Format .npy
    npy_path = tmp_path / "model.npy"
    saved_path = save_model(npy_path, weights)
    assert saved_path.is_file()
    loaded_w, meta = load_model(saved_path)
    assert np.allclose(loaded_w, weights)
    assert meta is None

    # 2. Format .npz z metadanymi
    npz_path = tmp_path / "model.npz"
    metadata = {"epochs": 15, "lr": 0.05, "mode": "seq"}
    saved_npz = save_model(npz_path, weights, metadata=metadata)
    assert saved_npz.is_file()
    loaded_w_npz, loaded_meta = load_model(saved_npz)
    assert np.allclose(loaded_w_npz, weights)
    assert loaded_meta is not None
    assert loaded_meta["epochs"] == 15
    assert loaded_meta["mode"] == "seq"


def test_save_and_load_history_json(tmp_path: Path):
    """Sprawdza serializacje historii uczenia do JSON i jej poprawne odczytanie."""
    history = {
        "loss": [0.65, 0.55, 0.45],
        "accuracy": [0.60, 0.72, 0.81],
        "epoch_times": [0.012, 0.011, 0.013],
        "target_accuracy": 0.75,
        "time_to_target_accuracy": 0.023,
        "epoch_to_target_accuracy": 2,
    }

    json_path = tmp_path / "history.json"
    saved_path = save_history(json_path, history)
    assert saved_path.is_file()

    loaded_h = load_history(saved_path)
    assert loaded_h["loss"] == history["loss"]
    assert loaded_h["accuracy"] == history["accuracy"]
    assert loaded_h["epoch_to_target_accuracy"] == 2
    assert np.isclose(loaded_h["time_to_target_accuracy"], 0.023)


def test_calculate_time_to_quality():
    """Weryfikuje wyznaczanie czasu i epoki osiagniecia zadanego progu jakosci."""
    history = {
        "accuracy": [0.50, 0.65, 0.76, 0.82],
        "epoch_times": [0.10, 0.10, 0.12, 0.09],
    }

    # Prog 0.70 osiagniety w epoce 3 (czas: 0.10 + 0.10 + 0.12 = 0.32)
    time_q, ep_q = calculate_time_to_quality(history, target_accuracy=0.70)
    assert ep_q == 3
    assert np.isclose(time_q, 0.32)

    # Prog 0.90 nieosiagniety
    time_unreached, ep_unreached = calculate_time_to_quality(history, target_accuracy=0.90)
    assert time_unreached is None
    assert ep_unreached is None


def test_cli_end_to_end_execution(tmp_path: Path):
    """
    DoD Issue #19:
    Weryfikuje kompletne wywolanie CLI, zapis modelu i historii oraz raportowanie Time-to-Quality.
    """
    repo_root = Path(__file__).resolve().parent.parent
    model_path = tmp_path / "cli_model.npy"
    hist_path = tmp_path / "cli_history.json"

    cmd = [
        sys.executable,
        str(repo_root / "src" / "main.py"),
        "--mode", "seq",
        "--epochs", "5",
        "--batch-size", "128",
        "--lr", "0.05",
        "--target-acc", "0.74",
        "--save-model", str(model_path),
        "--save-history", str(hist_path),
        "--quiet",
    ]

    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"Blad CLI:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    assert model_path.is_file(), "Plik wag modelu nie zostal utworzony"
    assert hist_path.is_file(), "Plik historii nie zostal utworzony"

    weights, _ = load_model(model_path)
    assert weights.ndim == 1
    assert len(weights) > 0

    history = load_history(hist_path)
    assert len(history["loss"]) == 5
    assert len(history["epoch_times"]) == 5
    assert history["target_accuracy"] == 0.74
    assert history["epoch_to_target_accuracy"] is not None
    assert history["time_to_target_accuracy"] is not None

