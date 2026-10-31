from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_data, split_dataset, scale_features


@pytest.fixture
def synthetic_csgo_csv(tmp_path: Path) -> Path:
    """Tworzy maly syntetyczny plik csv z kolumnami z cs:go do szybkich testow."""
    np.random.seed(42)
    n = 200

    data = {
        "ct_money": np.random.uniform(0, 16000, size=n),
        "t_money": np.random.uniform(0, 16000, size=n),
        "time_left": np.random.uniform(0.0, 175.0, size=n),
        "ct_health": np.random.uniform(0, 500, size=n),
        "t_health": np.random.uniform(0, 500, size=n),
        "bomb_planted": np.random.choice([True, False], size=n),
        "map": np.random.choice(["de_dust2", "de_mirage", "de_inferno", "de_nuke"], size=n),
        "round_winner": np.random.choice(["CT", "T"], size=n, p=[0.55, 0.45]),
    }

    csv_path = tmp_path / "test_csgo.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_memory_layout_and_dtypes(synthetic_csgo_csv: Path):
    """Sprawdzenie typow, ciaglosci C-contiguous i wycinania partii (batcha)."""
    target_dtype = np.float32
    X, y = load_and_preprocess_data(synthetic_csgo_csv, dtype=target_dtype)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype == target_dtype
    assert y.dtype == target_dtype
    assert X.flags["C_CONTIGUOUS"]
    assert y.flags["C_CONTIGUOUS"]

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.25, seed=42)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # Kazda tablica po podziale i skalowaniu powinna byc C-contiguous
    for arr in [X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled]:
        assert arr.flags["C_CONTIGUOUS"]
        assert arr.dtype == target_dtype

    # Test wycinania partii do mini-batch SGD: slice po wierszach powinien byc ciagly
    batch_size = 32
    batch_x = X_train_scaled[:batch_size]
    batch_y = y_train[:batch_size]
    assert batch_x.flags["C_CONTIGUOUS"]
    assert batch_y.flags["C_CONTIGUOUS"]
    assert batch_x.shape == (batch_size, X_train.shape[1])
    assert batch_y.shape == (batch_size,)


def test_dimension_consistency(synthetic_csgo_csv: Path):
    """Sprawdzenie czy liczby wierszy i kolumn po podziale sie zgadzaja."""
    test_size = 0.2
    X, y = load_and_preprocess_data(synthetic_csgo_csv, dtype=np.float32)
    n_samples, n_features = X.shape

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size, seed=42)

    n_test_exp = int(n_samples * test_size)
    n_train_exp = n_samples - n_test_exp

    assert X_train.shape[0] == y_train.shape[0] == n_train_exp
    assert X_test.shape[0] == y_test.shape[0] == n_test_exp
    assert X_train.shape[1] == X_test.shape[1] == n_features


def test_normalization_and_no_data_leakage(synthetic_csgo_csv: Path):
    """Sprawdzenie czy train ma srednia ~0 i std ~1 oraz czy test nie uzywa wlasnego fit."""
    X, y = load_and_preprocess_data(synthetic_csgo_csv, dtype=np.float32)
    X_train, X_test, _, _ = split_dataset(X, y, test_size=0.2, seed=42)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Zbiór treningowy powinien miec srednia ~ 0 i std ~ 1
    mean_train = np.mean(X_train_scaled, axis=0)
    std_train = np.std(X_train_scaled, axis=0)
    assert np.allclose(mean_train, 0.0, atol=1e-5)
    assert np.allclose(std_train, 1.0, atol=1e-5)

    # X_test_scaled musi byc przeskalowany dokladnie parametrami z traina (scaler.mean_, scaler.scale_)
    expected_test_scaled = ((X_test - scaler.mean_) / scaler.scale_).astype(X_train.dtype)
    assert np.allclose(X_test_scaled, expected_test_scaled, atol=1e-5)

    # Sprawdzenie data leakage: gdyby ktos zrobil fit na X_test, dostalby inne srednie
    test_scaler = StandardScaler()
    illicit_scaled = test_scaler.fit_transform(X_test).astype(X_train.dtype)
    assert not np.allclose(scaler.mean_, test_scaler.mean_, atol=1e-2)
    assert not np.allclose(X_test_scaled, illicit_scaled, atol=1e-2)
