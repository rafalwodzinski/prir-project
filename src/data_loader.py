from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(
    filepath: Path,
    dtype: np.dtype | type = np.float32,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Wczytuje dataset CS:GO, koduje kolumny kategoryczne i przygotowuje tablice numpy
    w formacie C-contiguous (pod mini-batch SGD i dalsze etapy z MPI/CUDA).
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")

    # Wczytujemy dane i usuwamy ewentualne puste wiersze
    df = pd.read_csv(filepath)
    df = df.dropna().reset_index(drop=True)

    if "round_winner" not in df.columns:
        raise KeyError("Brak kolumny 'round_winner' w pliku.")

    # Kodujemy wynik rundy binarie: CT to 1, T to 0
    raw_y = df["round_winner"]
    if raw_y.dtype == object or isinstance(raw_y.iloc[0], str):
        y_mapped = raw_y.map({"CT": 1, "T": 0})
        if y_mapped.isna().any():
            raise ValueError("Niepoprawne wartosci w round_winner - oczekiwano CT lub T.")
        y = y_mapped.to_numpy(dtype=dtype)
    else:
        y = raw_y.to_numpy(dtype=dtype)

    features = df.drop(columns=["round_winner"])

    # bomb_planted z bool/string na float
    if "bomb_planted" in features.columns:
        if features["bomb_planted"].dtype == bool or features["bomb_planted"].dtype == object:
            features["bomb_planted"] = (
                features["bomb_planted"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
                .fillna(0)
                .astype(dtype)
            )
        else:
            features["bomb_planted"] = features["bomb_planted"].astype(dtype)

    # One-hot na mapach
    if "map" in features.columns:
        features = pd.get_dummies(features, columns=["map"], dtype=dtype)

    # Konwertujemy na numpy i wymuszamy uklad C-contiguous w RAM
    # (wiersze po kolei, zeby wycinanie partii X[start:end] dzialalo jako zero-copy view)
    X = np.ascontiguousarray(features.to_numpy(dtype=dtype), dtype=dtype)
    y = np.ascontiguousarray(y, dtype=dtype)

    assert X.flags["C_CONTIGUOUS"], "X musi byc C-contiguous"
    assert y.flags["C_CONTIGUOUS"], "y musi byc C-contiguous"
    assert y.ndim == 1, "y musi byc wektorem 1D"
    assert X.shape[0] == y.shape[0], "Liczba wierszy w X i y sie nie zgadza"

    return X, y


def split_dataset(
    X: NDArray[Any],
    y: NDArray[Any],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Podzial na train/test ze stratyfikacja klas i ustalonym seedem.
    Zwraca tablice z zachowana flaga C_CONTIGUOUS.
    """
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Upewniamy sie, ze po splicie tablice nadal sa ciagle w pamieci
    X_train = np.ascontiguousarray(X_train_raw)
    X_test = np.ascontiguousarray(X_test_raw)
    y_train = np.ascontiguousarray(y_train_raw)
    y_test = np.ascontiguousarray(y_test_raw)

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: NDArray[Any],
    X_test: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], StandardScaler]:
    """
    Standaryzacja cech przez StandardScaler.
    Wazne: fit robimy TYLKO na zbiorze treningowym, zeby uniknac data leakage.
    Test jest tylko transformowany.
    """
    orig_dtype = X_train.dtype
    scaler = StandardScaler()

    # Dopasowanie tylko na train
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rzutujemy z powrotem na wyjsciowy typ (np. float32) i pilnujemy C_CONTIGUOUS
    X_train_scaled = np.ascontiguousarray(X_train_scaled, dtype=orig_dtype)
    X_test_scaled = np.ascontiguousarray(X_test_scaled, dtype=orig_dtype)

    return X_train_scaled, X_test_scaled, scaler
