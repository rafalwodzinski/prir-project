from pathlib import Path
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray

from data_loader import load_and_preprocess_data, split_dataset, scale_features
from math_utils import sigmoid, compute_bce_loss, compute_gradient


def add_intercept(X: NDArray[Any]) -> NDArray[Any]:
    """
    Dokleja kolumne jedynek na poczatku macierzy X (jako bias / wyraz wolny),
    zachowujac typ danych i ciaglosc pamieci C_CONTIGUOUS.
    """
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    X_with_bias = np.hstack([ones, X])
    return np.ascontiguousarray(X_with_bias, dtype=X.dtype)


def compute_accuracy(y_true: NDArray[Any], y_pred_probs: NDArray[Any], threshold: float = 0.5) -> float:
    """
    Oblicza dokladnosc klasyfikacji binarnej dla zadanych prawdopodobienstw.
    """
    preds = (y_pred_probs >= threshold).astype(y_true.dtype)
    return float(np.mean(preds == y_true))


def predict_proba(X: NDArray[Any], weights: NDArray[Any]) -> NDArray[Any]:
    """
    Oblicza prawdopodobienstwa klasy 1: sigma(X @ W).
    """
    return sigmoid(X @ weights)


def train_sequential_sgd(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_val: NDArray[Any] | None = None,
    y_val: NDArray[Any] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[NDArray[Any], dict[str, list[float]]]:
    """
    Sekwencyjna petla uczenia Mini-Batch SGD dla regresji logistycznej:
    Tasowanie zbioru -> dzielenie na Mini-Batche -> Gradient -> Aktualizacja Wag (W = W - alpha * grad).

    Zwraca wyuczone wagi W oraz slownik z historia Loss i Accuracy z epok.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    # Inicjalizacja wag zerami (ten sam dtype co X_train, np. float32 lub float64)
    weights = np.zeros(n_features, dtype=X_train.dtype)

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # 1. Tasowanie zbioru na poczatku kazdej epoki
        indices = rng.permutation(n_samples)
        X_shuffled = np.ascontiguousarray(X_train[indices])
        y_shuffled = np.ascontiguousarray(y_train[indices])

        # 2. Dzielenie na Mini-Batche -> Gradient -> Aktualizacja Wag
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Forward pass na mini-batchu
            y_pred_batch = sigmoid(X_batch @ weights)

            # Wektorowy gradient
            grad = compute_gradient(X_batch, y_batch, y_pred_batch)

            # Aktualizacja wag: W = W - alpha * grad
            weights = weights - learning_rate * grad

        # 3. Ewaluacja po epoce i zbieranie historii
        train_probs = predict_proba(X_train, weights)
        epoch_loss = compute_bce_loss(y_train, train_probs)
        epoch_acc = compute_accuracy(y_train, train_probs)

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        elapsed = time.perf_counter() - t0
        history["epoch_times"].append(elapsed)

        if X_val is not None and y_val is not None:
            val_probs = predict_proba(X_val, weights)
            val_loss = compute_bce_loss(y_val, val_probs)
            val_acc = compute_accuracy(y_val, val_probs)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                    f"Time: {elapsed:.4f}s"
                )
        else:
            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                    f"Time: {elapsed:.4f}s"
                )

    return weights, history


def generate_synthetic_dataset(
    n_samples: int = 2000,
    n_features: int = 10,
    seed: int = 42,
    dtype: np.dtype | type = np.float32,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Generuje syntetyczny zbior danych liniowo separowalny z niewielkim szumem,
    uzywany gdy plik csgo_round_snapshots.csv nie jest dostepny lokalnie.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(dtype)
    true_w = rng.uniform(-2.0, 2.0, size=(n_features,)).astype(dtype)
    true_b = dtype(0.35)

    logits = X @ true_w + true_b
    probs = sigmoid(logits)
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(dtype)

    return np.ascontiguousarray(X, dtype=dtype), np.ascontiguousarray(y, dtype=dtype)


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / "data" / "csgo_round_snapshots.csv"

    if data_path.is_file():
        print(f"Wczytywanie danych CS:GO z: {data_path}")
        X, y = load_and_preprocess_data(data_path, dtype=np.float32)
    else:
        print(f"Brak pliku {data_path} - uzywam syntetycznego zbioru danych do demonstracji.")
        X, y = generate_synthetic_dataset(n_samples=4000, n_features=12, seed=42, dtype=np.float32)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # Doklejamy kolumne jedynek (intercept/bias), zeby model uczyl sie rowniez progu odciecia
    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    print(
        f"Start treningu sekwencyjnego Mini-Batch SGD: "
        f"train={X_train_final.shape}, test={X_test_final.shape}"
    )

    weights, history = train_sequential_sgd(
        X_train_final,
        y_train,
        X_val=X_test_final,
        y_val=y_test,
        epochs=20,
        batch_size=256,
        learning_rate=0.01,
        seed=42,
        verbose=True,
    )

    print("\nTrening zakonczony pomyslnie!")
    print(f"Poczatkowy Loss (Epoka 1): {history['loss'][0]:.4f} -> Koncowy Loss: {history['loss'][-1]:.4f}")
    print(f"Koncowe Accuracy (Train): {history['accuracy'][-1] * 100:.2f}%")
    if history["val_accuracy"]:
        print(f"Koncowe Accuracy (Test):  {history['val_accuracy'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
