from pathlib import Path
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange, get_thread_id, get_num_threads

from data_loader import load_and_preprocess_data, split_dataset, scale_features
from math_utils import sigmoid, compute_bce_loss
from sequential_sgd import add_intercept, compute_accuracy, predict_proba, generate_synthetic_dataset


@njit(parallel=True)
def compute_gradient_numba(
    X: NDArray[Any],
    y_true: NDArray[Any],
    weights: NDArray[Any],
) -> NDArray[Any]:
    """
    Rownolegle obliczenie wektora gradientu na wielordzeniowym CPU przy uzyciu Numby (@njit(parallel=True)).
    Wykorzystuje wzorzec Map-Reduce oraz bufory prywatne dla watkow (thread-local accumulators),
    aby wyeliminowac wyscigi watkow (Race Conditions) i False Sharing w pamieci podrecznej L1/L2 CPU.
    """
    batch_size = X.shape[0]
    n_features = X.shape[1]
    n_threads = get_num_threads()

    # Lokalna tablica akumulujaca dla kazdego watku (wzorzec Map):
    # Wiersz `tid` nalezy wylacznie do watku o identyfikatorze `tid`.
    local_grads = np.zeros((n_threads, n_features), dtype=X.dtype)

    # Faza MAP: Rownolegla iteracja po wierszach mini-batcha za pomoca prange
    for i in prange(batch_size):
        tid = get_thread_id()

        # 1. Obliczenie iloczynu skalarnego z_i = X[i] @ weights
        z = 0.0
        for j in range(n_features):
            z += X[i, j] * weights[j]

        # 2. Stabilna numerycznie funkcja sigmoidalna (zabezpieczenie przed overflow w exp)
        if z < -85.0:
            z_safe = -85.0
        elif z > 85.0:
            z_safe = 85.0
        else:
            z_safe = z

        y_pred_i = 1.0 / (1.0 + np.exp(-z_safe))
        err_i = y_pred_i - y_true[i]

        # 3. Akumulacja wkładu próbki `i` do prywatnego bufora bieżącego wątku `tid`
        for j in range(n_features):
            local_grads[tid, j] += err_i * X[i, j]

    # Faza REDUCE: Bezpieczna redukcja prywatnych buforow watkow do wynikowego wektora gradientu
    grad = np.zeros(n_features, dtype=X.dtype)
    for j in range(n_features):
        acc = 0.0
        for t in range(n_threads):
            acc += local_grads[t, j]
        grad[j] = acc / batch_size

    return grad


def train_numba_sgd(
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
    Rownolegla petla uczenia Mini-Batch SGD wykorzystujaca kompilacje JIT (Numba + prange).
    Zachowuje identyczny przebieg generatora losowego i podzialu na mini-batche jak wariant Baseline.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features, dtype=X_train.dtype)

    # Rozgrzewka kompilatora JIT (warm-up) na malym wycinku przed pomiarem czasu epok
    _ = compute_gradient_numba(X_train[: min(8, n_samples)], y_train[: min(8, n_samples)], weights)

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # 1. Tasowanie zbioru (dokladnie ten sam generator rng co w Baseline)
        indices = rng.permutation(n_samples)
        X_shuffled = np.ascontiguousarray(X_train[indices])
        y_shuffled = np.ascontiguousarray(y_train[indices])

        # 2. Dzielenie na Mini-Batche -> Rownolegly Gradient (Numba) -> Aktualizacja Wag
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            grad = compute_gradient_numba(X_batch, y_batch, weights)
            weights = weights - learning_rate * grad

        # 3. Ewaluacja po epoce i zbieranie historii
        train_probs = predict_proba(X_train, weights)
        epoch_loss = compute_bce_loss(y_train, train_probs)
        epoch_acc = compute_accuracy(y_train, train_probs)

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        elapsed = time.perf_counter() - t0

        if X_val is not None and y_val is not None:
            val_probs = predict_proba(X_val, weights)
            val_loss = compute_bce_loss(y_val, val_probs)
            val_acc = compute_accuracy(y_val, val_probs)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if verbose:
                print(
                    f"[Numba {get_num_threads()}T] Epoch {epoch:03d}/{epochs:03d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                    f"Time: {elapsed:.4f}s"
                )
        else:
            if verbose:
                print(
                    f"[Numba {get_num_threads()}T] Epoch {epoch:03d}/{epochs:03d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                    f"Time: {elapsed:.4f}s"
                )

    return weights, history


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

    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    print(
        f"Start treningu rownoleglego Numba Mini-Batch SGD ({get_num_threads()} watkow CPU): "
        f"train={X_train_final.shape}, test={X_test_final.shape}"
    )

    weights, history = train_numba_sgd(
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

    print("\nTrening rownolegly (Numba) zakonczony pomyslnie!")
    print(f"Poczatkowy Loss (Epoka 1): {history['loss'][0]:.4f} -> Koncowy Loss: {history['loss'][-1]:.4f}")
    print(f"Koncowe Accuracy (Train): {history['accuracy'][-1] * 100:.2f}%")
    if history["val_accuracy"]:
        print(f"Koncowe Accuracy (Test):  {history['val_accuracy'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()

