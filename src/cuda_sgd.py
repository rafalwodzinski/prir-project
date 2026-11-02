import math
import os
from pathlib import Path
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray

# W razie braku GPU wlaczamy cudasim
try:
    from numba import cuda
    if not cuda.is_available():
        os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
        from numba import cuda
except Exception:
    os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
    from numba import cuda

from data_loader import load_and_preprocess_data, split_dataset, scale_features
from math_utils import compute_bce_loss
from sequential_sgd import add_intercept, compute_accuracy, predict_proba, generate_synthetic_dataset


@cuda.jit
def compute_errors_cuda_kernel(
    d_X: Any,
    d_y: Any,
    d_weights: Any,
    d_errors: Any,
    d_indices: Any,
    batch_start: int,
    batch_size: int,
    n_features: int,
) -> None:
    """
    Rownolegle liczenie predykcji i bledow na GPU.
    Kazdy watek liczy jedna probke z mini-batcha z uzyciem przetasowanego indeksu wiersza.
    """
    i = cuda.grid(1)

    if i < batch_size:
        row = d_indices[batch_start + i]

        z = 0.0
        for j in range(n_features):
            z += d_X[row, j] * d_weights[j]

        # Ochrona przed overflowem
        if z < -85.0:
            z_safe = -85.0
        elif z > 85.0:
            z_safe = 85.0
        else:
            z_safe = z

        pred = 1.0 / (1.0 + math.exp(-z_safe))
        d_errors[i] = pred - d_y[row]


@cuda.jit
def compute_gradient_cuda_kernel(
    d_X: Any,
    d_errors: Any,
    d_grad: Any,
    d_indices: Any,
    batch_start: int,
    batch_size: int,
    n_features: int,
) -> None:
    """
    Rownolegle liczenie gradientu dla kazdej cechy j z uzyciem przetasowanych indeksow.
    """
    j = cuda.grid(1)

    if j < n_features:
        acc = 0.0
        for i in range(batch_size):
            row = d_indices[batch_start + i]
            acc += d_errors[i] * d_X[row, j]
        d_grad[j] = acc / batch_size


@cuda.jit
def update_weights_cuda_kernel(
    d_weights: Any,
    d_grad: Any,
    learning_rate: float,
    n_features: int,
) -> None:
    """Aktualizacja wag bezposrednio w pamieci GPU."""
    j = cuda.grid(1)
    if j < n_features:
        d_weights[j] -= learning_rate * d_grad[j]


def train_cuda_sgd(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_val: NDArray[Any] | None = None,
    y_val: NDArray[Any] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    threads_per_block: int = 128,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[NDArray[Any], dict[str, list[float]]]:
    """
    Mini-batch SGD na GPU z uzyciem Numba CUDA.
    Kluczowa optymalizacja: dataset wysylamy do VRAM tylko raz na starcie.
    """
    n_samples, n_features = X_train.shape
    rng = np.random.default_rng(seed)

    # Przesylamy dane do pamieci GPU tylko raz przed petla epok
    t_transfer_start = time.perf_counter()
    d_X = cuda.to_device(np.ascontiguousarray(X_train, dtype=np.float32))
    d_y = cuda.to_device(np.ascontiguousarray(y_train, dtype=np.float32))

    # Alokujemy wagi i bufory pomocnicze od razu na urzadzeniu
    weights_init = np.zeros(n_features, dtype=np.float32)
    d_weights = cuda.to_device(weights_init)
    d_errors = cuda.device_array(batch_size, dtype=np.float32)
    d_grad = cuda.device_array(n_features, dtype=np.float32)
    d_indices = cuda.device_array(n_samples, dtype=np.int32)

    transfer_init_time = time.perf_counter() - t_transfer_start
    if verbose:
        print(f"Dane w VRAM (czas transferu PCIe: {transfer_init_time:.4f}s)")

    # Obliczamy liczbe blokow
    blocks_errors = (batch_size + threads_per_block - 1) // threads_per_block
    blocks_features = (n_features + threads_per_block - 1) // threads_per_block

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_times": [],
        "epoch_time": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Tasowanie indeksow i przeslanie lekkiej tablicy int32 do GPU
        indices = rng.permutation(n_samples).astype(np.int32)
        d_indices.copy_to_device(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            current_bs = end_idx - start_idx

            if current_bs <= 0:
                continue

            current_blocks_errors = (current_bs + threads_per_block - 1) // threads_per_block

            # Odpalamy kernele bez zadnego kopiowania danych macierzy X miedzy CPU a GPU
            compute_errors_cuda_kernel[current_blocks_errors, threads_per_block](
                d_X, d_y, d_weights, d_errors, d_indices, start_idx, current_bs, n_features
            )

            compute_gradient_cuda_kernel[blocks_features, threads_per_block](
                d_X, d_errors, d_grad, d_indices, start_idx, current_bs, n_features
            )

            update_weights_cuda_kernel[blocks_features, threads_per_block](
                d_weights, d_grad, learning_rate, n_features
            )

        elapsed = time.perf_counter() - t0
        history["epoch_times"].append(elapsed)
        history["epoch_time"].append(elapsed)

        # Pobieramy same wagi na koniec epoki zeby policzyc metryki
        current_weights = d_weights.copy_to_host()

        train_probs = predict_proba(X_train, current_weights)
        epoch_loss = compute_bce_loss(y_train, train_probs)
        epoch_acc = compute_accuracy(y_train, train_probs)

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        if X_val is not None and y_val is not None:
            val_probs = predict_proba(X_val, current_weights)
            val_loss = compute_bce_loss(y_val, val_probs)
            val_acc = compute_accuracy(y_val, val_probs)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if verbose:
                print(
                    f"Epoka {epoch:02d}/{epochs:02d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                    f"Val Acc: {val_acc * 100:.2f}% | Czas: {elapsed:.4f}s"
                )
        else:
            if verbose:
                print(
                    f"Epoka {epoch:02d}/{epochs:02d} | "
                    f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | Czas: {elapsed:.4f}s"
                )

    final_weights = d_weights.copy_to_host()
    return final_weights, history


def main() -> None:
    data_path = Path(__file__).resolve().parent.parent / "data" / "csgo_round_snapshots.csv"

    if data_path.is_file():
        print(f"Wczytywanie: {data_path}")
        X, y = load_and_preprocess_data(data_path, dtype=np.float32)
    else:
        print("Brak pliku CSV - uzywam syntetycznego zbioru danych.")
        X, y = generate_synthetic_dataset(n_samples=4000, n_features=12, seed=42, dtype=np.float32)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    print(f"Trening CUDA: train={X_train_final.shape}, test={X_test_final.shape}, batch=256")
    weights, history = train_cuda_sgd(
        X_train_final,
        y_train,
        X_val=X_test_final,
        y_val=y_test,
        epochs=15,
        batch_size=256,
        learning_rate=0.01,
        seed=42,
        verbose=True,
    )

    print(f"\nKoniec! Poczatkowy Loss: {history['loss'][0]:.4f} -> Koncowy: {history['loss'][-1]:.4f}")
    print(f"Accuracy (Train): {history['accuracy'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
