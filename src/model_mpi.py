from pathlib import Path
import time
from typing import Any
import numpy as np
from numpy.typing import NDArray

# Bezpieczny import mpi4py: jesli w systemie brak biblioteki C MPI (msmpi.dll),
# tworzymy prosta atrape jednowatkowa, zeby testy i kod nie rzucaly bledu importu
try:
    from mpi4py import MPI
except (ImportError, RuntimeError):
    class DummyComm:
        def Get_rank(self) -> int:
            return 0
        def Get_size(self) -> int:
            return 1
        def Allreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[...] = sendbuf[...]
        def Barrier(self):
            pass

    class DummyMPI:
        COMM_WORLD = DummyComm()
        SUM = "SUM"

    MPI = DummyMPI()  # type: ignore

from data_loader import load_and_preprocess_data, split_dataset, scale_features
from math_utils import sigmoid, compute_bce_loss
from sequential_sgd import add_intercept, compute_accuracy, predict_proba, generate_synthetic_dataset


def split_batch_for_rank(
    X_batch: NDArray[Any],
    y_batch: NDArray[Any],
    rank: int,
    size: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Dzieli globalny mini-batch (B_global) na rozlaczne czesci lokalne (B_local)
    dla procesu o zadanym identyfikatorze `rank` z `size` procesow w komunikatorze.
    Obsluguje rowniez przypadek, gdy rozmiar ostatniego mini-batcha nie dzieli sie
    calkowicie przez liczbe procesow `size`.
    """
    global_bs = X_batch.shape[0]
    base = global_bs // size
    rem = global_bs % size

    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)

    return X_batch[start:end], y_batch[start:end]


def train_mpi_sgd(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_val: NDArray[Any] | None = None,
    y_val: NDArray[Any] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    seed: int = 42,
    comm: Any | None = None,
    verbose: bool = True,
) -> tuple[NDArray[Any], dict[str, list[float]]]:
    """
    Rozproszona petla uczenia Synchronous Data-Parallel Mini-Batch SGD w standardzie MPI.
    Kazdy proces przetwarza swoja czesc partii i po KAZDYM batchu synchronizuje gradient
    przez comm.Allreduce.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features, dtype=X_train.dtype)
    local_grad_sum = np.zeros(n_features, dtype=X_train.dtype)
    global_grad_sum = np.zeros(n_features, dtype=X_train.dtype)

    history: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        indices = rng.permutation(n_samples)
        X_shuffled = np.ascontiguousarray(X_train[indices])
        y_shuffled = np.ascontiguousarray(y_train[indices])

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_global_batch = X_shuffled[start_idx:end_idx]
            y_global_batch = y_shuffled[start_idx:end_idx]
            current_global_bs = X_global_batch.shape[0]

            X_local, y_local = split_batch_for_rank(X_global_batch, y_global_batch, rank, size)

            if X_local.shape[0] > 0:
                y_pred_local = sigmoid(X_local @ weights)
                error_local = y_pred_local - y_local
                local_grad_sum[:] = X_local.T @ error_local
            else:
                local_grad_sum.fill(0.0)

            # Synchronizacja gradientu po kazdym pojedynczym batchu
            comm.Allreduce(local_grad_sum, global_grad_sum, op=MPI.SUM)
            grad = global_grad_sum / current_global_bs

            weights = weights - learning_rate * grad

        if rank == 0:
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
                        f"[MPI Sync {size}P | rank=0] Epoch {epoch:03d}/{epochs:03d} | "
                        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                        f"Time: {elapsed:.4f}s"
                    )
            else:
                if verbose:
                    print(
                        f"[MPI Sync {size}P | rank=0] Epoch {epoch:03d}/{epochs:03d} | "
                        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                        f"Time: {elapsed:.4f}s"
                    )

    return weights, history


def train_local_sgd_mpi(
    X_train: NDArray[Any],
    y_train: NDArray[Any],
    X_val: NDArray[Any] | None = None,
    y_val: NDArray[Any] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    sync_freq_k: int = 4,
    seed: int = 42,
    comm: Any | None = None,
    verbose: bool = True,
) -> tuple[NDArray[Any], dict[str, Any]]:
    """
    Local SGD z rzadka synchronizacja modeli (Issue #7).
    - Brak Allreduce na poziomie gradientu przy pojedynczej partii.
    - Wezly aktualizuja wagi lokalnie na swoich czesciach danych.
    - Dopiero co K krokow wywolywane jest comm.Allreduce sumujace i usredniajace wagi modeli.
    """
    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(seed)
    n_samples, n_features = X_train.shape

    # Kazdy wezel ma wlasny wektor wag
    weights = np.zeros(n_features, dtype=X_train.dtype)
    global_weights_sum = np.zeros(n_features, dtype=X_train.dtype)

    history: dict[str, Any] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "sync_count": 0,
        "epoch_times": [],
    }

    step_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        indices = rng.permutation(n_samples)
        X_shuffled = np.ascontiguousarray(X_train[indices])
        y_shuffled = np.ascontiguousarray(y_train[indices])

        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            X_global_batch = X_shuffled[start_idx:end_idx]
            y_global_batch = y_shuffled[start_idx:end_idx]

            X_local, y_local = split_batch_for_rank(X_global_batch, y_global_batch, rank, size)

            if X_local.shape[0] > 0:
                # 1. Lokalna predykcja i lokalny gradient (brak komunikacji sieciowej)
                y_pred_local = sigmoid(X_local @ weights)
                error_local = y_pred_local - y_local
                grad_local = (X_local.T @ error_local) / X_local.shape[0]

                # 2. Lokalna aktualizacja wag na wezle
                weights = weights - learning_rate * grad_local

            step_counter += 1

            # 3. Synchronizacja modeli co K krokow przez Allreduce na wektorach wag
            if sync_freq_k > 0 and step_counter % sync_freq_k == 0:
                comm.Allreduce(weights, global_weights_sum, op=MPI.SUM)
                weights[:] = global_weights_sum / size
                history["sync_count"] += 1

        # Upewniamy sie, ze na koniec epoki modele sa zsynchronizowane
        if sync_freq_k > 0 and step_counter % sync_freq_k != 0:
            comm.Allreduce(weights, global_weights_sum, op=MPI.SUM)
            weights[:] = global_weights_sum / size
            history["sync_count"] += 1

        elapsed = time.perf_counter() - t0
        history["epoch_times"].append(elapsed)

        if rank == 0:
            train_probs = predict_proba(X_train, weights)
            epoch_loss = compute_bce_loss(y_train, train_probs)
            epoch_acc = compute_accuracy(y_train, train_probs)

            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            if X_val is not None and y_val is not None:
                val_probs = predict_proba(X_val, weights)
                val_loss = compute_bce_loss(y_val, val_probs)
                val_acc = compute_accuracy(y_val, val_probs)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if verbose:
                    print(
                        f"[MPI Local-SGD (K={sync_freq_k}) {size}P | rank=0] Epoch {epoch:03d}/{epochs:03d} | "
                        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                        f"Val Acc: {val_acc * 100:.2f}% | "
                        f"Syncs so far: {history['sync_count']} | Time: {elapsed:.4f}s"
                    )
            else:
                if verbose:
                    print(
                        f"[MPI Local-SGD (K={sync_freq_k}) {size}P | rank=0] Epoch {epoch:03d}/{epochs:03d} | "
                        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}% | "
                        f"Syncs so far: {history['sync_count']} | Time: {elapsed:.4f}s"
                    )

    return weights, history


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data_path = Path(__file__).resolve().parent.parent / "data" / "csgo_round_snapshots.csv"

    if data_path.is_file():
        if rank == 0:
            print(f"[MPI rank=0/{size}] Wczytywanie danych CS:GO z: {data_path}")
        X, y = load_and_preprocess_data(data_path, dtype=np.float32)
    else:
        if rank == 0:
            print(f"[MPI rank=0/{size}] Brak pliku {data_path} - uzywam syntetycznego zbioru danych.")
        X, y = generate_synthetic_dataset(n_samples=4000, n_features=12, seed=42, dtype=np.float32)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, seed=42)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    X_train_final = add_intercept(X_train_scaled)
    X_test_final = add_intercept(X_test_scaled)

    k_val = 4
    if rank == 0:
        print(
            f"\nStart treningu Local SGD MPI (K={k_val}, {size} procesy): "
            f"train={X_train_final.shape}, test={X_test_final.shape}"
        )

    weights, history = train_local_sgd_mpi(
        X_train_final,
        y_train,
        X_val=X_test_final,
        y_val=y_test,
        epochs=15,
        batch_size=256,
        learning_rate=0.01,
        sync_freq_k=k_val,
        seed=42,
        comm=comm,
        verbose=(rank == 0),
    )

    if rank == 0:
        print("\nTrening Local SGD zakonczony!")
        print(f"Liczba wykonanych synchronizacji Allreduce: {history['sync_count']}")
        print(f"Koncowy Loss: {history['loss'][-1]:.4f} | Koncowe Accuracy: {history['accuracy'][-1] * 100:.2f}%")
        if history["val_accuracy"]:
            print(f"Koncowe Accuracy (Test): {history['val_accuracy'][-1] * 100:.2f}%")


if __name__ == "__main__":
    main()
