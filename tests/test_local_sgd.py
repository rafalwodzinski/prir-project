import numpy as np
import pytest

from sequential_sgd import add_intercept, generate_synthetic_dataset
from model_mpi import (
    MPI,
    local_sgd_step,
    run_local_sgd_epoch_simulated,
    train_local_sgd_mpi,
)


def test_local_sgd_convergence():
    """
    Sprawdza czy Local SGD (K=4) zbiega stabilnie:
    - koncowa strata powinna byc wyraznie mniejsza od poczatkowej,
    - dokladnosc powinna przekroczyc 80%.
    """
    X, y = generate_synthetic_dataset(n_samples=600, n_features=8, seed=42, dtype=np.float32)
    X_b = add_intercept(X)

    weights, history = train_local_sgd_mpi(
        X_b,
        y,
        epochs=15,
        batch_size=64,
        learning_rate=0.05,
        sync_freq_k=4,
        seed=123,
        verbose=False,
    )

    assert history["loss"][-1] < history["loss"][0], "Loss powinien malec w trakcie treningu"
    assert history["accuracy"][-1] > 0.80, "Accuracy powinno osiagnac co najmniej 80%"
    assert not np.isnan(weights).any(), "Wagi nie moga zawierac wartosci NaN"


def test_local_sgd_reduces_communication():
    """
    Weryfikuje czy parametr K rzeczywiscie ogranicza liczbe synchronizacji:
    Dla K=4 liczba synchronizacji Allreduce powinna byc okolo 4x mniejsza niz dla K=1.
    """
    X, y = generate_synthetic_dataset(n_samples=640, n_features=6, seed=42, dtype=np.float32)
    X_b = add_intercept(X)

    # 640 probek, batch_size=64 -> 10 batchy na epoke. 5 epok -> 50 batchy lacznie.
    _, hist_k1 = train_local_sgd_mpi(
        X_b, y, epochs=5, batch_size=64, learning_rate=0.01, sync_freq_k=1, seed=42, verbose=False
    )
    _, hist_k4 = train_local_sgd_mpi(
        X_b, y, epochs=5, batch_size=64, learning_rate=0.01, sync_freq_k=4, seed=42, verbose=False
    )

    # Przy K=1 synchronizujemy co kazdy batch (50 razy)
    # Przy K=4 synchronizujemy co 4 batche (okolo 13-15 razy)
    assert hist_k1["sync_count"] >= 50
    assert hist_k4["sync_count"] < hist_k1["sync_count"] // 2
    assert hist_k4["sync_count"] <= 15


def test_local_sgd_different_k_values():
    """
    Testuje rozne wartosci K (K=1, K=2, K=8) sprawdzajac brak bledow i stabilnosc numeryczna.
    """
    X, y = generate_synthetic_dataset(n_samples=400, n_features=6, seed=42, dtype=np.float32)
    X_b = add_intercept(X)

    for k in [1, 2, 8]:
        w, h = train_local_sgd_mpi(
            X_b, y, epochs=5, batch_size=64, learning_rate=0.03, sync_freq_k=k, seed=42, verbose=False
        )
        assert len(h["loss"]) == 5
        assert h["loss"][-1] < h["loss"][0]
        assert not np.isnan(w).any()


def test_weight_averaging_math():
    """
    Weryfikuje dokladna matematyke usredniania wag (Allreduce z op=MPI.SUM podzielone przez size).
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    w = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    global_sum = np.zeros_like(w)

    comm.Allreduce(w, global_sum, op=MPI.SUM)
    avg_w = global_sum / size

    assert np.allclose(avg_w, [2.0, 4.0, 6.0])


def test_local_sgd_step_and_simulated_epoch_match_mpi():
    """
    Sprawdza, ze wydzielone prymitywy (local_sgd_step i run_local_sgd_epoch_simulated)
    obsluguja pusty batch oraz dla 1 workera daja identyczne wagi i liczbe synchronizacji
    co train_local_sgd_mpi (1 proces).
    """
    X, y = generate_synthetic_dataset(n_samples=256, n_features=6, seed=42, dtype=np.float32)
    X_b = add_intercept(X)

    # Pusty batch nie zmienia wag
    w0 = np.ones(X_b.shape[1], dtype=np.float32)
    w_empty = local_sgd_step(X_b[:0], y[:0], w0, learning_rate=0.05)
    assert np.array_equal(w_empty, w0)

    # Porownanie 1-procesowego train_local_sgd_mpi z run_local_sgd_epoch_simulated (1 worker)
    w_mpi, h_mpi = train_local_sgd_mpi(
        X_b, y, epochs=3, batch_size=64, learning_rate=0.05, sync_freq_k=4, seed=77, verbose=False
    )

    rng = np.random.default_rng(77)
    worker_weights = [np.zeros(X_b.shape[1], dtype=np.float32)]
    step_counter = 0
    total_syncs = 0
    for _ in range(3):
        idx = rng.permutation(X_b.shape[0])
        step_counter, syncs = run_local_sgd_epoch_simulated(
            np.ascontiguousarray(X_b[idx]),
            np.ascontiguousarray(y[idx]),
            worker_weights,
            batch_size=64,
            learning_rate=0.05,
            sync_freq_k=4,
            step_counter=step_counter,
        )
        total_syncs += syncs

    assert total_syncs == h_mpi["sync_count"]
    assert np.allclose(worker_weights[0], w_mpi, atol=1e-6)
