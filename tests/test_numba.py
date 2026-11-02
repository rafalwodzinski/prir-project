import numpy as np

from math_utils import sigmoid, compute_gradient
from sequential_sgd import add_intercept, generate_synthetic_dataset, train_sequential_sgd
from numba_sgd import compute_gradient_numba, train_numba_sgd


def test_numba_gradient_matches_vectorized_gradient():
    """
    Sprawdza czy jawna petla pod @njit(parallel=True) z prange i lokalnymi
    akumulatorami watkow zwraca dokladnie ten sam wektor gradientu co wersja wektorowa.
    """
    rng = np.random.default_rng(42)
    X_raw = rng.normal(0.0, 1.0, size=(128, 16)).astype(np.float32)
    X = add_intercept(X_raw)
    y_true = rng.integers(0, 2, size=128).astype(np.float32)
    weights = rng.normal(0.0, 0.5, size=X.shape[1]).astype(np.float32)

    # Gradient wektorowy (Baseline)
    y_pred = sigmoid(X @ weights)
    grad_vec = compute_gradient(X, y_true, y_pred)

    # Gradient rownolegly (Numba JIT + prange + local accumulators)
    grad_numba = compute_gradient_numba(X, y_true, weights)

    assert grad_numba.shape == grad_vec.shape
    assert grad_numba.dtype == grad_vec.dtype
    assert np.allclose(grad_numba, grad_vec, atol=1e-5)


def test_numba_parallel_weights_match_baseline_after_10_epochs():
    """
    DoD Issue #5:
    - Kod kompiluje sie JIT bez bledow.
    - Wagi koncowe modelu rownoleglego po 10 epokach (przy identycznym ziarnie losowosci)
      zgadzaja sie (np.allclose) z wynikami z wariantu Baseline.
    """
    X, y = generate_synthetic_dataset(n_samples=1024, n_features=12, seed=42, dtype=np.float32)
    X_with_bias = add_intercept(X)

    epochs = 10
    batch_size = 64
    lr = 0.05
    seed = 2026

    # 1. Trening wariantu Baseline (sekwencyjny NumPy)
    weights_baseline, history_baseline = train_sequential_sgd(
        X_with_bias,
        y,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        verbose=False,
    )

    # 2. Trening wariantu Rownoleglego (Numba @njit(parallel=True) + prange)
    weights_numba, history_numba = train_numba_sgd(
        X_with_bias,
        y,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
        verbose=False,
    )

    assert weights_numba.shape == weights_baseline.shape
    assert np.allclose(weights_numba, weights_baseline, atol=1e-5)
    assert np.allclose(history_numba["loss"], history_baseline["loss"], atol=1e-5)
    assert np.allclose(history_numba["accuracy"], history_baseline["accuracy"], atol=1e-5)

