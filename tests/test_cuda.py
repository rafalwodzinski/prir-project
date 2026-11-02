import os
import numpy as np
import pytest

try:
    from numba import cuda
    cuda_available = cuda.is_available()
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(
    not cuda_available,
    reason="Brak dostepu do urzadzenia CUDA lub symulatora CUDASIM",
)

from sequential_sgd import add_intercept, generate_synthetic_dataset
from cuda_sgd import (
    compute_errors_cuda_kernel,
    compute_gradient_cuda_kernel,
    update_weights_cuda_kernel,
    train_cuda_sgd,
)


def test_cuda_kernels_execute_without_memory_error():
    """
    DoD 1: Sprawdza czy jadra CUDA wykonuja sie bez bledow pamieci
    (brak Out-Of-Bounds, poprawne indeksowanie siatki i blokow).
    """
    batch_size = 32
    n_features = 4
    threads_per_block = 16

    X = np.ones((batch_size, n_features), dtype=np.float32)
    y = np.ones(batch_size, dtype=np.float32)
    weights = np.zeros(n_features, dtype=np.float32)

    d_X = cuda.to_device(X)
    d_y = cuda.to_device(y)
    d_weights = cuda.to_device(weights)
    d_errors = cuda.device_array(batch_size, dtype=np.float32)
    d_grad = cuda.device_array(n_features, dtype=np.float32)
    d_indices = cuda.to_device(np.arange(batch_size, dtype=np.int32))

    blocks_b = (batch_size + threads_per_block - 1) // threads_per_block
    blocks_f = (n_features + threads_per_block - 1) // threads_per_block

    # Wywolanie jadra bledow
    compute_errors_cuda_kernel[blocks_b, threads_per_block](
        d_X, d_y, d_weights, d_errors, d_indices, 0, batch_size, n_features
    )

    # Wywolanie jadra gradientu
    compute_gradient_cuda_kernel[blocks_f, threads_per_block](
        d_X, d_errors, d_grad, d_indices, 0, batch_size, n_features
    )

    # Wywolanie jadra aktualizacji wag
    update_weights_cuda_kernel[blocks_f, threads_per_block](
        d_weights, d_grad, 0.01, n_features
    )

    errors = d_errors.copy_to_host()
    grad = d_grad.copy_to_host()
    w_updated = d_weights.copy_to_host()

    assert errors.shape == (batch_size,)
    assert grad.shape == (n_features,)
    assert w_updated.shape == (n_features,)
    assert not np.isnan(grad).any()


def test_cuda_sgd_convergence():
    """
    DoD 2: Weryfikuje czy petla uczenia CUDA zbiega stabilnie:
    - koncowy Loss jest mniejszy niz poczatkowy,
    - Accuracy osiaga wysoki poziom (> 80%),
    - wagi nie zawieraja wartosci NaN.
    """
    X, y = generate_synthetic_dataset(n_samples=500, n_features=6, seed=42, dtype=np.float32)
    X_b = add_intercept(X)

    weights, history = train_cuda_sgd(
        X_b,
        y,
        epochs=10,
        batch_size=64,
        learning_rate=0.05,
        threads_per_block=32,
        seed=123,
        verbose=False,
    )

    assert history["loss"][-1] < history["loss"][0], "Loss w wersji CUDA powinien malec"
    assert history["accuracy"][-1] > 0.80, "Accuracy powinno przekroczyc 80%"
    assert not np.isnan(weights).any(), "Wagi nie moga zawierac wartosci NaN"

