from typing import Any
import numpy as np
from numpy.typing import NDArray

from math_utils import sigmoid, compute_bce_loss, compute_gradient
from sequential_sgd import add_intercept, train_sequential_sgd, generate_synthetic_dataset


def compute_numerical_gradient(
    X: NDArray[Any],
    y_true: NDArray[Any],
    weights: NDArray[Any],
    eps: float = 1e-5,
) -> NDArray[Any]:
    """
    Numeryczna aproksymacja gradientu za pomoca roznic skonczonych centralnych:
        dL / dW_j ≈ (L(W + eps * e_j) - L(W - eps * e_j)) / (2 * eps)

    Dla kazdej wagi W[j] sprawdzamy jak zmieni sie Loss, gdy lekko "pchniemy"
    wlasnie te jedna wage w prawo (+eps) i w lewo (-eps).
    """
    grad_num = np.zeros_like(weights, dtype=np.float64)

    for j in range(weights.shape[0]):
        w_plus = weights.astype(np.float64, copy=True)
        w_minus = weights.astype(np.float64, copy=True)

        w_plus[j] += eps
        w_minus[j] -= eps

        y_pred_plus = sigmoid(X @ w_plus)
        y_pred_minus = sigmoid(X @ w_minus)

        loss_plus = compute_bce_loss(y_true, y_pred_plus)
        loss_minus = compute_bce_loss(y_true, y_pred_minus)

        grad_num[j] = (loss_plus - loss_minus) / (2.0 * eps)

    return grad_num


def test_analytical_vs_numerical_gradient_small_vector():
    """
    Numeryczny dowod dzialania wektorowego gradientu (Gradient Checking)
    dla malego wektora wejsciowego przy uzyciu roznic skonczonych centralnych.
    DoD: np.allclose(grad_analityczny, grad_numeryczny, atol=1e-5).
    """
    rng = np.random.default_rng(42)

    # Maly zbior wejsciowy: 8 probek, 4 cechy (+ 1 kolumna jedynek jako bias = 5 wag)
    X_raw = rng.normal(0.0, 1.0, size=(8, 4)).astype(np.float64)
    X = add_intercept(X_raw)
    y_true = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    weights = rng.normal(0.0, 0.5, size=(X.shape[1],)).astype(np.float64)

    # 1. Gradient analityczny (wektorowy)
    y_pred = sigmoid(X @ weights)
    grad_analityczny = compute_gradient(X, y_true, y_pred)

    # 2. Gradient numeryczny (roznice skonczone centralne)
    grad_numeryczny = compute_numerical_gradient(X, y_true, weights, eps=1e-5)

    assert grad_analityczny.shape == grad_numeryczny.shape
    assert np.allclose(grad_analityczny, grad_numeryczny, atol=1e-5)


def test_sequential_sgd_minimizes_loss_and_collects_history(capsys):
    """
    Sprawdza czy skrypt sekwencyjny stabilnie minimalizuje Loss,
    zbiera historie Loss i Accuracy z epok do list oraz loguje postep na konsole.
    """
    X, y = generate_synthetic_dataset(n_samples=600, n_features=6, seed=123, dtype=np.float32)
    X_with_bias = add_intercept(X)

    epochs = 15
    weights, history = train_sequential_sgd(
        X_with_bias,
        y,
        epochs=epochs,
        batch_size=32,
        learning_rate=0.1,
        seed=123,
        verbose=True,
    )

    # Sprawdzenie wymiarow wyuczonych wag i historii
    assert weights.shape == (X_with_bias.shape[1],)
    assert len(history["loss"]) == epochs
    assert len(history["accuracy"]) == epochs

    # Sprawdzenie czy Loss stabilnie spada, a Accuracy rosnie
    assert history["loss"][-1] < history["loss"][0]
    assert history["accuracy"][-1] > history["accuracy"][0]
    assert history["accuracy"][-1] > 0.75

    # Sprawdzenie czy po kazdej epoce na konsole trafia log
    captured = capsys.readouterr()
    output_lines = [line for line in captured.out.strip().splitlines() if "Epoch" in line]
    assert len(output_lines) == epochs

