import warnings
import numpy as np
import pytest

from math_utils import sigmoid, compute_bce_loss, compute_gradient


def test_sigmoid_overflow_protection():
    """
    Sprawdzenie czy sigmoid radzi sobie ze skrajnymi wartosciami [-1000, 1000]
    bez rzucania zadnych ostrzezen o overflowie w exp.
    """
    z = np.array([-1000.0, 1000.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sigmoid(z)

    assert np.allclose(res, [0.0, 1.0])


def test_sigmoid_float32_and_zeros():
    """Sprawdzenie typow float32 oraz standardowej wartosci dla z=0."""
    z_zero = np.array([0.0])
    assert np.isclose(sigmoid(z_zero)[0], 0.5)

    # float32 ze skrajnymi wartosciami tez nie powinien rzucac warningow
    z_f32 = np.array([-1000.0, 1000.0], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res_f32 = sigmoid(z_f32)

    assert np.allclose(res_f32, [0.0, 1.0])


def test_bce_loss_stability_with_zeros_and_ones():
    """
    Sprawdzenie czy funkcja straty nie zwraca nan ani inf
    dla skrajnych predykcji rownych dokladnie 0.0 lub 1.0.
    """
    y_true = np.array([1.0, 0.0, 1.0, 0.0])

    # Same zera i same jedynki
    y_pred_zeros = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred_ones = np.array([1.0, 1.0, 1.0, 1.0])

    loss_zeros = compute_bce_loss(y_true, y_pred_zeros)
    loss_ones = compute_bce_loss(y_true, y_pred_ones)

    assert not np.isnan(loss_zeros)
    assert not np.isinf(loss_zeros)
    assert not np.isnan(loss_ones)
    assert not np.isinf(loss_ones)
    assert loss_zeros > 0.0
    assert loss_ones > 0.0


def test_bce_loss_perfect_prediction():
    """Dla idealnych predykcji strata powinna byc bliska 0."""
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.9999, 0.0001, 0.9999])

    loss = compute_bce_loss(y_true, y_pred)
    assert loss < 0.01


def test_compute_gradient_shape_and_values():
    """
    Sprawdzenie wymiarow i wartosci wektora gradientu:
    grad = X.T @ (y_pred - y_true) / batch_size
    """
    # 3 probki, 2 cechy
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([1.0, 0.0, 1.0])  # idealne dopasowanie -> blad = 0

    grad_zero = compute_gradient(X, y_true, y_pred)
    assert grad_zero.shape == (2,)
    assert np.allclose(grad_zero, [0.0, 0.0])

    # Celowy blad: predykcja o 1 za duza dla kazdej probki
    y_pred_err = y_true + 1.0
    grad = compute_gradient(X, y_true, y_pred_err)
    # error = [1, 1, 1], X.T @ error / 3 to po prostu srednia po kolumnach X: [3.0, 4.0]
    assert np.allclose(grad, [3.0, 4.0])
