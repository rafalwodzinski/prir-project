from typing import Any
import numpy as np
from numpy.typing import NDArray


def sigmoid(z: NDArray[Any]) -> NDArray[Any]:
    """
    Funkcja sigmoidalna z zabezpieczeniem przed overflow/underflow w exp.
    Dla bardzo duzych lub malych z obcinamy zakres, zeby exp nie rzucal warningow.
    """
    # Dla float32 granica to ~88, dla float64 ~709
    # Ograniczenie do [-85, 85] dla float32 lub [-500, 500] dla float64 daje stabilne [0.0, 1.0]
    if np.issubdtype(z.dtype, np.floating) and z.dtype == np.float32:
        clip_bound = 85.0
    else:
        clip_bound = 500.0

    z_safe = np.clip(z, -clip_bound, clip_bound)
    return 1.0 / (1.0 + np.exp(-z_safe))


def compute_bce_loss(
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    eps: float = 1e-15,
) -> float:
    """
    Binary Cross-Entropy (BCE) loss.
    Obcinamy y_pred do [eps, 1 - eps], zeby nie liczyc log(0) co daloby nan.
    """
    # Zabezpieczenie przed log(0)
    y_pred_safe = np.clip(y_pred, eps, 1.0 - eps)

    loss = -np.mean(
        y_true * np.log(y_pred_safe) + (1.0 - y_true) * np.log(1.0 - y_pred_safe)
    )
    return float(loss)


def compute_gradient(
    X: NDArray[Any],
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
) -> NDArray[Any]:
    """
    Wektorowe obliczenie gradientu dla partii danych:
    grad = X.T @ (y_pred - y_true) / batch_size
    """
    batch_size = X.shape[0]
    error = y_pred - y_true
    return (X.T @ error) / batch_size
