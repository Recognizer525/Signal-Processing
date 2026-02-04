import numpy as np


def noise_m_step(A, X, Y, Z, eps = 1e-12):
    """
    Реализует М-шаг для поиска неизвестного равномерного шума.

    A: np.ndarray
        Матрица векторов направленности от фиксированного угла.
    X: np.ndarray
        Ковариация E[X X*].
    Y: np.ndarray
        Кросс-ковариация E[X S^*].
    Z: np.ndarray
        Ковариация E[S S^*].
    eps: float
        Минимальное значение диагональных элементов матрицы шума.
    """
    L = X.shape[0]
    res = (1/L) * (np.trace(X) - 2 * np.real(np.trace(A.conj().T @ Y)) + np.trace(A @ Z @ A.conj().T))
    return max(res.real, eps)
