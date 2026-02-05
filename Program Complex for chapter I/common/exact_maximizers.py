import numpy as np

def opt_noise(A: np.ndarray,
              Sigma_XX: np.ndarray,
              Sigma_XS: np.ndarray,
              Sigma_SS: np.ndarray,
              eps: float) -> float:
    """
    М-шаг для равномерного шума.
    A: np.ndarray
        Матрица векторов направленности.
    Sigma_XX: np.ndarray
        Условный второй момент наблюдений.
    Sigma_XS: np.ndarray
        Условный смешанный второй момент наблюдений и сигналов.
    Sigma_SS: np.ndarray
        Условный второй момент сигналов.
    """
    L = Sigma_XX.shape[0]
    ans = np.trace(Sigma_XX) - 2 * np.real(np.trace(A.conj().T @ Sigma_XS)) + np.trace(A @ Sigma_SS @ A.conj().T)
    return max(np.real(ans), eps)


def MAP_est_of_P(Sigma_SS: np.ndarray, 
                 Psi: np.ndarray, 
                 nu: int, 
                 n: int) -> np.ndarray:
    """
    MAP-оценка ковариации на М-шаге ЕМ-алгоритма.
    Использует обратное распределение Уисхарта.

    Sigma_SS: np.ndarray
        Условный второй момент сигналов.
    Psi: np.ndarray
        Априорная информация о ковариации.
    nu: np.ndarray
        Степень уверенности.
    n: np.ndarray
        Количество наблюдений.
    """
    d = Sigma_SS.shape[0]
    return (Sigma_SS + Psi) / (n + nu + d + 1)
