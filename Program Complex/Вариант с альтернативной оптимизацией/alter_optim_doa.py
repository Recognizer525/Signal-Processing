import numpy as np
import torch
from scipy.optimize import dual_annealing

import sensors

DIST_RATIO = 0.5


def cost(theta: np.array,
         X: np.array, 
         S: np.array, 
         Q_inv_sqrt: np.array) -> np.array:
    """
    Вычисляет значение фробениусовой нормы ||Q^{-1/2}(X-AS)||^2_F, 
    которая подлежит минимизации.

    Parameters
    ---------------------------------------------------------------------------
    theta: np.array
        Оценка DoA.
    X: np.array
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    S: np.array
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt: np.array
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.

    Returns
    ---------------------------------------------------------------------------
    res: np.array
        Значение фробениусовой нормы ||Q^{-1/2}(X-AS)||^2_F.
    """
    A = sensors.A_ULA(X.shape[0], theta)
    #print(f"shapes are X={X.shape}, S={S.shape}, theta={theta.shape}")
    #print(f"Q_inv_sqrt={Q_inv_sqrt.shape}")
    #print(f"A={A.shape}")
    E = Q_inv_sqrt @ (X - A @ S) 
    return np.linalg.norm(E, 'fro')**2


def CM_step_theta(X: np.ndarray, 
                  theta: np.ndarray, 
                  S: np.ndarray, 
                  Q_inv_sqrt: np.ndarray, 
                  method: str = 'SLSQP') -> np.ndarray:
    """
    Функция предназначена для поиска оценки DoA, которая минимизирует норму
    ||Q^{-1/2}(X-AS)||^2_F.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    theta: np.ndarray
        Оценка DoA, полученная на предыдущей итерации,
        либо же начальная оценка DoA.
    S: np.ndarray
        Текущая оценка детерминированных исходных сигналов.
    Q_inv_sqrt: np.ndarray
        Квадратный корень от матрицы, обратной к ковариационной матрице шума.
    method: str
        Метод оптимизации функции потерь для DoA.

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Полученная в ходе процесса оптимизации наилучшая оценка вектора DoA.
    """
    K = theta.shape[0]
    #print(f"K={K}")
    bounds = [(-np.pi, np.pi) for _ in range(K)]
    #print(f"shapes are X={X.shape}, S={S.shape}, theta={theta.shape}")
    #print(f"Q_inv_sqrt={Q_inv_sqrt.shape}")
    def fun(theta):
        return cost(theta, X, S, Q_inv_sqrt)
    res = dual_annealing(fun, bounds)
    #print(res.fun)
    return res.x