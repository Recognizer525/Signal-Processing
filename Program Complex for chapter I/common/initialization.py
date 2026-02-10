import numpy as np

from . import diff_sensor_structures as dss
from . import sensors

def init_est_kn1(K: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 theta_guess: np.ndarray,
                 L: int| None = None,
                 iter: int|None = None,
                 eps: float = 1e-3,
                 seed: int|None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Создает первоначальную оценку DoA и ковариационной матрицы 
    исходных сигналов. Ориентируется на имеющуюся грубую оценку угловых координат, 
    связывает начальные оценки мощности источников с их угловыми координатами.

    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    Q: np.ndarray
        Ковариация шума.
    R: np.ndarray
        Оценка ковариации наблюдений.
    theta_guess: np.ndarray
        Текущая начальная оценка углов.
    L: int|None
        Количество сенсоров в антенной решетке.
    iter: int|None
        Номер итерации мультистарта. Влияет на то, как будет выбрана начальная оценка углов.
    eps: float
        Минимальное значение мощности источника.
    seed: int|None
        Randomstate для генерации данных.

    Returns
    ---------------------------------------------------------------------------
    theta: np.ndarray
        Оценка DoA. Представляет собой одномерный массив размера (K,1).
    R: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    """
    if seed is None: 
        seed = 100
    
    if iter == 0:
        theta = theta_guess
    elif iter > 0 and iter < 8:
        sigma = 0.05  
        bias = np.random.RandomState(seed).normal(0, sigma, size=K)
        theta = theta_guess + bias
        theta = np.clip(theta, -np.pi/2, np.pi/2)
    elif iter >= 8 and iter < 16:
        sigma = 0.2
        bias = np.random.RandomState(seed+30).normal(0, sigma, size=K)
        theta = theta_guess + bias
        theta = np.clip(theta, -np.pi/2, np.pi/2)
    else:
        sigma = 0.35
        bias = np.random.RandomState(seed+108).normal(0, sigma, size=K)
        theta = theta_guess + bias
        theta = np.clip(theta, -np.pi/2, np.pi/2)



    A = dss.A_ULA(L, theta)
    the_norm = np.linalg.norm(A, axis=0)
    A1 = A / the_norm
    pA = np.linalg.pinv(A1)
    res = R - Q
    P_normed = np.diag(pA @ res @ pA.conj().T).copy()
    for i in range(P_normed.shape[0]):
        P_normed[i] = max(P_normed[i], eps)
    P = np.diag(P_normed / the_norm)
    W = P - P @ A.conj().T @ np.linalg.inv(R) @ A @ P
    while True:
        if sensors.is_pd(W):
            break
        else:
            P = 0.5 * P
            W = P - P @ A.conj().T @ np.linalg.inv(R) @ A @ P

    print(f"theta={theta},P={P}")
    return theta, P


def init_est_kn2(K: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 L: int| None = None,
                 eps: float = 1e-3,
                 seed: int|None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Создает первоначальную оценку DoA и ковариационной матрицы 
    исходных сигналов. Улучшенная версия функции new_init_est, 
    связывает начальные оценки мощности источников с их угловыми координатами.

    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    Q: np.ndarray
        Ковариация шума.
    R: np.ndarray
        Оценка ковариации наблюдений.
    L: int|None
        Количество сенсоров в антенной решетке.
    eps: float
        Минимальное значение мощности источника.
    seed: int|None
        Randomstate для генерации данных.

    Returns
    ---------------------------------------------------------------------------
    theta: np.ndarray
        Оценка DoA. Представляет собой одномерный массив размера (K,1).
    R: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    """
    if seed is None: 
        seed = 100
        
    start = np.random.RandomState(seed).uniform(-np.pi/2, np.pi/2)
    theta = np.array([(start + i * np.pi / K + np.pi / 2) % np.pi - np.pi/2 for i in range(K)])
    theta = np.sort(theta)
    A = dss.A_ULA(L, theta)
    the_norm = np.linalg.norm(A, axis=0)
    A1 = A / the_norm
    pA = np.linalg.pinv(A1)
    res = R - Q
    P_normed = np.diag(pA @ res @ pA.conj().T).copy()
    for i in range(P_normed.shape[0]):
        P_normed[i] = max(P_normed[i], eps)
    P = np.diag(P_normed / the_norm)
    W = P - P @ A.conj().T @ np.linalg.inv(R) @ A @ P
    while True:
        if sensors.is_pd(W):
            break
        else:
            P = 0.5 * P
            W = P - P @ A.conj().T @ np.linalg.inv(R) @ A @ P

    print(f"theta={theta},P={P}")
    return theta, P