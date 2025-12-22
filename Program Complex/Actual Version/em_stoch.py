import numpy as np

import sensors
import optim_doa
import diff_sensor_structures as dss


def init_est(K: int,
             seed: int|None = None,
             type_of_theta_init: str = "circular") -> tuple[np.ndarray,
                                                            np.ndarray]:
    """
    Создает первоначальную оценку DoA и ковариационной матрицы 
    исходных сигналов.

    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    seed: int
        Randomstate для генерации данных.
    type_of_theta_init: str
        Способ инициализации начальной оценки DoA. Либо случайный вектор
        из многомерного равномерного распределения ("unstructured"),
        либо выбирается первый угол случайным образом, а затем относительно
        него вычисляется арифметическая прогрессия по модулю 2pi, число
        членов прогрессии равно K.

    Returns
    ---------------------------------------------------------------------------
    theta: np.ndarray
        Оценка DoA. Представляет собой одномерный массив размера (K,1).
    R: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    """
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/K)%(2 * np.pi) 
                          for i in range(K)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, K) 
    P_diag = np.random.RandomState(seed).uniform(0.2, 5, K)
    return np.sort(theta), np.diag(P_diag)


def Cov_signals(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Реализует вычисление ковариационной матрицы сигналов.

    Parameters
    ---------------------------------------------------------------------------
    mu: np.ndarray
        Массив, составленный из векторов УМО исходного сигнала, 
        в зависимости от наблюдений. 
        Число столбцов соответствует числу наблюдений.
    sigma: np.ndarray
        Условная ковариация исходных сигналов с учетом наблюдений.

    Returns
    ---------------------------------------------------------------------------
    res: np.ndarray
        Новая оценка ковариационной матрицы исходных сигналов.
    """
    G = mu.shape[1]
    res = (1/G) * mu @ mu.conj().T + sigma
    # Оставляем только диагональные элементы
    res = res * np.eye(res.shape[0], res.shape[1], dtype=np.complex128)
    return res.real


def incomplete_lkhd(X: np.ndarray,
                    theta: np.ndarray, 
                    P: np.ndarray, 
                    Q: np.ndarray) -> np.float64:
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений 
    и текущей оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    theta: np.ndarray
        Одномерный массив размера (K,1). Соответствует оценке DoA.
    P: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    Q: np.ndarray
        Ковариационная матрица шума.

    Returns
    ---------------------------------------------------------------------------
    res: np.float64
        Значение неполного правдоподобия.
    """
    A = dss.A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q
    R = 0.5* (R + R.conj().T) + 1e-6 * np.eye(R.shape[0])
    #print(f"is_spd(R)={sensors.is_spd(R)}")
    #print(f"is_spd(P)={sensors.is_spd(P)}")
    #print(f"is_spd(Q)={sensors.is_spd(Q)}")
    #print(f"Positive P? Ans is {np.all(np.diag(P) >= 0)}")
    inv_R = np.linalg.inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            R_o = R[np.ix_(O_i, O_i)]
            R_o = R_o + 1e-6 * np.eye(R_o.shape[0])
            res += (- np.log(np.linalg.det(R_o)) - (X[i, O_i].T).conj().T @ 
                      np.linalg.inv(R_o) @ (X[i, O_i].T))
        else:
            res += (- np.log(np.linalg.det(R)) - 
                    (X[i].T).conj().T @ inv_R @ (X[i].T))
    return res.real


def EM(angles: np.ndarray,
        P: np.ndarray,
        X: np.ndarray,
        Q: np.ndarray,
        max_iter: int = 50,
        rtol: float = 1e-3) -> tuple[np.ndarray,
                                     np.ndarray,
                                     np.float64]:
    """
    Запускает ЕМ-алгоритм для выбранной начальной оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    angles: np.ndarray
        Начальная оценка вектора углов, которые соответствуют DOA.
    P: np.ndarray
        Начальная оценка ковариационной матрицы исходных сигналов.
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    Q: np.ndarray
        Ковариационная матрица шума.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.

    Returns
    ---------------------------------------------------------------------------
    angles: np.ndarray
        Новая оценка DoA.
    P: np.ndarray
        Новая оценка ковариации исходных сигналов.
    lkhd: np.float64
        Новая оценка неполного правдоподобия.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    
    L = Q.shape[0]
    G = X.shape[0]

    print(f'Initial angles = {angles}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]

    R = sensors.robust_complex_cov(X[observed_rows, ])
    if np.isnan(R).any():
        R = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')

    Mu_Xm_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    Mu_S_cond = np.zeros((L, G), dtype=np.complex128)
    K_S_cond = np.zeros(P.shape, dtype=np.complex128)
    X_modified = X.copy()

    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = dss.A_ULA(L, angles)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]

                # Вычисляем блоки ковариации принятых сигналов (наблюдений)
                R_OO = R[np.ix_(O_i, O_i)]
                R_OO = R_OO + 1e-6 * np.eye(R_OO.shape[0])
                R_MO = R[np.ix_(M_i, O_i)]
                R_MM = R[np.ix_(M_i, M_i)]

                # Оцениваем параметры апостериорного распределения 
                # ненаблюдаемых данных и пропущенные значения
                Mu_Xm_cond[i] = R_MO @ np.linalg.inv(R_OO) @ X_modified[i, O_i]
                X_modified[i, M_i] = Mu_Xm_cond[i]
                K_Xm_cond_accum[np.ix_(M_i, M_i)] += (R_MM - R_MO @ 
                                                      np.linalg.inv(R_OO) @ 
                                                      R_MO.conj().T)

        
        # Вычисляем блоки совместной ковариации исходных и принятых сигналов
        K_XX = sensors.robust_complex_cov(X_modified) + K_Xm_cond_accum / G
        K_XX = 0.5 * (K_XX + K_XX.conj().T) + 1e-6 * np.eye(Q.shape[0])
        K_SS = P
        K_XS = A @ P
        K_SX = K_XS.conj().T

        # Е-шаг
        Mu_S_cond = K_SX @ np.linalg.inv(K_XX) @ X_modified.T
        K_S_cond = K_SS - K_SX @ np.linalg.inv(K_XX) @ K_XS
        
        Mu_XS = K_XX @ np.linalg.inv(R) @ A @ P

        Mu_SS = Cov_signals(Mu_S_cond, K_S_cond)


        # М-шаг
        R = sensors.robust_complex_cov(X_modified) + K_Xm_cond_accum / G
        new_angles = optim_doa.CM_step_angles(X_modified.T, angles, 
                                            Mu_S_cond, Q_inv_sqrt)
        new_angles = sensors.angle_correcter(new_angles)
        idx = np.argsort(new_angles)
        new_angles[:] = new_angles[idx]
        new_P = Mu_SS
        new_P[:] = new_P[np.ix_(idx, idx)]
        if (np.linalg.norm(angles - new_angles) < rtol 
            and np.linalg.norm(P - new_P, ord = 2) < rtol):
            break
        angles, P = new_angles, new_P
        #print(f'sorted? angles = {angles}')
        lkhd = incomplete_lkhd(X, angles, P, Q)
        print(f'likelihood is {lkhd} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return angles, P, lkhd


def multi_start_EM(X: np.ndarray,
                    K: int,
                    Q: np.ndarray,
                    num_of_starts: int = 10,
                    max_iter: int = 20,
                    rtol: float = 1e-6) -> tuple[np.ndarray,
                                                 np.ndarray,
                                                 np.float64]:
    """
    Реализует мультистарт для ЕМ-алгоритма.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    K: int
        Число источников.
    Q: np.ndarray
        Ковариационная матрица шума.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.

    Returns
    ---------------------------------------------------------------------------
    best_angles: np.ndarray
        Оценка DoA.
    best_P: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    best_lhd: np.float64
        Оценка неполного правдоподобия.
    """
    best_lhd, best_angles, best_P, best_start = -np.inf, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        angles, P = init_est(K, seed=i*100)
        est_angles, est_P, est_lhd = EM(angles, P, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_start = est_lhd, i
            best_P, best_angles = est_P, est_angles
    print(f"best_start={best_start}")
    return best_angles, best_P, best_lhd




