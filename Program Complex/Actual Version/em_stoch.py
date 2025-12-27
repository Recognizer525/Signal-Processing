import numpy as np

import sensors
import optim_doa
import diff_sensor_structures as dss


def old_init_est(K: int,
                 seed: int|None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Создает первоначальную оценку DoA и ковариационной матрицы 
    исходных сигналов.

    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    seed: int
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
    nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
    theta = np.array([(nu + i * 2 * np.pi/K)%(2 * np.pi) 
                      for i in range(K)]) - np.pi
    theta = np.sort(theta)

    P_diag = np.random.RandomState(seed).uniform(0.2, 5, K)
    print(f"theta={theta},P={P_diag}")
    return np.sort(theta), np.diag(P_diag)


def new_init_est(K: int,
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
    L: int
        Количество сенсоров в антенной решетке.
    eps: float
        Минимальное значение мощности источника.
    seed: int
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
    A = A / the_norm
    pA = np.linalg.pinv(A)
    res = R - Q
    P_normed = np.diag(pA @ res @ pA.conj().T).copy()
    for i in range(P_normed.shape[0]):
        P_normed[i] = max(P_normed[i], eps)
    P = P_normed / the_norm
    print(f"theta={theta},P={P}")
    return np.sort(theta), np.diag(P)


def Cov_signals(mu: np.ndarray, 
                sigma: np.ndarray) -> np.ndarray:
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
    #print(f"sigma={sigma}")
    #print(f"M[X]M[X]={(1/G) * mu @ mu.conj().T}")
    res = (1/G) * mu @ mu.conj().T + sigma
    #print(f'Cov_signals ={res}')
    # Оставляем только диагональные элементы
    res_masked = res.copy()
    res_masked[~np.eye(res.shape[0], dtype=bool)] = 0
    print(f'res={res_masked}')
    return res_masked.real


def if_converged(angles:np.ndarray, 
                 new_angles: np.ndarray, 
                 P: np.ndarray, 
                 new_P: np.ndarray, 
                 rtol: float) -> bool:
    """
    Проверка достижения сходимости алгоритма на текущей итерации, 
    сравниваются отсортированные вектора/матрицы параметров 
    на текущей и предшествующей итерации.
    """
    idx1 = np.argsort(angles)
    idx2 = np.argsort(new_angles)
    angles[:] = angles[idx1]
    new_angles[:] = new_angles[idx2]
    P = P[np.ix_(idx1, idx1)]
    new_P[:] = new_P[np.ix_(idx2, idx2)]
    if (np.linalg.norm(angles - new_angles) < rtol 
        and np.linalg.norm(P - new_P, ord = 2) < rtol):
        return True
    return False


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
    R = 0.5 * (R + R.conj().T) + 1e-6 * np.eye(R.shape[0])
    #print(f"is_spd(R)={sensors.is_spd(R)}")
    #print(f"is_spd(P)={sensors.is_spd(P)}")
    #print(f"is_spd(Q)={sensors.is_spd(Q)}")
    #print(f"Positive P? Ans is {np.all(np.diag(P) >= 0)}")
    inv_R = np.linalg.inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1]+1)
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

    #print(f'Initial angles = {angles}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1]+1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1

    R = sensors.initial_Cov(X)

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
        K_XX = sensors.complex_cov(X_modified) + K_Xm_cond_accum / G

        if not sensors.is_psd(K_XX):
            print(f"K_XX is unusual")
        #print(f"Is PSD K_XX {sensors.is_psd(K_XX)}")
        K_XX = 0.5 * (K_XX + K_XX.conj().T) + 1e-6 * np.eye(Q.shape[0])
        K_SS = P
        K_XS = A @ P
        K_SX = K_XS.conj().T

        # Е-шаг
        Mu_S_cond = K_SX @ np.linalg.inv(K_XX) @ X_modified.T
        K_S_cond = K_SS - K_SX @ np.linalg.inv(K_XX) @ K_XS
        
        Sigma_XS = K_XX @ np.linalg.inv(R) @ A @ P
        Sigma_SS = Cov_signals(Mu_S_cond, K_S_cond)

        if not sensors.is_psd(Sigma_SS):
            print(f"Sigma_SS is unusual")

        # М-шаг
        new_angles = optim_doa.find_angles(Sigma_XS, angles, 
                                            Sigma_SS, Q_inv_sqrt)
        print(f"new_angles={new_angles}")
        new_angles = sensors.angle_correcter(new_angles)
        idx = np.argsort(new_angles)
        new_angles[:] = new_angles[idx]
        new_P = Sigma_SS
        new_P[:] = new_P[np.ix_(idx, idx)]
        lkhd = incomplete_lkhd(X, new_angles, new_P, Q)
        if if_converged(angles, new_angles, P, new_P, rtol):
            break
        angles, P = new_angles, new_P
        A = dss.A_ULA(L, angles)
        R = A @ P @ A.conj().T + Q
        print(f'likelihood is {lkhd} on iteration {EM_Iteration}')
        if lkhd > 0:
            print(f"Parameters of interest are angles={angles}, P={P}")
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
    L = X.shape[1]
    R = sensors.initial_Cov(X)
    for i in range(num_of_starts):
        print(f'{i}-th start')
        angles, P = new_init_est(K, Q, R, L, seed=i*100)
        est_angles, est_P, est_lhd = EM(angles, P, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_start = est_lhd, i
            best_P, best_angles = est_P, est_angles
    print(f"best_start={best_start}")
    return best_angles, best_P, best_lhd




