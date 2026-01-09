import numpy as np

import sensors
import optim_doa as od
import diff_sensor_structures as dss
import debug_funcs as df
import angle_finding

def init_est(K: int,
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
    T = mu.shape[1]
    #print(f"sigma={sigma}")
    #print(f"M[X]M[X]={(1/G) * mu @ mu.conj().T}")
    res = (1/T) * mu @ mu.conj().T + sigma
    #print(f'Cov_signals ={res}')
    # Оставляем только диагональные элементы
    res_masked = res.copy()
    res_masked[~np.eye(res.shape[0], dtype=bool)] = 0
    print(f'res={res_masked}')
    return res_masked.real


def if_params_converged(angles:np.ndarray, 
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


def if_lkhd_converged(old_lkhd: float,
                      lkhd: float,
                      rtol: float = 1e-6) -> bool:
    """
    Проверяет степень близости старого и нового значения 
    неполного правдоподобия.
    """
    if lkhd == 0:
        return np.abs(lkhd - old_lkhd) < rtol
    return np.abs(lkhd-old_lkhd)/np.abs(lkhd) < rtol



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
    
    T = X.shape[0]
    L = Q.shape[0]
    K = P.shape[0]
    
    mask = ~np.isnan(X).any(axis=1)

    #print(f'Initial angles = {angles}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1]+1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1

    R = sensors.initial_Cov(X)
    A = dss.A_ULA(L, angles)

    print(f"Initial diagonal of diff is {np.diag(R-Q-A @ P @ A.conj().T)}")

    K_Xm_cond = np.zeros((T, L, L), dtype=np.complex128)
    K_S_cond = np.zeros((T, K, K), dtype=np.complex128)

    Gap_based_Cov = np.zeros((T, K, K), dtype=np.complex128)
    Gap_based_Cross_cov = np.zeros((T, L, K), dtype=np.complex128)
    E_X_cond = X.copy()

    EM_Iteration = 0
    while EM_Iteration < max_iter:
        R_inv_A_P = np.linalg.inv(R) @ A @ P
        R_inv_A_P_H = R_inv_A_P.conj().T
        Common_Cov_S = P - R_inv_A_P_H @ A @ P

        for i in range(T):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]

                # Вычисляем блоки ковариации наблюдений
                R_OO = R[np.ix_(O_i, O_i)]
                R_OO = R_OO + 1e-6 * np.eye(R_OO.shape[0])
                R_MO = R[np.ix_(M_i, O_i)]
                R_MM = R[np.ix_(M_i, M_i)]

                # Оцениваем параметры апостериорного распределения 
                # ненаблюдаемых данных и пропущенные значения
                E_X_cond[i, M_i] = R_MO @ np.linalg.inv(R_OO) @ E_X_cond[i, O_i]
                K_Xm_cond[np.ix_([i], M_i, M_i)] += (R_MM - R_MO @ 
                                                      np.linalg.inv(R_OO) @ 
                                                      R_MO.conj().T)


        E_X_E_X_H = np.einsum('li,lj -> lij', E_X_cond, E_X_cond.conj())
        Sigma_XX_arr = E_X_E_X_H + K_Xm_cond


        Gap_based_Cov[~mask] = R_inv_A_P_H @ Sigma_XX_arr[~mask] @ R_inv_A_P
        Gap_based_Cross_cov[~mask] = Sigma_XX_arr[~mask] @ R_inv_A_P

        Mu_S_cond = R_inv_A_P_H @ E_X_cond.T
        K_S_cond = Common_Cov_S + Gap_based_Cov

        E_X_E_S_H = np.einsum('li,lj -> lij', E_X_cond, Mu_S_cond.conj().T)
        E_S_E_S_H = np.einsum('li,lj -> lij', Mu_S_cond.T, Mu_S_cond.conj().T)
        
        #Sigma_XX = np.mean(Sigma_XX_arr, axis=0)
        Sigma_XS = np.mean(E_X_E_S_H + Gap_based_Cross_cov, axis=0)
        Sigma_SS = np.mean(E_S_E_S_H + K_S_cond, axis=0)

        df.is_valid_result(E_X_E_X_H,'E_X_E_X_H', expected_shape=(T, L, L))
        df.is_valid_result(Sigma_XX_arr,'Sigma_XX_arr', expected_shape=(T, L, L), check_psd=True)
        df.is_valid_result(Mu_S_cond,'Mu_S_cond', expected_shape=(K,T))
        df.is_valid_result(K_S_cond,'K_S_cond', expected_shape=(T,K,K), check_psd=True)
        df.is_valid_result(E_X_E_S_H,'E_X_E_S_H', expected_shape=(T,L,K))
        df.is_valid_result(E_S_E_S_H,'E_S_E_S_H', expected_shape=(T,K,K), check_psd=True)
        df.is_valid_result(Sigma_XS,'Sigma_XS', expected_shape=(L, K))
        df.is_valid_result(Sigma_SS,'Sigma_SS', expected_shape=(K, K), check_psd=True)

        # М-шаг
        new_angles = angle_finding.find_angles(Sigma_XS, angles, 
                                            Sigma_SS, Q_inv_sqrt)
        print(f"new_angles={new_angles}")
        idx = np.argsort(new_angles)
        new_angles[:] = new_angles[idx]
        new_P = Sigma_SS
        new_P[:] = new_P[np.ix_(idx, idx)]
        lkhd = incomplete_lkhd(X, new_angles, new_P, Q)
        if if_params_converged(angles, new_angles, P, new_P, rtol):
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
        angles, P = init_est(K, Q, R, L, seed=i*100)
        est_angles, est_P, est_lhd = EM(angles, P, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_start = est_lhd, i
            best_P, best_angles = est_P, est_angles
    print(f"best_start={best_start}")
    return best_angles, best_P, best_lhd




