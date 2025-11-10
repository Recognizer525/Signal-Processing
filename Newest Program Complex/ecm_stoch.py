import numpy as np
import sensors
import optim_doa

dist_ratio = 0.5

def is_diagonal(A):
    """
    Проверяет свойство диагональности для матрицы A.
    """
    return np.all(A == np.diag(np.diagonal(A)))


def is_spd(A, tol=1e-3):
    """
    Проверяет, что матрица A симметрична и положительно определена.
    """
    # Проверим симметрию
    if not np.allclose(A, A.conj().T, atol=tol):
        return False
    # Проверим положительную определённость
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def init_est(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/M)%(2 * np.pi) for i in range(M)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M) 
    P_diag = np.random.RandomState(seed).uniform(0.2, 5, M)
    return theta, np.diag(P_diag)


def CM_step_P(mu: np.ndarray, sigma: np.ndarray):
    """
    Реализует шаг условной максимизации по ковариации исходных сигналов.

    Параметры:
    mu - массив, составленный из векторов УМО исходного сигнала, в зависимости от наблюдений. Число столбцов соответствует числу наблюдений.
    sigma - условная ковариация исходного сигнала с учетом наблюдения.

    Возвращает:
    res: np.ndarray - новая оценка ковариации исходных сигналов.
    """
    G = mu.shape[1]
    res = (1/G) * mu @ mu.conj().T + sigma
    # Оставляем только диагональные элементы
    res = res * np.eye(res.shape[0], res.shape[1], dtype=np.complex128)
    return res.real


def incomplete_lkhd(X: np.ndarray, theta: np.ndarray, P: np.ndarray, Q: np.ndarray):
    A = sensors.A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q
    R = 0.5* (R + R.conj().T) + 1e-6 * np.eye(R.shape[0])
    #print(f"is_spd(R)={is_spd(R)}")
    #print(f"is_spd(P)={is_spd(P)}")
    #print(f"is_spd(Q)={is_spd(Q)}")
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
            res += - np.log(np.linalg.det(R_o)) - (X[i, O_i].T).conj().T @ np.linalg.inv(R_o) @ (X[i, O_i].T)
        else:
            res += - np.log(np.linalg.det(R)) - (X[i].T).conj().T @ inv_R @ (X[i].T)
    return res.real


def ECM(theta: np.ndarray, P: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, rtol: float=1e-6):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    theta: np.ndarray
        Начальная оценка вектора углов, которые соответствуют DOA.
    P: np.ndarray
        Ковариация исходных сигналов.
    X: np.ndarray
        Коллекция полученных сигналов.
    Q: np.ndarray
        Ковариация шума.
    max_iter: int
        Предельное число итерация.
    rtol: float
        Величина, используемая для проверки сходимости последних оценок параметров.

    Возвращает:
    theta: np.ndarray
        Новая оценка DoA.
    P: np.ndarray
        Новая оценка ковариации исходных сигналов.
    lkhd: np.complex128
        Новая оценка неполного правдоподобия.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    
    L = Q.shape[0]
    G = X.shape[0]

    print(f'Initial theta = {theta}')

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
        A = sensors.A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                # Вычисляем блоки ковариации принятых сигналов (наблюдений)
                R_OO = R[np.ix_(O_i, O_i)]
                R_OO = R_OO + 1e-6 * np.eye(R_OO.shape[0])
                R_MO = R[np.ix_(M_i, O_i)]
                R_MM = R[np.ix_(M_i, M_i)]
                # Оцениваем параметры апостериорного распределения ненаблюдаемых данных и пропущенные значения
                Mu_Xm_cond[i] = R_MO @ np.linalg.inv(R_OO) @ X_modified[i, O_i]
                X_modified[i, M_i] = Mu_Xm_cond[i]
                K_Xm_cond_accum[np.ix_(M_i, M_i)] += R_MM - R_MO @ np.linalg.inv(R_OO) @ R_MO.conj().T
        # Вычисляем блоки совместной ковариации исходных и принятых сигналов
        K_XX = A @ P @ A.conj().T + Q
        K_XX = 0.5 * (K_XX + K_XX.conj().T) + 1e-6 * np.eye(Q.shape[0])
        K_SS = P
        K_XS = A @ P
        K_SX = K_XS.conj().T
        Mu_S_cond = K_SX @ np.linalg.inv(K_XX) @ X_modified.T
        K_S_cond = K_SS - K_SX @ np.linalg.inv(K_XX) @ K_XS
        # Шаги условной максимизации
        R = sensors.robust_complex_cov(X_modified) + K_Xm_cond_accum / G
        new_theta = optim_doa.CM_step_theta(X_modified.T, theta, Mu_S_cond, Q_inv_sqrt)
        #if EM_Iteration in [0, 1, 5, 11, 16, 21, 26]:
            #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        A = sensors.A_ULA(L, new_theta)
        new_P = CM_step_P(Mu_S_cond, K_S_cond)
        if np.linalg.norm(np.array(sorted(theta)) - np.array(sorted(new_theta))) < rtol and np.linalg.norm(P - new_P, ord = 2) < rtol:
            break
        #print(f'diff of P is {np.sum((new_P-P)**2)} on iteration {EM_Iteration}')
        theta, P = new_theta, new_P
        lkhd = incomplete_lkhd(X_modified, theta, P, Q)
        if EM_Iteration in set([0, 1, 5, 11, 16, 21, 26]):
            print(f'likelihood is {lkhd} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, P, lkhd


def multi_start_ECM(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 10, max_iter: int = 20, rtol: float = 1e-6):
    """
    Мультистарт для ЕCМ-алгоритма.

    Параметры:
    X: np.ndarray 
      Коллекция полученных сигналов.
    M: int
      Число источников.
    Q: np.ndarray
      Ковариация шума.
    num_of_starts: int
      Число запусков.
    max_iter: int
      Предельное число итераций.
    rtol: float
      Величина, используемая для проверки сходимости последних итераций.

    Возвращает:
    best_theta: np.ndarray
      Оценка DoA.
    best_P: np.ndarray
      Оценка ковариационной матрицы исходных сигналов.
    best_lhd: np.complex128
      Оценка неполного правдоподобия.
    """
    best_lhd, best_theta, best_P, best_start = -np.inf, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, P = init_est(X, M, seed=i * 100)
        est_theta, est_P, est_lhd = ECM(theta, P, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_P, best_theta, best_start = est_lhd, est_P, est_theta, i
    best_theta = sensors.angle_correcter(best_theta)
    print(f"best_start={best_start}")
    return best_theta, best_P, best_lhd




