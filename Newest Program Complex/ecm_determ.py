import numpy as np
import sensors
import optim_doa
from functools import partial

dist_ratio = 0.5


def init_est(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    """
    Создает первоначальную оценку DoA и детерминированных сигналов.

    Параметры:
    X: np.ndarray - наблюдения с пропусками.
    M: int - число источников.
    seed: int - randomstate для генерации данных.

    Возвращает:
    theta: np.ndarray - оценка DoA.
    S: np.ndarray - оценка детерминированных сигналов.
    """
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/M)%(2 * np.pi) for i in range(M)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M) 
    S = sensors.gds(M, len(X), seed=seed+20)
    return theta, S


def init_Q(X: np.ndarray, theta: np.ndarray, signals: np.ndarray):
    """
    Вычисляет начальную диагональную оценку ковариации шума.
    """
    L = len(X[0])
    A = sensors.A_ULA(L, theta)
    # Остатки шума
    r = X.T - A @ signals.T 
    # Для каждого канала считаем дисперсию по времени
    Sigma_N_diag = np.nanvar(r, axis=1, ddof=0)
    epsilon = 1e-6
    Sigma_N_diag = Sigma_N_diag + epsilon
    return np.diag(Sigma_N_diag)


def CM_step_S(X: np.ndarray, A: np.ndarray, Q: np.ndarray):
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return (np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X).T


def CM_step_Q(X: np.ndarray, A: np.ndarray, S: np.ndarray):
    R = X.T - A @ S.T 
    Q = np.nanvar(R, axis=1, ddof=0)  
    epsilon = 1e-6
    Q = np.diag(Q + epsilon)
    #print(f"Q={Q}")
    #print(f'Q.shape={Q.shape}')
    return Q


def incomplete_lkhd(X: np.ndarray, theta: np.ndarray, S: np.ndarray, Q: np.ndarray, inv_Q: np.ndarray):
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений и текущей оценки параметров.

    Параметры:
    X - наблюдения (np.ndarray)
    theta - оценки углов прибытия (np.ndarray)
    S - оценки исходных сигналов (np.ndarray)
    Q - матрица ковариации шума, либо ее оценка (np.ndarray)
    inv_Q - обратная к Q матрица (np.ndarray)

    Возвращает:
    res - значение неполного правдоподобия
    """
    A = sensors.A_ULA(X.shape[1], theta)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    sign, Q_logdet = np.linalg.slogdet(Q)
    if sign <= 0:
        raise ValueError("Q is not positive definite (det <= 0)")
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            A_o, Q_o = A[O_i, :], Q[np.ix_(O_i, O_i)]
            sign, Q_o_logdet = np.linalg.slogdet(Q_o)
            if sign <= 0:
               raise ValueError("Q_o is not positive definite (det <= 0)") 
            res += - Q_o_logdet - (X[i, O_i].T - A_o @ S[i].T).conj().T @ np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T)
        else:
            res += - Q_logdet - (X[i].T - A @ S[i].T).conj().T @ inv_Q @ (X[i].T - A @ S[i].T)
    return res.real


def ECM_kn(theta: np.ndarray, S: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, rtol: float=1e-6):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    L = Q.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = sensors.robust_complex_cov(X[observed_rows, ])
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_cond = {}
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = sensors.A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                Q_o = Q[np.ix_(O_i, O_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                X_modified[i, M_i] = Mu_cond[i]
        print(f"Sum of mv is {np.sum(np.isnan(X_modified))}")
        # Шаги условной максимизации
        K = sensors.robust_complex_cov(X_modified)
        #print(f"K={K}")
        print('eigvals',np.linalg.eigvals(K))
        new_theta = optim_doa.CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        print(f"new_theta={new_theta}")
        A = sensors.A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        #print(f'diff of S is {np.sum((new_S-S)**2)} on iteration {EM_Iteration}')
        lkhd = incomplete_lkhd(X_modified, new_theta, new_S, Q, np.linalg.inv(Q))
        if np.linalg.norm(theta - new_theta) < rtol and np.linalg.norm(S - new_S, ord = 2) < rtol:
            break
        theta, S = new_theta, new_S
        print(f'incomplete likelihood is {lkhd.real} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, S, lkhd


def multi_start_ECM_kn(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, rtol: float = 1e-6):
    """
    Реализует мультистарт для ЕCМ-алгоритма.

    Параметры:
    X: np.ndarray - коллекция полученных сигналов;
    M: int - число источников;
    Q: np.ndarray - ковариация шума;
    num_of_starts: int - число запусков;
    max_iter: int - предельное число итерация;
    rtol: float - величина, используемая для проверки сходимости последних двух оценок параметров.
    """
    best_lhd, best_theta, best_S = -np.inf, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S = init_est(X, M, seed=i * 100)
        est_theta, est_S, est_lhd = ECM_kn(theta, S, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_theta, best_S = est_lhd, est_theta, est_S
    best_theta = sensors.angle_correcter(best_theta)
    return best_theta, best_S, best_lhd



def ECM_un(theta: np.ndarray, S: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=20, rtol: float=1e-5):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.
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
    K = sensors.robust_complex_cov(X[observed_rows, ])
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = sensors.A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                Q_o, Q_m = Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                K_OM = K_MO.conj().T
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                K_Xm_cond_accum[np.ix_(M_i, M_i)] = Q_m - K_MO @ np.linalg.inv(Q_o) @ K_OM
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        K = sensors.robust_complex_cov(X_modified)
        #print(f"K={K}")
        new_theta = optim_doa.CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        A = sensors.A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        #print(f'diff of S is {np.sum((new_S-S)**2)} on iteration {EM_Iteration}')
        new_Q = CM_step_Q(X_modified, A, new_S)
        #print(f"new_Q = {new_Q}")
        #print(f'diff of Q is {np.sum((new_Q-Q)**2)} on iteration {EM_Iteration}')
        lkhd = incomplete_lkhd(X_modified, new_theta, new_S, new_Q, np.linalg.inv(Q))
        if np.linalg.norm(theta - new_theta) < rtol and np.linalg.norm(S - new_S, ord = 2) < rtol and np.linalg.norm(Q - new_Q, ord = 2) < rtol:
            break
        theta, S, Q = new_theta, new_S, new_Q
        print(f'incomplete likelihood is {lkhd.real} on iteration {EM_Iteration}')
        EM_Iteration += 1
    return theta, S, Q, lkhd


def multi_start_ECM_un(X: np.ndarray, M: int, num_of_starts: int = 30, max_iter: int = 20, rtol: float = 1e-6):
    """
    Реализует мультистарт для ЕCМ-алгоритма.

    Параметры:
    X - коллекция полученных сигналов;
    M - число источников;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.

    Возвращает:
    best_theta: np.ndarray
    best_S: np.ndarray
    best_Q: np.ndarray
    best_lhd:
    """
    best_lhd, best_theta, best_S, best_Q, best_start = -np.inf, None, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S = init_est(X, M, seed=i*100)
        Q = init_Q(X, theta, S)
        est_theta, est_S, est_Q, est_lhd = ECM_un(theta, S, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_theta, best_S, best_Q, best_lhd, best_start = est_theta, est_S, est_Q, est_lhd, i
    best_theta = sensors.angle_correcter(best_theta)
    print(f'best_start={best_start}')
    return best_theta, best_S, best_Q, best_lhd