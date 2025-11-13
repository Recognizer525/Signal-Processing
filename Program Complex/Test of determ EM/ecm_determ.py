import numpy as np
from scipy.linalg import sqrtm

import sensors
import optim_doa

def init_est(X: np.ndarray, 
             M: int, 
             seed: int = None, 
             type_of_theta_init="circular"):
    """
    Создает первоначальную оценку DoA и детерминированных сигналов.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    M: int
        Число источников.
    seed: int - randomstate для генерации данных.

    Возвращает:
    ---------------------------------------------------------------------------
    theta: np.ndarray
        Одномерный массив размера (M,1). Соответствует оценке DoA.
    S: np.ndarray - оценка детерминированных исходных сигналов.
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


def init_Q(X: np.ndarray) -> np.ndarray:
    """
    Вычисляет начальную диагональную оценку ковариационной матрицы шума.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).

    Возвращает:
    ---------------------------------------------------------------------------
    Q: np.ndarray
        Первоначальная оценка ковариационной матрицы шума.
    """
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    Q = sensors.robust_complex_cov(X[observed_rows, ])
    if np.isnan(Q).any():
        #To rewrite in order to consider complex nature of values!
        Q = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of Q')
    return Q


def CM_step_S(X: np.ndarray, 
              A: np.ndarray, 
              Q: np.ndarray) -> np.ndarray:
    """
    Осуществляет условную максимизацию по ковариационной матрице шума.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    A: np.ndarray
        Двумерный массив, соответствующий матрице векторов направленности.
    Q: np.ndarray
        Двумерный массив, соответствующий ковариационной матрице шума.
    """
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return (np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X).T


def CM_step_Q(X: np.ndarray, 
              A: np.ndarray, 
              S: np.ndarray, 
              K: np.ndarray, 
              epsilon = 1e-6):
    """
    Осуществляет условную максимизацию по ковариационной матрице шума.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    A: np.ndarray
        Двумерный массив, соответствующий матрице векторов направленности.
    S: np.ndarray
        Двумерный массив, соответствующий оценке 
        последовательности исходных сигналов.
    K: np.ndarray
        Двумерный массив, сформированный путем сложения условных ковариаций, 
        полученных на Е-шаге.
    epsion: float
        Коэффициент регуляризации, используется для того, 
        чтобы гарантировать обратимость матрицы.
    """
    R = X.T - A @ S.T 
    R2 =  sensors.robust_complex_cov(R.T) + K/X.shape[0] + epsilon  * np.eye(X.shape[1], dtype=np.complex128)
    return R2


def incomplete_lkhd(X: np.ndarray, 
                    theta: np.ndarray, 
                    S: np.ndarray, 
                    Q: np.ndarray, 
                    inv_Q: np.ndarray):
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений 
    и текущей оценки параметров.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям
        (с учетом оценок пропущенных значений).
    theta: np.ndarray
        Одномерный массив размера (M,1). Соответствует оценке DoA.
    S: np.ndarray
        Двумерный массив, соответствующий оценке 
        последовательности исходных сигналов.
    Q: np.ndarray
        Двумерный массив, соответствующий ковариационной матрице шума.
    inv_Q: np.ndarray
        Матрица, обратная к Q.

    Возвращает:
    ---------------------------------------------------------------------------
    res - значение неполного правдоподобия
    """
    A = sensors.A_ULA(X.shape[1], theta)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            A_o, Q_o = A[O_i, :], Q[np.ix_(O_i, O_i)]
            res += - np.log(np.linalg.det(Q_o)) - (X[i, O_i].T - A_o @ S[i].T).conj().T @ np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T)
        else:
            res += - np.log(np.linalg.det(Q)) - (X[i].T - A @ S[i].T).conj().T @ inv_Q @ (X[i].T - A @ S[i].T)
    return res.real


def ECM_kn(theta: np.ndarray, 
           S: np.ndarray, 
           X: np.ndarray, 
           Q: np.ndarray, 
           max_iter: int=50, rtol: float=1e-6):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    ---------------------------------------------------------------------------
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    Q: np.ndarray
        Ковариационная матрица шума.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    L = Q.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    Mu_cond = {}
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = sensors.A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                K_OO = Q[np.ix_(O_i, O_i)]
                K_MO = Q[np.ix_(M_i, O_i)]
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(K_OO) @ (X_modified[i, O_i] - A_o @ S[i])
                X_modified[i, M_i] = Mu_cond[i]
        print(f"Sum of mv is {np.sum(np.isnan(X_modified))}")
        # Шаги условной максимизации
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
        print(f'incomplete likelihood is {lkhd} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, S, lkhd


def multi_start_ECM_kn(X: np.ndarray, 
                       M: int, 
                       Q: np.ndarray, 
                       num_of_starts: int = 20, 
                       max_iter: int = 20, 
                       rtol: float = 1e-6):
    """
    Реализует мультистарт для ЕCМ-алгоритма.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям
        (с учетом оценок пропущенных значений).
    M: int
        Число источников.
    Q: np.ndarray
        Двумерный массив, соответствующий ковариационной матрице шума.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.
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



def ECM_un(theta: np.ndarray,
           S: np.ndarray, 
           X: np.ndarray,
           Q: np.ndarray,
           max_iter: int=20,
           rtol: float=1e-5):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    ---------------------------------------------------------------------------
    theta: np.ndarray
      Одномерный массив размера (M,1). Соответствует оценке DoA.
    S: np.ndarray
        Первоначальная оценка последовательности исходных сигналов,
        представленная двумерным массивом.
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям
        (с учетом оценок пропущенных значений).
    Q: np.ndarray
        Первоначальная оценка ковариационной матрицы шума.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.
    """
    Q_inv = np.linalg.inv(Q)
    #Q_inv_sqrt = np.sqrt(Q_inv)
    Q_inv_sqrt = sqrtm(Q_inv)
    L = Q.shape[0]
    G = X.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    Mu_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    X_modified = X.copy()
    ECM_Iteration = 0
    while ECM_Iteration < max_iter:
        A = sensors.A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                Q_o, Q_m = Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
                K_MO = Q[np.ix_(M_i, O_i)]
                K_OM = K_MO.conj().T
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                K_Xm_cond_accum[np.ix_(M_i, M_i)] = Q_m - K_MO @ np.linalg.inv(Q_o) @ K_OM
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        new_theta = optim_doa.CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        A = sensors.A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        new_Q = CM_step_Q(X_modified, A, new_S, K_Xm_cond_accum)
        lkhd = incomplete_lkhd(X_modified, new_theta, new_S, new_Q, np.linalg.inv(Q))
        theta, S, Q = new_theta, new_S, new_Q
        print(f'incomplete likelihood is {lkhd} on iteration {ECM_Iteration}')
        ECM_Iteration += 1
    return theta, S, Q, lkhd


def multi_start_ECM_un(X: np.ndarray,
                       M: int,
                       num_of_starts: int = 30,
                       max_iter: int = 20,
                       rtol: float = 1e-6):
    """
    Реализует мультистарт для ЕCМ-алгоритма.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям
        (с учетом оценок пропущенных значений).
    M: int
        Число источников.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.

    Возвращает:
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
    best_S: np.ndarray
    best_Q: np.ndarray
    best_lhd:
    """
    best_lhd, best_theta, best_S, best_Q, best_start = -np.inf, None, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S = init_est(X, M, seed=i*100)
        Q = init_Q(X)
        est_theta, est_S, est_Q, est_lhd = ECM_un(theta, S, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_theta, best_S, best_Q, best_lhd, best_start = est_theta, est_S, est_Q, est_lhd, i
    best_theta = sensors.angle_correcter(best_theta)
    print(f'best_start={best_start}')
    return best_theta, best_S, best_Q, best_lhd