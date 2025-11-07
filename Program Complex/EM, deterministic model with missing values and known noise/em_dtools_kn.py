import numpy as np
import scipy
from functools import partial
from scipy.optimize import minimize

dist_ratio = 0.5

def MCAR(X: np.ndarray, mis_cols: object, size_mv: object , rs: int = 42) -> np.ndarray:
    '''
    Функция реализует создание случайных пропусков, пропуски произвольной переменной не зависят от наблюдаемых или пропущенных значений.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    if type(size_mv)==int:
        size_mv=[size_mv]
    assert len(mis_cols)==len(size_mv)
    X1 = X.copy()
    for i in range(len(mis_cols)):
        h = np.array([1]*size_mv[i]+[0]*(len(X)-size_mv[i]))
        np.random.RandomState(rs+i).shuffle(h)
        X1[:,mis_cols[i]][np.where(h==1)] = np.nan
    return X1


def gds(M, G, A = None, f = None, phi = None, seed: int = None):
    """
    Генерирует детерминированные сигналы, представляющие из себя комплексные нормальные вектора (circularly-symmetric case).
    M - размер вектора сигнала;
    G - количество векторов;
    A - вектор амплитуд;
    f - вектор частот;
    phi - вектор фаз;
    """ 
    if seed is None:
        seed = 10
    # G - размер выборки, M - число источников
    if A is None:
        A = np.random.RandomState(seed + 40).uniform(0.5, 1.5, M)         
    if f is None:
        f = np.random.RandomState(seed + 10).uniform(0.01, 0.1, M)        
    if phi is None:
        phi = np.random.RandomState(seed + 1).uniform(0, 2*np.pi, M)
    
    g = np.arange(G)
    signals = np.zeros((M, G), dtype=complex)
    for m in range(M):
        signals[m] = A[m] * np.exp(1j * (2 * np.pi * f[m] * g + phi[m]))
    signals = signals.T
    return signals


def gss(size: int, number: int, Gamma: np.ndarray, seed: int = None):
    """
    Генерирует стохастические сигналы, представляющие из себя комплексные нормальные вектора (circularly-symmetric case).
    size - размер вектора;
    number - количество векторов;
    Gamma - ковариационная матрица.
    """ 
    if seed is None:
        seed = 70
    n = 2 * size # Размер ковариационной матрицы совместного распределения
    C = np.zeros((n,n), dtype=np.float64)
    C[:size,:size] = Gamma.real
    C[size:,size:] = Gamma.real
    C[:size,size:] = -Gamma.imag
    C[size:,:size] = Gamma.imag
    mu = np.zeros(n)
    B = np.random.RandomState(seed).multivariate_normal(mu, 0.5*C, number)
    signals = B[:,:size] + 1j * B[:, size:]
    return signals


def complex_cov(X: np.ndarray):
    """
    Метод предназначен для формирования оценки матрицы пространственной ковариации.
    X - коллекция полученных сигналов.
    """
    return (np.einsum('ni,nj->ij', X, X.conj()) / X.shape[0])


def angle_correcter(theta: np.ndarray) -> np.ndarray:
    """
    Приводит углы к диапазону [-pi/2, pi/2], сохраняя то же значение синуса.
    """
    # Приведение к диапазону [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    # Отражение значений, выходящих за пределы [-pi/2, pi/2]
    mask = theta > np.pi/2
    theta[mask] = np.pi - theta[mask]
    mask = theta < -np.pi/2
    theta[mask] = -np.pi - theta[mask]
    return theta


def A_ULA(L, theta):
    """
    Создает матрицу управляющих векторов для массива сенсоров типа ULA
    """
    return np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta))


def initializer(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/M)%(2 * np.pi) for i in range(M)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M) 
    S = gds(M, len(X), seed=seed+20)
    return theta, S
    
    
def cost_theta(theta, X, S, weights):
    """
    theta - вектор углов прибытия;
    X - набор принятых сигналов, с учетом заполненных пропусков;
    S - набор отправленных сигналов;
    weights - вектор, полученный следующим образом:  диагональная ковариационная матрица шума обращается и возводится в степень 1/2, 
    а затем диагональ этой матрицы приводится к вектору
    """
    A = A_ULA(X.shape[0], theta)
    res = X - A @ S
    sum_row_wise = np.sum(res**2, axis=1)
    cost = np.sum((weights**2) * sum_row_wise)  
    return cost.real


def CM_step_theta(X, theta_guess, S, Q_inv_sqrt):
    res = minimize(
            lambda th: cost_theta(th, X, S, Q_inv_sqrt),
            theta_guess,
            method='L-BFGS-B',
            bounds=[(-np.pi/2, np.pi/2)] * len(theta_guess)
        )
    return res.x    


def CM_step_S(X, A, Q):
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return (np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X).T


def likelihood(X, theta, S, Q, inv_Q):
    """
    X - выборка, состоящая из принятых сигналов, с учетом оценок пропущенных значений, 
    каждый столбец соответствует одному наблюдению;
    theta - оценка вектора углов;
    S - оценка сигналов, каждый столбец соответствует одному сигналу;
    Q - матрица ковариации шума;
    inv_Q - матрица, обратная к Q.
    """
    A = A_ULA(X.shape[0], theta)
    M = X - A @ S
    return (- X.shape[1] * np.linalg.det(Q) - np.trace(M.conj().T @ inv_Q @ M)).real


def incomplete_lkhd(X, theta, S, Q, inv_Q):
    A = A_ULA(X.shape[1], theta)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            A_o, Q_o = A[np.ix_(O_i, O_i)], Q[np.ix_(O_i, O_i)]
            res += - np.linalg.det(Q_o) - (X[i, O_i].T - A_o @ S[i].T).conj().T @ np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T)
        else:
            res += - np.linalg.det(Q) - (X[i].T - A @ S[i].T).conj().T @ inv_Q @ (X[i].T - A @ S[i].T)
    return res


def EM(theta: np.ndarray, S: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    Q_vec = np.diagonal(Q)
    Q_inv_sqrt = np.sqrt(1/Q_vec)
    L = Q.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = complex_cov(X[observed_rows, ])
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_cond = {}
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[np.ix_(O_i, O_i)], A[np.ix_(M_i, M_i)]
                Q_o = Q[np.ix_(O_i, O_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                K_OM = K_MO.T
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        K = complex_cov(X_modified)
        new_theta = CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        A = A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        print(f'diff of S is {np.sum((new_S-S)**2)} on iteration {EM_Iteration}')
        theta, S = new_theta, new_S
        lkhd = incomplete_lkhd(X_modified, theta, S, Q, np.linalg.inv(Q))
        print(f'incomplete likelihood is {lkhd.real} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, lkhd


def multi_start_EM(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 30, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M - число источников;
    Q - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_lhd, best_theta = -np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S = initializer(X, M, seed=i * 100)
        #print(f"On multistart shape of S is {S.shape}")
        est_theta, est_lhd = EM(theta, S, X, Q, max_iter, eps)
        if est_lhd > best_lhd:
            best_lhd, best_theta = est_lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_lhd

