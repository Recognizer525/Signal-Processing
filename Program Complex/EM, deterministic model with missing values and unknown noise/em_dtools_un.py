import numpy as np
import scipy
import math
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
        np.random.RandomState(rs).shuffle(h)
        X1[:,mis_cols[i]][np.where(h==1)] = np.nan
    return X1

def gds(G, M, A = None, f = None, phi = None, seed: int = None):
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
    return signals.T

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

def space_covariance_matrix(X: np.ndarray):
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
    return np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))


def initial_noise_covariance(X, theta, signals):
    """
    Вычисляет начальную диагональную оценку ковариации шума.
    """
    L = len(X[0])
    A = A_ULA(L, theta)
    # Остатки шума
    R = X.T - A @ signals.T 
    # Для каждого канала считаем дисперсию по времени
    Sigma_N_diag = np.nanvar(R, axis=1, ddof=0)  # ddof=0 — оценка по ML (делим на T)
    # Добавим регуляризацию, чтобы избежать нулевых дисперсий (по желанию)
    epsilon = 1e-6
    Sigma_N_diag = Sigma_N_diag + epsilon
    return np.diag(Sigma_N_diag)

def initializer(X: np.ndarray, M: int, seed: int = None):
    if seed is None:
        seed = 100
    theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M)
    signals = gds(len(X), M, seed=seed+20) 
    noise_cov = initial_noise_covariance(X, theta, signals)
    return theta, signals, noise_cov

def cost_theta(theta, X, S, Q_inv_sqrt):
    L, G = X.shape
    A = A_ULA(L, theta)
    cost = np.sum(Q_inv_sqrt @ (X - A @ S))
    return cost

def CM_step_theta(X, theta_guess, S, Q_inv_sqrt):
    res = minimize(
            lambda th: cost_theta(th, X.T, S.T, Q_inv_sqrt),
            theta_guess,
            method='L-BFGS-B',
            bounds=[(-np.pi/2, np.pi/2)] * len(theta_guess)
        )
    return res.x    

def CM_step_S(X, A, Q):
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X

def CM_step_noise_cov(X, A, S):
    R = X.T - A @ S.T 
    Sigma_Noise_diag = np.nanvar(R, axis=1, ddof=0)  
    epsilon = 1e-6
    Sigma_Noise_diag = Sigma_Noise_diag + epsilon
    return np.diag(Sigma_Noise_diag)

def EM(theta: np.ndarray, S: np.ndarray, Q: np.ndarray,  X: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    Q - ковариация шума;
    X - коллекция полученных сигналов;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    Q_vec = np.diagonal(Q)
    Q_inv_sqrt = 1.0 / Q_vec
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = np.cov(X[observed_rows, ].T)
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
                Q_o, Q_m = Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                K_OM = K_MO.T
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ signals[i])
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        new_theta = CM_step_theta(X.T, theta, S.T, Q_inv_sqrt)
        A = A_ULA(L, theta)
        new_S = CM_step_S(X.T, A, Q)
        new_Q = CM_step_noise_cov(X.T, A, new_S.T)


        
        #if np.linalg.norm(mu - mu_new) < rtol:
            #break        
        theta, S, Q = new_theta, new_S, new_Q
        

        EM_Iteration += 1
    return theta, neg_likelihood
    
    

def multi_start_EM(X: np.ndarray, M: int, num_of_starts: int = 20, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M - число источников;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_neg_lhd, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S, Q = initializer(X, M)
        est_theta, neg_lhd = EM(theta, S, Q, X, max_iter, eps)
        if neg_lhd < best_neg_lhd:
            best_neg_lhd, best_theta = neg_lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_neg_lhd





