import numpy as np
import scipy
import math
from functools import partial

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

def generate_deterministic_signals(M, N, A=2, fs=1000):
    """
    A - амплитуда
    M - число источников
    N - размер выборки
    """
    t = np.arange(N) / fs
    signals = np.zeros((M, N), dtype=complex)

    for m in range(M):
        freq = 50 + 10 * m
        phase = np.pi / 4 * m
        signals[m, :] = A * np.exp(1j * (2 * np.pi * freq * t + phase))

    return signals.T

def generate_stochastic_signals(size: int, number: int, Gamma: np.ndarray):
    """
    Генерирует комплексные нормальные вектора (circularly-symmetric case).
    size - размер вектора;
    number - количество векторов;
    Gamma - ковариационная матрица.
    """ 
    n = 2 * size # Размер ковариационной матрицы совместного распределения
    C = np.zeros((n,n), dtype=np.float64)
    C[:size,:size] = Gamma.real
    C[size:,size:] = Gamma.real
    C[:size,size:] = -Gamma.imag
    C[size:,:size] = Gamma.imag
    mu = np.zeros(n)
    B = np.random.RandomState(70).multivariate_normal(mu, 0.5*C, number)
    D = B[:,:size] + 1j * B[:, size:]
    return D


def space_covariance_matrix(X: np.ndarray):
    """
    Метод предназначен для формирования оценки матрицы пространственной ковариации.
    X - коллекция полученных сигналов.
    """
    N = len(X)
    ans = np.zeros((len(X[0]), len(X[0])), dtype = np.complex128)
    for i in range(len(X)):
        ans += X[i][:, None] @ X[i][:, None].conj().T
    return ans * (1/N)

def MUSIC(a: np.ndarray, R: np.ndarray, M: int):
    """
    Выходная мощность для формирователя луча MUSIC.
    a - управляющий вектор;
    M - число сигналов;
    R - матрица пространственной ковариации.
    """
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    E_n = eigvecs[:, M:]
    return 1/(a[:,None].conj().T @ E_n @ E_n.conj().T @ a[:,None])[0,0]

def angle_correcter(theta: np.ndarray):
    """
    Набор углов преобразуется таким образом, чтобы все углы были в области [-pi/2; pi/2], для этого по мере необходимости добавляется/вычитается 2*pi 
    требуемое число раз, кроме того, учитывается тот факт, что синус симметричен относительно pi/2 и -pi/2.
    theta - вектор углов, которые соответствуют DOA.
    """
    for i in range(len(theta)):
        while theta[i] > np.pi:
            theta[i] -= 2*np.pi
        while theta[i] < -np.pi:
            theta[i] += 2*np.pi

    for i in range(len(theta)):
        if theta[i] > np.pi/2:
            theta[i] = np.pi - theta[i]
        elif theta[i] < -np.pi/2:
            theta[i] = - np.pi - theta[i]
    return theta


def EM(theta: np.ndarray, signals: np.ndarray, X: np.ndarray, M: int, Q: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    signals - вектор исходных сигналов;
    X - коллекция полученных сигналов;
    M - число источников;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = np.cov(X[observed_rows, ].T)
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_cond = {}
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m, Q_o, Q_m = A[np.ix_(O_i, O_i)], A[np.ix_(M_i, M_i)], Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                K_OM = K_MO.T
                Mu_cond[i] = A_m @ signals[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ signals[i])
                X_modified[i, M_i] = Mu_cond[i]
        mu_new = np.mean(X_modified, axis = 0)
        #if np.linalg.norm(mu - mu_new) < rtol:
            #break
        theta_new = equation_solver1(X_modified, signals, len(Q))
        signals_new = equation_solver2(X_modified, len(Q), theta_new)
        
        theta = theta_new
        signals = signals_new

        EM_Iteration += 1
    return theta, neg_likelihood


def initializer(X: np.ndarray, M: int):
    theta = np.random.uniform(-np.pi, np.pi, M).reshape(M,1)
    L = len(X[0])
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    X_clean = X[~np.isnan(X).any(axis=1)].T
    return theta, (np.linalg.pinv(A) @ X_clean).T
    
    

def multi_start_EM(X: np.ndarray, M: int, Ga_n: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M - число источников;
    Ga_n - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_neg_lhd, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, signals = initializer(X, M)
        est_theta, neg_lhd = EM(theta, signals, X, M, Ga_n, max_iter, eps)
        if neg_lhd < best_neg_lhd:
            best_neg_lhd, best_theta = neg_lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_neg_lhd


##########################################################################################################
def generate_initial_signals(G, K):
    # G - размер выборки, M - число источников
    A = np.random.uniform(0.5, 1.5, M)         # амплитуды
    f = np.random.uniform(0.01, 0.1, M)        # нормированные частоты
    phi = np.random.uniform(0, 2*np.pi, M)     # фазы
    
    g = np.arange(G)
    signals = np.zeros((M, G), dtype=complex)
    for m in range(M):
        signals[m] = A[m] * np.exp(1j * (2 * np.pi * f[m] * g + phi[m]))
    return signals.T


def EM(X: np.ndarray, max_iter: int = 20, rtol: float = 1e-8) -> np.ndarray:
    '''
    Функция применяет алгоритм максимального правдоподобия к полученным данным для восстановления пропущенных значений.
    '''
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = np.cov(X[observed_rows, ].T)
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
    Mu_cond, K_cond_accum = {}, np.zeros((X.shape[1], X.shape[1]))
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_m, A_o = A[np.ix_(M_i, M_i)], A[np.ix_(O_i, O_i)]
                
                K_MM, K_MO, K_OO = K[np.ix_(M_i, M_i)], K[np.ix_(M_i, O_i)], K[np.ix_(O_i, O_i)]
                K_OM = K_MO.T
                Mu_cond[i] = Mu[np.ix_(M_i)] + K_MO @ np.linalg.inv(K_OO) @ (X_modified[i, O_i] - Mu[np.ix_(O_i)])
                X_modified[i, M_i] = Mu_cond[i]
                K_cond = K_MM - K_MO @ np.linalg.inv(K_OO) @ K_OM
                K_cond_accum[np.ix_(M_i, M_i)] += K_cond
        Mu_new, K_new = np.mean(X_modified, axis = 0), np.cov(X_modified.T, bias = 1) + K_cond_accum / X.shape[0]
        if np.linalg.norm(Mu - Mu_new) < rtol and np.linalg.norm(K - K_new, ord = 2) < rtol:
            break
        Mu, K = Mu_new, K_new
        for i in range(K.shape[0]):
            assert K[i,i]>=0, f'Variance of {i} feature on iteration {EM_Iteration} is negative'
            assert np.linalg.det(K)>=0, f'Determinant of Covariance matrix on iteration {EM_Iteration} is negative'
        EM_Iteration += 1
    return X_modified