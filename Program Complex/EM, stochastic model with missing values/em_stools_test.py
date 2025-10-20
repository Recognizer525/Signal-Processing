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
    print(f'Shape of signals is {signals.shape} before')
    signals = signals.T
    print(f'Shape of signals is {signals.shape} after')
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
    return np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta))


def initializer(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/M)%(2 * np.pi) for i in range(M)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M) 
    P_diag = np.random.RandomState(seed).uniform(0.2, 5, M)
    return theta, np.diag(P_diag)

    
def cost_theta(theta, R, P, Q_inv):
    """
    theta - оценка углов прибытия;
    R - оценка ковариации наблюдений;
    P - оценка ковариции сигналов;
    Q_inv - обратная матрица к ковариации шума.
    """
    A = A_ULA(R.shape[0], theta)
    cost = np.trace(Q_inv @ (R - A @ P @ A.conj().T))
    return cost


def CM_step_theta(theta_init, R, P, Q_inv, true_theta, true_P):
    res = minimize(
            lambda th: cost_theta(th, R, P, Q_inv),
            theta_init,
            method='L-BFGS-B',
            bounds=[(-np.pi/2, np.pi/2)] * len(theta_init)
        )
    true_value = cost_theta(true_theta, R, true_P, Q_inv)
    false_value = cost_theta(res.x, R, P, Q_inv)
    print(f"true_value = {true_value}, false_value = {false_value}")
    return res.x    


def CM_step_P(mu, sigma):
    """
    mu - массив, составленный из векторов УМО исходного сигнала, в зависимости от наблюдений. Число столбцов соответствует числу наблюдений.
    sigma - условная ковариация исходного сигнала с учетом наблюдения.
    """
    G = len(sigma)
    res = (1/G) * mu @ mu.conj().T + sigma
    # Оставляем только диагональные элементы
    res = res * np.eye(res.shape[0], res.shape[1], dtype=np.complex128)
    return res


def cond_inv(A):
    is_invertible = np.linalg.matrix_rank(A) == A.shape[0]
    if is_invertible:
        inv_A = np.linalg.inv(A)
    else:
        inv_A = np.linalg.pinv(A)
    return inv_A


def incomplete_lkhd(X, theta, P, Q):
    A = A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q
    inv_R = cond_inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
            A_o, R_o = A[np.ix_(O_i, O_i)], R[np.ix_(O_i, O_i)]
            res += - np.linalg.det(R_o) - (X[i, O_i].T).conj().T @ cond_inv(R_o) @ (X[i, O_i].T)
        else:
            res += - np.linalg.det(R) - (X[i].T).conj().T @ inv_R @ (X[i].T)
    return res


def EM_attempt(theta: np.ndarray, P: np.ndarray, X: np.ndarray, Q: np.ndarray, true_theta, true_P):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    P - ковариация исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    """
    Q_inv = np.linalg.inv(Q)
    L = Q.shape[0]
    G = X.shape[0]

    print(f'Initial theta = {theta}')
    print(f'True theta = {true_theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = np.cov(X[observed_rows, ].T)
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_Xm_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    Mu_S_cond = np.zeros((L, G), dtype=np.complex128)
    K_S_cond = np.zeros(P.shape, dtype=np.complex128)
    X_modified = X.copy()

    A = A_ULA(L, theta)
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
            A_o, A_m = A[np.ix_(O_i, O_i)], A[np.ix_(M_i, M_i)]
            Q_o, Q_m = Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
            # Вычисляем блоки ковариации принятых сигналов (наблюдений)
            K_OO = K[np.ix_(O_i, O_i)]
            K_MM = K[np.ix_(M_i, M_i)]
            K_MO = K[np.ix_(M_i, O_i)]
            K_OM = K_MO.T
            # Оцениваем параметры апостериорного распределения ненаблюдаемых данных и пропущенные значения
            Mu_Xm_cond[i] = K_MO @ cond_inv(K_OO) @ X_modified[i, O_i]
            X_modified[i, M_i] = Mu_Xm_cond[i]
            K_Xm_cond_accum[np.ix_(M_i, M_i)] += K_MM - K_MO @ cond_inv(K_OO) @ K_OM
            # Вычисляем блоки совместной ковариации исходных и принятых сигналов
    K_XX = A @ P @ A.conj().T + Q
    K_SS = P
    K_XS = A @ P
    K_SX = K_XS.conj().T
    Mu_S_cond = K_SX @ cond_inv(K_XX) @ X_modified.T
    K_S_cond = K_SS - K_SX @ cond_inv(K_XX) @ K_XS

    # Шаги условной максимизации
    K = np.cov(X_modified.T)
    R = K + K_Xm_cond_accum / G
    new_theta = CM_step_theta(theta, R, P, Q_inv, true_theta, true_P)
    print(f'diff of theta is {new_theta-theta}')
    A = A_ULA(L, new_theta)
    new_P = CM_step_P(Mu_S_cond, K_S_cond)
    print(f'diff of P is {np.sum((new_P-P)**2)}')
    theta, P = new_theta, new_P
    lkhd = incomplete_lkhd(X_modified, theta, P, Q)
    print(f'incomplete likelihood is {lkhd.real}')
    return theta, lkhd


def test_EM(X: np.ndarray, M: int, Q: np.ndarray, true_theta = None, true_P = None):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M - число источников;
    Q - ковариация шума;
    true_theta - истинное значение углов прибытия;
    true_P - истинное значение ковариации сигналов.
    """

    for i in range(3):
        print(f'{i}-th start')
        theta, P = initializer(X, M, seed=i * 100)
        est_theta, est_lhd = EM_attempt(theta, P, X, Q, true_theta, true_P)
    return None


##########################################################################################################

