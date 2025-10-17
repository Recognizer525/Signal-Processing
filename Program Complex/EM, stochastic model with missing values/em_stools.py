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

def initializer(X: np.ndarray, M: int, seed: int = None):
    if seed is None:
        seed = 100
    theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M).reshape(M,1)
    L = len(X[0])
    P = np.random.RandomState(seed+10).uniform(0.25, 4, L).reshape(L)
    return theta, np.diag(P)


def cost_theta(theta, P, R, Q):
    """
    theta - вектор углов прибытия;
    P - оценка матрицы ковариации исходных сигналов, полученная на предыдущем шаге;
    R - текущая оценка пространственной ковариационной матрицы сигналов;
    Q - матрица, обратная к ковариационной матрице шума;
    """
    A = A_ULA(R.shape[0], theta)
    return Q @ (R - A.conj().T @ P @ A)

def CM_step_theta(theta_guess, P, R, Q):
    res = minimize(
            lambda th: cost_theta(th, P, R, Q),
            theta_guess,
            method='L-BFGS-B',
            bounds=[(-np.pi/2, np.pi/2)] * len(theta_guess)
        )
    return res.x

def CM_step_P():
    pass

def likelihood(X, theta, S, Q, inv_Q, P, inv_P):
    """
    X - выборка, состоящая из принятых сигналов, с учетом оценок пропущенных значений, 
    каждый столбец соответствует одному наблюдению;
    theta - оценка вектора углов;
    S - оценка сигналов, каждый столбец соответствует одному сигналу;
    Q - матрица ковариации шума;
    inv_Q - матрица, обратная к Q;
    P - матрица ковариации сигнала;
    inv_P - матрица, обратная к P.
    """
    A = A_ULA(X.shape[0], theta)
    M = X - A @ S
    return - X.shape[1] * np.linalg.det(Q) - np.trace(M.conj().T @ inv_Q @ M) - X.shape[1] * np.linalg.det(P) - np.trace(S.conj().T @ inv_P @ S)

def EM(X: np.ndarray, Q: np.ndarray, theta: np.ndarray, P: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    theta - вектор углов, которые соответствуют DOA;
    P - ковариация сигнала;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    return theta, neg_likelihood



def multi_start_EM(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M -  число источников;
    Q - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_lhd, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        init_theta, init_sig_cov = initializer(seed = 3 * i + 8)
        est_theta, lhd = EM(X, init_theta, init_sig_cov, Q, max_iter, eps)
        if lhd < best_lhd:
            best_lhd, best_theta = lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_lhd