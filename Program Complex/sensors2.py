import numpy as np
import scipy
import math
from functools import partial

dist_ratio = 0.5

def CN(size: int, number: int, Gamma: np.ndarray):
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