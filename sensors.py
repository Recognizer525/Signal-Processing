import numpy as np

def CN(size:int, number:int, Gamma):
    # Размер ковариационной матрицы совместного распределения 
    n = 2 * size
    C = np.zeros((n,n), dtype=np.float64)
    C[:size,:size] = Gamma.real
    C[size:,size:] = Gamma.real
    C[:size,size:] = -Gamma.imag
    C[size:,:size] = Gamma.imag
    mu = np.zeros(n)
    B = np.random.multivariate_normal(mu, 0.5*C, number)
    D = B[:,:size] + 1j * B[:, size:]
    return D 

def steering_vector(dist_ratio, angle, Num_sensors):
    return np.exp(-2j * np.pi * dist_ratio * np.arange(Num_sensors) * np.sin(angle))
    
def space_covariance_matrix(X):
    """
    Метод предназначен для формирования оценки матрицы пространственной ковариации.
    """
    N = len(X)
    ans = 0
    for i in range(len(X)):
        ans += X[i][:, None] @ X[i][:, None].conj().T
    return ans * (1/N)

def bartlett_func(a, R):
    """
    Выходная мощность для формирователя луча Bartlett.
    """
    return (a[:,None].conj().T @ R @ a[:, None] / (a[:,None].conj().T @ a[:, None]))[0,0]

def capon_func(a, R):
    """
    Выходная мощность для формирователя луча CAPON.
    """
    return 1/(a[:,None].conj().T @ np.linalg.inv(R) @ a[:,None])[0,0]