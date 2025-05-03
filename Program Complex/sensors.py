import numpy as np
import scipy
import math
from functools import partial

dist_ratio = 0.5

def CN(size:int, number:int, Gamma):
    """
    Генерирует комплексные нормальные вектора (circularly-symmetric case).
    Gamma - ковариационная матрица
    size - размер вектора
    number - количество векторов
    """
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
    """
    Метод генерирует управляющий вектор.
    """
    return np.exp(-2j * np.pi * dist_ratio * np.arange(Num_sensors) * np.sin(angle))
    
def space_covariance_matrix(X):
    """
    Метод предназначен для формирования оценки матрицы пространственной ковариации.
    """
    N = len(X)
    ans = np.zeros((len(X[0]), len(X[0])), dtype = np.complex128)
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


def f(theta, Ga_s, Ga_n, X, K, mu):
    M, L, G = Ga_s.shape[0], Ga_n.shape[0], X.shape[0]
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    inv_Ga_s, inv_Ga_n = np.linalg.inv(Ga_s), np.linalg.inv(Ga_n)
    
    inv_Ga_A = inv_Ga_n @ A
    A_H_inv_Ga = A_H @ inv_Ga_n
    A_H_inv_Ga_A = A_H @ inv_Ga_n @ A
    ans = 0
    for k in range(G):
        ans += -X[k].conj() @ inv_Ga_A @ mu[:, k]
        ans += -mu[:,k].conj().T @ A_H_inv_Ga @ X[k]
        ans += mu[:,k].conj().T @ A_H_inv_Ga_A @ mu[:,k] 
    return ans.real


def equation_solver(theta, Ga_s, Ga_n, X, K, mu):
    """
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    K - ковариация апостериорного распределения;
    mu - мат.ожидание апостериорного распределения.
    """
    simplified_f = partial(f, Ga_s=Ga_s, Ga_n=Ga_n, X=X, K=K, mu=mu)
    ans = scipy.optimize.minimize(simplified_f, theta.reshape(-1,), method='Nelder-Mead').x
    #pdce_real_wp = partial(pdce_real, Ga_s=Ga_s, Ga_n=Ga_n, X=X, K=K, mu=mu)
    #ans = gradient_descent(theta=theta, deriv_func=pdce_real_wp)
    #print(f"not_corrected_theta={ans}")
    #ans = correcter(ans)
    print(f'theta_new={ans}') 
    return ans


def EM(X, Ga_s, Ga_n, max_iter=20, eps=1e-6):
    """
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов.
    """
    no_conv = True
    iteration = 0
    M, L = Ga_s.shape[0], Ga_n.shape[0]
    theta = np.random.RandomState(30).uniform(-np.pi, np.pi, M).reshape(M,1)
    init_theta = theta.copy()
    print(f"Initial theta = {theta}")
    inv_Ga_s, inv_Ga_n = np.linalg.inv(Ga_s), np.linalg.inv(Ga_n)
    while no_conv and iteration < max_iter:
        #E-step
        A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
        A_H = A.conj().T
        K = Ga_s - Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ A @ Ga_s
        mu = Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ X.T
        print(f"K={K}")
        #print(f"mu={mu}")
        #M-step
        theta_new = equation_solver(theta, Ga_s, Ga_n, X, K, mu)
        no_conv = np.linalg.norm(theta - theta_new) >= eps
        if not no_conv:
            print(f"norm={np.linalg.norm(theta - theta_new)}")
        iteration += 1
        print(f"Iteration={iteration}")
        theta = theta_new
    theta = theta_new
    return theta


def goal_function(X, Ga_s, Ga_n, num_of_points):
    """
    Данный метод реализует Е-шаг алгоритма, затем, в отрезке [-pi; pi] выделяется заданное число равноудаленных точек, для каждой из которых вычисляется
    значение функции, которую нужно минимизировать на М-шаге.
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов.
    """
    initial_theta = np.random.RandomState(10).uniform(-np.pi, np.pi, 1)
    L = np.shape(Ga_n)[0]
    M = np.shape(Ga_s)[0]
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(initial_theta).reshape(1,-1))
    A_H = A.conj().T
    K = Ga_s - Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ A @ Ga_s
    mu = Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ X.T
    funct = partial(f, Ga_s=Ga_s, Ga_n=Ga_n, X=X, K=K, mu=mu)
    B = np.linspace(-np.pi, np.pi, num_of_points)
    f_B = np.zeros(num_of_points, dtype=np.complex128)
    for i in range(num_of_points):
        f_B[i] = funct(np.array([[B[i]]]))
    return B, f_B


'''
Методы, не используемые в текущей версии.
'''

def dA(angle, Num_sensors, Num_emitters, i):
    """
    angle - точка, в которой ищем производную;
    i - компонент theta, по которому ищем производную;
    Num_sensors - число датчиков;
    Num_emitters - число источников.
    """
    A = np.zeros((Num_sensors, Num_emitters), dtype=np.complex128)
    A[:, i] = -2j * np.pi * dist_ratio * np.cos(angle) * np.exp(-2j * np.pi * dist_ratio * np.arange(Num_sensors) * np.sin(angle))
    #print(f'dA={A}')
    return A

def pdce(i, theta, Ga_s, Ga_n, X, K, mu):
    """
    pdce - частная производная условного мат.ожидания, которое нужно оптимизировать; 
    i - номер компоненты вектора углов, для которой ищем частную производную;
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    K - ковариация апостериорного распределения;
    mu - мат.ожидание апостериорного распределения.
    """
    M, L, G = Ga_s.shape[0], Ga_n.shape[0], X.shape[0]
    A = np.zeros((L, M), dtype=np.complex128)
    for i2 in range(M):
        A[:,i2] = np.exp(-2j * np.pi * dist_ratio * np.arange(L) * np.sin(theta[i2]))
    #print(f"A={A}")
    A_H = A.conj().T
    deriv_A = dA(theta[i,0], L, M, i) 
    deriv_A_H = deriv_A.conj().T
    inv_Ga_s = np.linalg.inv(Ga_s)
    inv_Ga_n = np.linalg.inv(Ga_n)
    ans = 0
    for k in range(G):
        P1 = mu[:,k].conj().T
        P2 = -deriv_A_H @ inv_Ga_n @ X[k]
        P3 = deriv_A_H @ inv_Ga_n @ A @ mu[:,k]
        P4 = A_H @ inv_Ga_n @ deriv_A @ mu[:,k]
        P5 = - X[k].conj() @ inv_Ga_n @ deriv_A @ mu[:, k]
        ans += P1@P2 + P1@P3 + P1@P4 + P5
    #print(f"derivative of X={ans.real} for theta_i={i}")
    return ans

def pdce_real(i, theta, Ga_s, Ga_n, X, K, mu):
    return pdce(i, theta, Ga_s, Ga_n, X, K, mu).real

def gradient_descent(theta, deriv_func, lr = 0.1, iters = 5):
    G = np.zeros(theta.shape)
    ans = theta.copy()
    for i in range(iters):
        #print(f"GD_iter={i+1}")
        for k in range(G.shape[0]):
            G[k,0] = deriv_func(i=k, theta=ans)
        ans = ans - lr * G
        print('lr*G', lr*G)
    return ans

def correcter(theta):
    """
    Работает с углами, удаляя дополнительные периоды. Т.е., мы вычитаем/добавляем 2*pi, до тех пор, пока угол не будет принадлежать отрезку [-2*pi, 2*pi]
    """
    for i in range(theta.shape[0]):
        while theta[i,0] > np.pi:
            theta[i,0] -= 2*np.pi
        while theta[i,0] < -np.pi:
            theta[i,0] += 2*np.pi
    #print(f'theta={theta}')
    return theta