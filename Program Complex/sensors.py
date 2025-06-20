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

def bartlett_func(a: np.ndarray, R: np.ndarray):
    """
    Выходная мощность для формирователя луча Bartlett.
    a - управляющий вектор;
    R - матрица пространственной ковариации.
    """
    return (a[:,None].conj().T @ R @ a[:, None] / (a[:,None].conj().T @ a[:, None]))[0,0]

def capon_func(a: np.ndarray, R: np.ndarray):
    """
    Выходная мощность для формирователя луча CAPON.
    a - управляющий вектор;
    R - матрица пространственной ковариации.
    """
    return 1/(a[:,None].conj().T @ np.linalg.inv(R) @ a[:,None])[0,0]

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

def ESPRIT(M: int, L: int, R: np.ndarray):
    """
    Оценки углов для ESPRIT.
    M - число сигналов;
    L - число элементов антенной решетки;
    R - матрица пространственной ковариации.
    """
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    E = eigvecs[:, :signal_count]
    E1 = E[:-1, :]
    E2 = E[1:, :]
    Psi = np.linalg.pinv(E1) @ E2
    eigvals_psi, _ = np.linalg.eig(Psi)
    angles = np.arcsin(np.angle(eigvals_psi) / (2 * dist_ratio * np.pi))
    return np.sort(angles)


def f(theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, K: np.ndarray, mu: np.ndarray):
    """
    Минимизируемая, на М-шаге, функция.
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    K - ковариация апостериорного распределения;
    mu - мат.ожидание апостериорного распределения.
    """
    M, L, G = Ga_s.shape[0], Ga_n.shape[0], X.shape[0]
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    inv_Ga_s, inv_Ga_n = np.linalg.inv(Ga_s), np.linalg.inv(Ga_n)
    inv_Ga_A, A_H_inv_Ga, A_H_inv_Ga_A = inv_Ga_n @ A, A_H @ inv_Ga_n, A_H @ inv_Ga_n @ A
    ans = 0
    ans1 = sum([-X[k].conj() @ inv_Ga_A @ mu[:, k] for k in range(G)])
    ans2 = sum([-mu[:,k].conj().T @ A_H_inv_Ga @ X[k] for k in range(G)])
    ans3 = sum([mu[:,k].conj().T @ A_H_inv_Ga_A @ mu[:,k] for k in range(G)])
    ans = ans1 + ans2 + ans3
    return ans.real


def equation_solver(theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, K: np.ndarray, mu: np.ndarray):
    """
    Нахождение оптимального вектора углов на М-шаге.
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    K - ковариация апостериорного распределения;
    mu - мат.ожидание апостериорного распределения.
    """
    simplified_f = partial(f, Ga_s=Ga_s, Ga_n=Ga_n, X=X, K=K, mu=mu)
    ans = scipy.optimize.minimize(simplified_f, theta.reshape(-1,), method='Nelder-Mead').x
    return ans, simplified_f(ans)


def EM(theta: np.ndarray, X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    X - коллекция полученных сигналов;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    no_conv = True
    iteration = 0
    M, L = Ga_s.shape[0], Ga_n.shape[0]
    #print(f"Initial theta = {theta}")
    inv_Ga_s, inv_Ga_n = np.linalg.inv(Ga_s), np.linalg.inv(Ga_n)
    while no_conv and iteration < max_iter:
        #E-step
        A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
        A_H = A.conj().T
        K = Ga_s - Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ A @ Ga_s
        mu = Ga_s @ A_H @ np.linalg.inv(A @ Ga_s @ A_H + Ga_n) @ X.T
        #print(f"K={K}")
        #print(f"mu={mu}")
        #M-step
        theta_new, neg_likelihood = equation_solver(theta, Ga_s, Ga_n, X, K, mu)
        no_conv = np.linalg.norm(theta - theta_new) >= eps
        if not no_conv:
            print(f"norm={np.linalg.norm(theta - theta_new)}")
        iteration += 1
        print(f"Iteration={iteration}, theta_new={theta_new:}, -likelihood = {neg_likelihood:.5f}")
        theta = theta_new
    return theta, neg_likelihood, K, mu


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


def multi_start_EM(X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_neg_lhd, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        M = Ga_s.shape[0]
        theta = np.random.uniform(-np.pi, np.pi, M).reshape(M,1)
        est_theta, neg_lhd, K, mu = EM(theta, X, Ga_s, Ga_n, max_iter, eps)
        if neg_lhd < best_neg_lhd:
            best_neg_lhd, best_theta = neg_lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_neg_lhd, K, mu


def dA(theta: float, L: int, M: int, i: int):
    """
    Вычисление производной для матрицы опорных векторов.
    theta - оценка DoA, по которой составляется производная матрицы управляющих векторов;
    L - число датчиков;
    M - число источников;
    i - компонент theta, по которому происходит дифференцирование матрицы.
    """
    dev_A = np.zeros((L, M), dtype=np.complex128)
    dev_A[:, i] = -2j * np.pi * dist_ratio * np.cos(theta[i]) * np.exp(-2j * np.pi * dist_ratio * np.arange(L) * np.sin(theta[i]))
    #print(f'dev_A={dev_A}')
    return dev_A


def dCov(theta: np.ndarray, Ga_s: np.ndarray, L: int, M: int, i: int):
    """
    Вычисление производной для ковариации распределения полученных сигналов.
    theta - оценка DoA, по которой составляется матрица управляющих векторов;
    Ga_s - ковариация сигнала;
    L - число датчиков;
    M - число источников;
    i - номер компоненты theta, по которой ищем частную производную.
    """
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    dev_A = dA(theta, L, M, i)
    dev_A_H = dev_A.conj().T
    dev_cov = dev_A @ Ga_s @ A_H + A @ Ga_s @ dev_A_H
    return dev_cov

def d_ML1(X: np.ndarray, theta: np.ndarray, L: int, M: int):
    """
    Вычисление градиента функции ML1.
    X - коллекция полученных сигналов;
    theta - начальная оценка направлений прибытия сигнала;
    L - число датчиков;
    M - число источников.
    """
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    W2 = A_H @ A
    inv_W2 = np.linalg.inv(W2)
    I = np.eye(L, dtype=np.float64)
    P = I - A @ inv_W2 @ A_H
    R = space_covariance_matrix(X)
    
    est_sigma2 = np.trace(P@R)/(L-M)
    est_Gamma = inv_W2 @ A_H @ (R - est_sigma2*I) @ A @ inv_W2

    inv1 = np.linalg.inv(A_H @ R @ A)
    inv2 = np.linalg.inv(A_H @ A)
    left_expr = (inv1 - inv2 / est_sigma2) @ A_H @ R @ P
    dev_A = [dA(theta, L, M, i) for i in range(M)]
    dev_ML2 = [2*np.trace(left_expr @ dev_A[i]).real for i in range(M)]
    return dev_ML2

def ML1(theta: np.ndarray, L: int, M: int, X: np.ndarray):
    """
    Вычисление максимального правдоподобия в соответствии со статьей Rao (1994). Предполагается, что ковариационные матрицы сигнала и шума неизвестны.
    theta - начальная оценка DoA;
    L - число датчиков;
    M - число источников;
    X - коллекция полученных сигналов.
    """
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    R = space_covariance_matrix(X)
    W1 = A_H @ R @ A
    W2 = A_H @ A
    inv_W2 = np.linalg.inv(W2)
    I = np.eye(L, dtype=np.float64)
    P = I - A @ inv_W2 @ A_H
    L1 = (L - M) * np.log(np.trace(P @ R)) + np.log(np.linalg.det(W1)) - np.log(np.linalg.det(W2))
    return L1.real

def ML1_solution(theta: np.ndarray, X: np.ndarray, L: int, M: int, method: str = 'Nelder-Mead'):
    """
    Запуск поиска параметров со случайно выбранной начальной точки.
    theta - начальная оценка DoA;
    L - число датчиков;
    M - число источников;
    X - коллекция полученных сигналов;
    method - используемый метод оптимизации.
    """

    ML1_with_one_arg = partial(ML1, L=L, M=M, X=X)
    if method == 'Nelder-Mead':
        ans = scipy.optimize.minimize(ML1_with_one_arg, theta.reshape(-1,), method='Nelder-Mead').x
        return ans, ML1_with_one_arg(ans).real
    if method in ['BFGS', 'Newton-CG', 'CG']:
        dev_ML1 = d_ML1(X, theta, L, M)
        ans = scipy.optimize.minimize(ML1_with_one_arg, theta.reshape(-1,), jac=dev_ML1, method=method).x
        return ans, ML1_with_one_arg(ans).real

def multi_start_ML1(X: np.ndarray, L: int, M: int, num_of_starts: int = 20, method: str = 'Nelder-Mead'):
    """
    Метод реализует нахождение углов, соответствующих максимальному правдоподобию (на основе статьи Рао 1994 года). 
    Для нахождения глобального оптимума используется мультистарт.
    X - коллекция полученных сигналов;
    L - число датчиков;
    M - число источников;
    num_of_starts - число запусков.
    """
    best_func_val, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta = np.random.uniform(-np.pi, np.pi, M).reshape(M,1)
        est_theta, func_val = ML1_solution(theta, X, L, M, method)
        if func_val < best_func_val:
            best_func_val, best_theta = func_val, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_func_val


def d_ML2(X: np.ndarray, theta: np.ndarray, Ga_s: np.ndarray, sigma2: float, L: int, M: int):
    """
    Вычисление градиента функции ML2.
    X - коллекция полученных сигналов;
    theta - начальная оценка направлений прибытия сигнала;
    Ga_s - ковариация сигнала;
    sigma2 - величина на которую умножаем единичную матрицу, чтобы получить ковариацию шума;
    L - число датчиков;
    M - число источников.
    """
    N = len(X)
    dev_Cov = [dCov(theta, Ga_s, L, M, i) for i in range(M)]
    I = np.eye(L, dtype=np.float64)
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    Cov = A @ Ga_s @ A_H + sigma2 * I
    inv_Cov = np.linalg.inv(Cov)
    R = space_covariance_matrix(X)
    dev_ML2 = [N*np.trace((I - inv_Cov @ R) @ inv_Cov @ dev_Cov[i]).real for i in range(M)]
    return dev_ML2


def ML2(theta: np.ndarray, L: int, M: int, Ga_s: np.ndarray, sigma2: np.ndarray, X: np.ndarray):
    """
    Вычисление максимального правдоподобия в соответствии со статьей Rao (1994). Предполагается, что ковариационные матрицы сигнала и шума известны.
    theta - начальная оценка DoA;
    L - число датчиков;
    M - число источников;
    Ga_s - ковариация сигнала;
    sigma2 - величина на которую умножаем единичную матрицу, чтобы получить ковариацию шума;
    X - коллекция полученных сигналов.
    """
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    A_H = A.conj().T
    N = len(X)
    I = np.eye(L, dtype=np.float64)
    Cov = A @ Ga_s @ A_H + sigma2 * I
    R = space_covariance_matrix(X)
    L2 = N * np.log(np.linalg.det(Cov)) + N * np.trace(np.linalg.inv(Cov) @ R)
    return L2.real


def ML2_solution(theta: np.ndarray, X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, method: str = 'Nelder-Mead'):
    """
    Запуск поиска параметров со случайно выбранной начальной точки.
    theta - начальная оценка DoA;
    X - коллекция полученных сигналов;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума.
    """
    sigma2 = Ga_n[0,0]
    L = Ga_n.shape[0]
    M = Ga_s.shape[0]
    ML2_with_one_arg = partial(ML2, L=L, M=M, Ga_s=Ga_s, sigma2=sigma2, X=X)
    if method == 'Nelder-Mead':
        ans = scipy.optimize.minimize(ML2_with_one_arg, theta.reshape(-1,), method='Nelder-Mead').x
        return ans, ML2_with_one_arg(ans).real
    if method in ['BFGS', 'Newton-CG', 'CG']:
        dev_ML2 = d_ML2(X, theta, Ga_s, sigma2, L, M)
        ans = scipy.optimize.minimize(ML2_with_one_arg, theta.reshape(-1,), jac=dev_ML2, method=method).x
        return ans, ML2_with_one_arg(ans).real

def multi_start_ML2(X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, num_of_starts: int = 20, method: str = 'Nelder-Mead'):
    """
    Метод реализует нахождение углов, соответствующих максимальному правдоподобию (на основе статьи Рао 1994 года). 
    Для нахождения глобального оптимума используется мультистарт.
    X - коллекция полученных сигналов;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    num_of_starts - число запусков.
    """
    best_func_val, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        M = Ga_s.shape[0]
        theta = np.random.uniform(-np.pi, np.pi, M).reshape(M,1)
        est_theta, func_val = ML2_solution(theta, X, Ga_s, Ga_n, method)
        if func_val < best_func_val:
            best_func_val, best_theta = func_val, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_func_val


def goal_function_EM(X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, num_of_points: int):
    """
    Данный метод реализует Е-шаг алгоритма, затем, в отрезке [-pi; pi] выделяется заданное число равноудаленных точек, для каждой из которых вычисляется
    значение функции, которую нужно минимизировать на М-шаге.
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    num_of_points - число точек, для которых определяем значение функции.
    """
    initial_theta = np.random.RandomState(10).uniform(-np.pi, np.pi, 1)
    L, M = np.shape(Ga_n)[0], np.shape(Ga_s)[0]
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


def pdce(i: int, theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, K: np.ndarray, mu: np.ndarray):
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

def pdce_real(i: int, theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, K: np.ndarray, mu: np.ndarray):
    return pdce(i, theta, Ga_s, Ga_n, X, K, mu).real

def gradient_descent(theta: np.ndarray, deriv_func, lr: float = 0.1, iters: int = 5):
    G = np.zeros(theta.shape)
    ans = theta.copy()
    for i in range(iters):
        #print(f"GD_iter={i+1}")
        for k in range(G.shape[0]):
            G[k,0] = deriv_func(i=k, theta=ans)
        ans = ans - lr * G
        print('lr*G', lr*G)
    return ans


# Functions for case, when X has unobsevred part
def EM2():
    pass

def multistart2():
    pass

