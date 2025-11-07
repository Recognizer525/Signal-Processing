import numpy as np
import torch
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
    

def A_ULA_torch(L, theta):
    """
    Создает матрицу управляющих векторов для массива сенсоров типа ULA (PyTorch)
    L - число сенсоров,
    theta - тензор углов прибытия (размер [n_angles])
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float()  # (L,1)
    return torch.exp(-2j * np.pi * dist_ratio * sensor_indices * torch.sin(theta))  # (L, n_angles)


def cost_theta_torch(theta, X, S, Q_inv_sqrt):
    """
    theta - тензор углов прибытия (requires_grad=True)
    X, S, Q_inv_sqrt - тоже тензоры PyTorch, dtype=torch.cfloat или torch.float
    """
    A = A_ULA_torch(X.shape[0], theta)  # (L, n_angles)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор

def CM_step_theta_start(X_np, theta0_np, S_np, Q_inv_sqrt_np, method='L-BFGS-B', tol=1e-6):
    """
    X_np, theta0_np, S_np, Q_inv_sqrt_np - numpy массивы
    """
    
    # Объявляем функцию для scipy, которая принимает numpy theta, внутри переводим в torch и вычисляем
    def fun(theta_np):
        theta_t = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
        X_t = torch.tensor(X_np, dtype=torch.cfloat)
        S_t = torch.tensor(S_np, dtype=torch.cfloat)
        Q_inv_sqrt_t = torch.tensor(Q_inv_sqrt_np, dtype=torch.cfloat)

        loss = cost_theta_torch(theta_t, X_t, S_t, Q_inv_sqrt_t)
        loss.backward()
        grad = theta_t.grad.detach().numpy().astype(np.float64)
        return loss.item(), grad

    res = minimize(lambda th: fun(th)[0], theta0_np, jac=lambda th: fun(th)[1], method=method, tol=tol)
    #print(f"Optim.res={res.success}")
    return res.x, res.fun

def CM_step_theta(X_np, theta0_np, S_np, Q_inv_sqrt_np, num_of_starts=5):
    best_theta, best_fun = None, np.inf
    for i in range(num_of_starts):
        if i == 0:
            est_theta, est_fun = CM_step_theta_start(X_np, theta0_np, S_np, Q_inv_sqrt_np)
        else:
            M = len(theta0_np)
            nu = np.random.RandomState(42+i).uniform(-np.pi, np.pi)
            theta = np.array([(nu + j * 2 * np.pi/M)%(2 * np.pi) for j in range(M)]) - np.pi
            est_theta, est_fun = CM_step_theta_start(X_np, theta, S_np, Q_inv_sqrt_np)
        if est_fun < best_fun:
            best_fun, best_theta = est_fun, est_theta
    return best_theta

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
            A_o, Q_o = A[O_i, :], Q[np.ix_(O_i, O_i)]
            res += - np.linalg.det(Q_o) - (X[i, O_i].T - A_o @ S[i].T).conj().T @ np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T)
        else:
            res += - np.linalg.det(Q) - (X[i].T - A @ S[i].T).conj().T @ inv_Q @ (X[i].T - A @ S[i].T)
    return res


def EM(theta: np.ndarray, S: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, rtol: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    S - вектор исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
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
                A_o, A_m = A[O_i, :], A[M_i, :]
                Q_o = Q[np.ix_(O_i, O_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        K = complex_cov(X_modified)
        new_theta = CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        print(f"new_theta={new_theta}")
        A = A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        #print(f'diff of S is {np.sum((new_S-S)**2)} on iteration {EM_Iteration}')
        lkhd = incomplete_lkhd(X_modified, new_theta, new_S, Q, np.linalg.inv(Q))
        if np.linalg.norm(theta - new_theta) < rtol and np.linalg.norm(S - new_S, ord = 2) < rtol:
            break
        theta, S = new_theta, new_S
        print(f'incomplete likelihood is {lkhd.real} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, S, lkhd


def multi_start_EM(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, rtol: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    M - число источников;
    Q - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.
    """
    best_lhd, best_theta, best_S = -np.inf, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S = initializer(X, M, seed=i * 100)
        est_theta, est_S, est_lhd = EM(theta, S, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_theta, best_S = est_lhd, est_theta, est_S
    best_theta = angle_correcter(best_theta)
    return best_theta, best_S, best_lhd


##########################################################################################################
