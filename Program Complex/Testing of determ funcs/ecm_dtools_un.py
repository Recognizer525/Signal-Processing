import numpy as np
import torch
from functools import partial
from scipy.optimize import minimize
from numpy.linalg import eig
from scipy.signal import find_peaks

import sensors

dist_ratio = 0.5

def MCAR(X: np.ndarray, mis_cols: object, num_mv: object , rs: int = 42) -> np.ndarray:
    '''
    Реализует создание абсолютно случайных пропусков.

    Параметры:
    X: np.ndarray
      Двумерный массив, представляет из себя выборку, состоящую из наблюдений, каждому из них соответствует своя строка.
    mis_cols: object
      Целое число (int), либо список (list[int]). Указывает на индексы столбцов, в которые следует добавить пропуски.
    num_mv: object 
      Целое число (int), либо список (list[int]). Указывает на количество пропусков, которые следует добавить 
      в каждый столбец из числа указанных в mis_cols.
    rs: int
      Целое число (int), соответствует randomstate для выбора конкретных позиций, где будут размещены пропуски.

    Возвращает:
    X1: np.ndarray
      Двумерный массив, представляет собой выборку, в которую добавили абсолютно случайные пропуски.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    if type(num_mv)==int:
        num_mv=[num_mv]
    assert len(mis_cols)==len(num_mv)
    X1 = X.copy()
    for i in range(len(mis_cols)):
        h = np.array([1]*num_mv[i]+[0]*(len(X)-num_mv[i]))
        np.random.RandomState(rs+i).shuffle(h)
        X1[:,mis_cols[i]][np.where(h==1)] = np.nan
    return X1


def gds(M, G, A = None, f = None, phi = None, seed: int = None):
    """
    Генерирует детерминированные сигналы, представляющие из себя комплексные нормальные вектора с круговой симметрией.
    
    Параметры:
    M: int
      Целое число (int), соответствует числу источников.
    G: int
      Целое число (int), соответствует числу наблюдений.
    A: np.ndarray
      Одномерный массив размера M, в котором указаны амплитуды сигналов от источников.
    f: np.ndarray
      Одномерный массив размера M, в котором указаны частоты сигналов от источников.
    phi: np.ndarray
      Одномерный массив размера M, в котором указаны фазы сигналов от источников.
    seed: int
      Целое число (int), randomstate для генерации амплитуд, частот и фаз, если таковые не указаны пользователем.

    Возвращает:
    signals: np.ndarray
      Сгенерированные сигналы в формате двумерного массива numpy размера (G,M).
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


def gss(M: int, G: int, Gamma: np.ndarray, seed: int = None):
    """
    Генерирует детерминированные сигналы, представляющие из себя комплексные нормальные вектора с круговой симметрией.
    
    Параметры:
    M: int
      Целое число (int). Количество компонент, составляющих вектор сигнала.
    G: int
      Целое число (int). Размер создаваемой выборки.
    Gamma: np.ndarray
      Двумерный массив, соответствует ковариационной матрице создаваемых сигналов.

    Возвращает:
    signals: np.ndarray
      Сгенерированные сигналы в формате двумерного массива размера (G,M).
    """
    if seed is None:
        seed = 70
    n = 2 * M # Размер ковариационной матрицы совместного распределения
    C = np.zeros((n,n), dtype=np.float64)
    C[:M,:M] = Gamma.real
    C[M:,M:] = Gamma.real
    C[:M,M:] = -Gamma.imag
    C[M:,:M] = Gamma.imag
    mu = np.zeros(n)
    B = np.random.RandomState(seed).multivariate_normal(mu, 0.5*C, G)
    signals = B[:,:M] + 1j * B[:, M:]
    return signals


def complex_cov(X: np.ndarray):
    """
    Вычисляет оценку пространственной ковариационной матрицы.

    Параметры:
    X: np.ndarray
      Двумерный массив. Представляет собой коллекцию полученных сигналов. Каждая строка соответствует одному вектору сигналов.
   
    Возвращает:
    cov: np.ndarray
      Двумерный массив. Представляет собой оценку ковариационной матрицы.
    """
    return np.einsum('ni,nj->ij', X, X.conj()) / X.shape[0]


def robust_complex_cov(X: np.ndarray):
    """
    Вычисляет оценку пространственной ковариационной матрицы, таким образом, чтобы обнулить мнимую часть диагональных элементов.

    Параметры:
    X: np.ndarray
      Двумерный массив. Представляет собой коллекцию полученных сигналов. Каждая строка соответствует одному вектору сигналов.

    Возвращает:
    K: np.ndarray
      Двумерный массив. Представляет собой оценку ковариационной матрицы.
    """
    K = np.einsum('ni,nj->ij', X, X.conj()) / X.shape[0]
    K = (K + K.conj().T)/2
    return K


def angle_correcter(theta: np.ndarray) -> np.ndarray:
    """
    Приводит углы к диапазону [-pi/2, pi/2], сохраняя то же значение синуса.

    Параметры:
    theta: np.ndarray
      Одномерный массив размера (M,1). Соответствует DoA.
    
    Возвращает:
    theta: np.ndarray
      Одномерный массив размера (M,1). Соответствует DoA. Значение углов скорректированы, 
      применение синуса к новому вектору углов возвращает те же значения, что и применение синуса к старому вектору углов.
    """
    # Приведение к диапазону [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    # Отражение значений, выходящих за пределы [-pi/2, pi/2]
    mask = theta > np.pi/2
    theta[mask] = np.pi - theta[mask]
    mask = theta < -np.pi/2
    theta[mask] = -np.pi - theta[mask]
    return theta


def A_ULA(L: int, theta:np.ndarray):
    """
    Создает матрицу векторов направленности для равномерного линейного массива сенсоров (ULA).

    Параметры:
    L: int
      Целое число (int). Соответствует числу сенсоров.
    theta: np.ndarray
      Одномерный массив размера (M,1). Соответствует DoA.
    
    Возвращает:
    A: np.ndarray
      Двумерный массив размера (L,M). Матрица векторов направленности.
    """
    return np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta))


def MUSIC_DoA(R, num_sources, scan_angles=np.arange(-90, 90.5, 0.5)):
    """
    Вычисляет оценку DoA через использование алгоритма MUSIC.

    Параметры:
    R: np.ndarray
      Двумерный массив, соответствует пространственной ковариационной матрице.
    num_sources: int
      Целое число, соответствует числу источников.
    scan_angles: np.ndarray
      Одномерный массив, соответствует сетке углов.

    Возвращает:
    doa_estimates: np.ndarray
      Одномерный массив, соответствует оценкам углов прибытия.
    """
    L = R.shape[0]
    eigvals, eigvecs = eig(R)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, :-num_sources] 
    P_music = []
    for theta in scan_angles:
        a = np.exp(-1j * 2 * np.pi * 0.5 * np.arange(L) * np.sin(np.deg2rad(theta)))
        a = a.reshape(-1, 1)
        denom = np.conjugate(a.T) @ En @ np.conjugate(En.T) @ a
        P_music.append(1 / np.abs(denom)[0, 0])

    P_music = np.array(P_music)
    P_music_db = 10 * np.log10(P_music / np.max(P_music))
    peaks, _ = find_peaks(P_music_db, distance=5)
    peak_vals = P_music_db[peaks]
    top_peaks = peaks[np.argsort(peak_vals)[-num_sources:]]
    doa_estimates = np.sort(scan_angles[top_peaks])
    return np.deg2rad(doa_estimates)


def initial_noise_covariance(X, theta, signals):
    """
    Вычисляет начальную диагональную оценку ковариации шума.
    """
    L = len(X[0])
    A = A_ULA(L, theta)
    # Остатки шума
    R = X.T - A @ signals.T 
    # Для каждого канала считаем дисперсию по времени
    Sigma_N_diag = np.nanvar(R, axis=1, ddof=0)
    epsilon = 1e-6
    Sigma_N_diag = Sigma_N_diag + epsilon
    return np.diag(Sigma_N_diag)


def initializer(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    if seed is None:
        seed = 100
    if type_of_theta_init=="circular":
        nu = np.random.RandomState(seed).uniform(-np.pi, np.pi)
        theta = np.array([(nu + i * 2 * np.pi/M)%(2 * np.pi) for i in range(M)]) - np.pi
    elif type_of_theta_init=="unstructured":
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, M) 
    S = gds(M, len(X), seed=seed+20) 
    noise_cov = initial_noise_covariance(X, theta, S)
    return theta, S, noise_cov


def A_ULA_torch(L, theta):
    """
    Создает матрицу управляющих векторов для массива сенсоров типа ULA (PyTorch)
    L - число сенсоров,
    theta - тензор углов прибытия (размер [n_angles])
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float()  # (L,1)
    return torch.exp(-2j * torch.pi * dist_ratio * sensor_indices * torch.sin(theta))  # (L, n_angles)


def cost_theta_torch(theta, X, S, Q_inv_sqrt):
    """
    theta - тензор углов прибытия (requires_grad=True)
    X, S, Q_inv_sqrt - тоже тензоры PyTorch, dtype=torch.cfloat или torch.float
    """
    A = A_ULA_torch(X.shape[0], theta)  # (L, n_angles)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор

def CM_step_theta_start(X_np, theta0_np, S_np, Q_inv_sqrt_np, method='SLSQP', tol=1e-6):
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

def CM_step_S(X: np.ndarray, A: np.ndarray, Q: np.ndarray):
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return (np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X).T


def CM_step_Q(X: np.ndarray, A: np.ndarray, S: np.ndarray):
    r = X.T - A @ S.T 
    Q = np.nanvar(r, axis=1, ddof=0)
    epsilon = 1e-6
    Q = np.diag(Q + epsilon)
    return Q


def incomplete_lkhd(X: np.ndarray, theta: np.ndarray, S: np.ndarray, Q: np.ndarray, inv_Q: np.ndarray):
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений и текущей оценки параметров.

    Параметры:
    X - наблюдения (np.ndarray)
    theta - оценки углов прибытия (np.ndarray)
    S - оценки исходных сигналов (np.ndarray)
    Q - матрица ковариации шума, либо ее оценка (np.ndarray)
    inv_Q - обратная к Q матрица (np.ndarray)

    Возвращает:
    res - значение неполного правдоподобия
    """
    A = A_ULA(X.shape[1], theta)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            A_o, Q_o = A[O_i, :], Q[np.ix_(O_i, O_i)]
            res += - np.log(np.linalg.det(Q_o)) - (X[i, O_i].T - A_o @ S[i].T).conj().T @ np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T)
        else:
            res += - np.log(np.linalg.det(Q)) - (X[i].T - A @ S[i].T).conj().T @ inv_Q @ (X[i].T - A @ S[i].T)
    return res.real


def ECM(theta: np.ndarray, S: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=20, rtol: float=1e-5):
    """
    Запускает ЕCМ-алгоритм из случайно выбранной точки.

    Параметры:
    theta: np.ndarray 
      Вектор углов, которые соответствуют DOA.
    S: np.ndarray 
      Вектор исходных сигналов.
    X: np.ndarray 
      Коллекция полученных сигналов.
    Q: np.ndarray
      Ковариация шума.
    max_iter: int
      Предельное число итераций.
    rtol: float
      Величина, используемая для проверки сходимости последних итераций.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    L = Q.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = robust_complex_cov(X[observed_rows, ])
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    X_modified = X.copy()
    ECM_Iteration = 0
    while ECM_Iteration < max_iter:
        A = A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                Q_o, Q_m = Q[np.ix_(O_i, O_i)], Q[np.ix_(M_i, M_i)]
                K_MO = K[np.ix_(M_i, O_i)]
                K_OM = K_MO.conj().T
                Mu_cond[i] = A_m @ S[i] + K_MO @ np.linalg.inv(Q_o) @ (X_modified[i, O_i] - A_o @ S[i])
                K_Xm_cond_accum[np.ix_(M_i, M_i)] = Q_m - K_MO @ np.linalg.inv(Q_o) @ K_OM
                X_modified[i, M_i] = Mu_cond[i]
        # Шаги условной максимизации
        K = robust_complex_cov(X_modified)
        new_theta = CM_step_theta(X_modified.T, theta, S.T, Q_inv_sqrt)
        #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        A = A_ULA(L, new_theta)
        new_S = CM_step_S(X_modified.T, A, Q)
        #print(f'diff of S is {np.sum((new_S-S)**2)} on iteration {EM_Iteration}')
        new_Q = CM_step_Q(X_modified, A, new_S)
        #print(f'diff of Q is {np.sum((new_Q-Q)**2)} on iteration {EM_Iteration}')
        lkhd = incomplete_lkhd(X_modified, new_theta, new_S, new_Q, np.linalg.inv(Q))
        #if np.linalg.norm(theta - new_theta) < rtol and np.linalg.norm(S - new_S, ord = 2) < rtol and np.linalg.norm(Q - new_Q, ord = 2) < rtol:
            #break
        theta, S, Q = new_theta, new_S, new_Q
        print(f'incomplete likelihood is {lkhd.real} on iteration {ECM_Iteration}')
        ECM_Iteration += 1
    return theta, S, Q, lkhd


def multi_start_ECM(X: np.ndarray, M: int, num_of_starts: int = 30, max_iter: int = 20, rtol: float = 1e-6):
    """
    Мультистарт для ЕCМ-алгоритма.

    Параметры:
    X: np.ndarray 
      Коллекция полученных сигналов.
    M: int
      Число источников.
    num_of_starts: int
      Число запусков.
    max_iter: int
      Предельное число итераций.
    rtol: float
      Величина, используемая для проверки сходимости последних итераций.

    Возвращает:
    best_theta: np.ndarray
      Оценка DoA.
    best_S: np.ndarray
      Оценка детерминированных исходных сигналов.
    best_Q: np.ndarray
      Оценка ковариационной матрицы шума.
    best_lhd: np.complex128
      Оценка неполного правдоподобия.
    """
    best_lhd, best_theta, best_S, best_Q, best_start = -np.inf, None, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, S, Q = initializer(X, M, seed=i * 100)
        est_theta, est_S, est_Q, est_lhd = ECM(theta, S, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_theta, best_S, best_Q, best_lhd, best_start = est_theta, est_S, est_Q, est_lhd, i
    best_theta = angle_correcter(best_theta)
    print(f'best_start={best_start}')
    return best_theta, best_S, best_Q, best_lhd





