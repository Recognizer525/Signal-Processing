import numpy as np
import scipy
import torch
from functools import partial
from scipy.optimize import minimize
from numpy.linalg import eig
from scipy.signal import find_peaks

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


def A_ULA_torch(L, theta):
    """
    Создает матрицу управляющих векторов для массива сенсоров типа ULA, аналогичен A_ULA, но приспособлен для Torch.
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float()  # (L,1)
    return torch.exp(-2j * np.pi * dist_ratio * sensor_indices * torch.sin(theta))  # (L, n_angles)


def cost_theta_torch(theta, X, S, Q_inv_sqrt):
    """
    Целевая функция, которую следует минимизировать.

    theta - тензор углов прибытия (requires_grad=True)
    X, S, Q_inv_sqrt - тоже тензоры PyTorch, dtype=torch.cfloat или torch.float
    """
    A = A_ULA_torch(X.shape[0], theta)  # (L, n_angles)
    E = torch.matmul(Q_inv_sqrt, X - torch.matmul(A, S))  
    return torch.norm(E, 'fro')**2  # скалярный тензор


def CM_step_theta_start(X_np, theta0_np, S_np, Q_inv_sqrt_np, method='SLSQP', tol=1e-6):
    """
    Шаг условной максимизации по DoA, старт из выбранной начальной оценки.

    X_np, theta0_np, S_np, Q_inv_sqrt_np - numpy массивы
    """
    
    # Объявляем функцию для scipy, которая принимает numpy theta, внутри переводим в torch и вычисляем
    def fun(theta_np):
        theta_t = torch.tensor(theta_np, dtype=torch.float32, requires_grad=True)
        if theta_t.grad is not None:
            theta_t.grad.zero_()
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


def CM_step_theta(X_np, theta0_np, S_np, Q_inv_sqrt_np, num_of_starts=20):
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


def incomplete_lkhd(X, theta, P, Q):
    A = A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q
    R = 0.5 * (R + R.conj().T) + 1e-6 * np.eye(R.shape[0])
    #print(f"is_spd(R)={is_spd(R)}")
    #print(f"is_spd(P)={is_spd(P)}")
    #print(f"is_spd(Q)={is_spd(Q)}")
    #print(f"Positive P? Ans is {np.all(np.diag(P) >= 0)}")
    inv_R = np.linalg.inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            R_o = R[np.ix_(O_i, O_i)]
            res += - np.log(np.linalg.det(R_o)) - (X[i, O_i].T).conj().T @ np.linalg.inv(R_o) @ (X[i, O_i].T)
        else:
            res += - np.log(np.linalg.det(R)) - (X[i].T).conj().T @ inv_R @ (X[i].T)
    return res.real


def ECM(theta: np.ndarray, P: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, rtol: float=1e-6):
    """
    Запуск ЕCМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    P - ковариация исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    rtol - величина, используемая для проверки сходимости последних итераций.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)
    
    L = Q.shape[0]
    G = X.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = robust_complex_cov(X[observed_rows, ])
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')
    Mu_Xm_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    Mu_S_cond = np.zeros((L, G), dtype=np.complex128)
    K_S_cond = np.zeros(P.shape, dtype=np.complex128)
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        A = A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                # Вычисляем блоки ковариации принятых сигналов (наблюдений)
                K_OO = K[np.ix_(O_i, O_i)]
                K_OO = K_OO + 1e-6 * np.eye(K_OO.shape[0])
                K_MO = K[np.ix_(M_i, O_i)]
                K_MM = K[np.ix_(M_i, M_i)]
                # Оцениваем параметры апостериорного распределения ненаблюдаемых данных и пропущенные значения
                Mu_Xm_cond[i] = K_MO @ np.linalg.inv(K_OO) @ X_modified[i, O_i]
                X_modified[i, M_i] = Mu_Xm_cond[i]
                K_Xm_cond_accum[np.ix_(M_i, M_i)] += K_MM - K_MO @ np.linalg.inv(K_OO) @ K_MO.conj().T
        # Вычисляем блоки совместной ковариации исходных и принятых сигналов
        K_XX = A @ P @ A.conj().T + Q
        K_XX = 0.5 * (K_XX + K_XX.conj().T) + 1e-6 * np.eye(Q.shape[0])
        K_SS = P
        K_XS = A @ P
        K_SX = K_XS.conj().T
        Mu_S_cond = K_SX @ np.linalg.inv(K_XX) @ X_modified.T
        K_S_cond = K_SS - K_SX @ np.linalg.inv(K_XX) @ K_XS

        # Шаги условной максимизации
        K = robust_complex_cov(X_modified) + K_Xm_cond_accum / G
        new_theta = CM_step_theta(X_modified.T, theta, Mu_S_cond, Q_inv_sqrt)
        #if EM_Iteration in [0, 1, 5, 11, 16, 21, 26]:
            #print(f'diff of theta is {new_theta-theta} on iteration {EM_Iteration}')
        A = A_ULA(L, new_theta)
        new_P = CM_step_P(Mu_S_cond, K_S_cond)
        #print(f'diff of P is {np.sum((new_P-P)**2)} on iteration {EM_Iteration}')
        theta, P = new_theta, new_P
        lkhd = incomplete_lkhd(X_modified, theta, P, Q)
        if EM_Iteration in range(20):
            print(f'likelihood is {lkhd.real} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return theta, P, lkhd


def multi_start_ECM(X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, rtol: float = 1e-6):
    """
    Мультистарт для ЕCМ-алгоритма.

    Параметры:
    X: np.ndarray 
      Коллекция полученных сигналов.
    M: int
      Число источников.
    Q: np.ndarray
      Ковариация шума.
    num_of_starts: int
      Число запусков.
    max_iter: int
      Предельное число итераций.
    rtol: float
      Величина, используемая для проверки сходимости последних итераций.

    Возвращает:
    best_theta: np.ndarray
      Оценка DoA.
    best_P: np.ndarray
      Оценка ковариационной матрицы исходных сигналов.
    best_lhd: np.complex128
      Оценка неполного правдоподобия.
    """
    best_lhd, best_theta, best_P, best_start = -np.inf, None, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        theta, P = initializer(X, M, seed=i * 100)
        est_theta, est_P, est_lhd = ECM(theta, P, X, Q, max_iter, rtol)
        if est_lhd > best_lhd:
            best_lhd, best_P, best_theta, best_start = est_lhd, est_P, est_theta, i
    best_theta = angle_correcter(best_theta)
    print(f"best_start={best_start}")
    return best_theta, best_P, best_lhd
