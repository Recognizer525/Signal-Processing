import numpy as np

DIST_RATIO = 0.5

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


def normalization(y: object):
    return (y-min(y))/(max(y)-min(y))


def A_ULA(L, theta):
    """
    Создает матрицу управляющих векторов для массива сенсоров типа ULA
    """
    return np.exp(-2j * np.pi * DIST_RATIO * np.arange(L).reshape(-1,1) * np.sin(theta))


def initializer(X: np.ndarray, M: int, seed: int = None, type_of_theta_init="circular"):
    if seed is None:
        seed = 100 
    P_diag = np.random.RandomState(seed).uniform(0.2, 5, M)
    #P_diag = np.ones(M)
    return np.diag(P_diag)


def M_step(mu, sigma):
    """
    mu - массив, составленный из векторов УМО исходного сигнала, в зависимости от наблюдений. Число столбцов соответствует числу наблюдений.
    sigma - условная ковариация исходного сигнала с учетом наблюдения.
    """
    G = mu.shape[1]
    res = (1/G) * mu @ mu.conj().T + sigma
    # Оставляем только диагональные элементы
    res = res * np.eye(res.shape[0], res.shape[1], dtype=np.complex128)
    return res


def incomplete_lkhd(X, theta, P, Q):
    A = A_ULA(X.shape[1], theta)
    R = A @ P @ A.conj().T + Q
    R = R + 1e-6 * np.eye(R.shape[0])
    inv_R = np.linalg.inv(R)
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    O = col_numbers * (Indicator == False) - 1
    res = 0
    for i in range(X.shape[0]):
        if set(O[i, ]) != set(col_numbers - 1):
            O_i = O[i, ][O[i, ] > -1]
            R_o = R[np.ix_(O_i, O_i)]
            R_o = R_o + 1e-6 * np.eye(R_o.shape[0])
            res += - np.log(np.linalg.det(R_o)) - (X[i, O_i].T).conj().T @ np.linalg.inv(R_o) @ (X[i, O_i].T)
        else:
            res += - np.log(np.linalg.det(R)) - (X[i].T).conj().T @ inv_R @ (X[i].T)
    return res.real


def EM(theta: np.ndarray, P: np.ndarray, X: np.ndarray, Q: np.ndarray, max_iter: int=50, eps: float=1e-6):
    """
    Запуск ЕМ-алгоритма из случайно выбранной точки.
    theta - вектор углов, которые соответствуют DOA;
    P - ковариация исходных сигналов;
    X - коллекция полученных сигналов;
    Q - ковариация шума;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    L = Q.shape[0]
    G = X.shape[0]
    A = A_ULA(L, theta)

    print(f'Initial theta = {theta}')
    # Для каждого наблюдения определяем группы индексов, соответствующие наблюдаемым и пропущенным данным
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1

    # Строим первоначальную оценку ковариации наблюдений
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    R = complex_cov(X[observed_rows, ])
    if np.isnan(R).any():
        R = np.diag(np.nanvar(X, axis = 0))
        print('Special estimate of K')

    Mu_Xm_cond = {}
    K_Xm_cond_accum = np.zeros((L,L), dtype=np.complex128)
    Mu_S_cond = np.zeros((L, G), dtype=np.complex128)
    K_S_cond = np.zeros(P.shape, dtype=np.complex128)
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                # Вычисляем блоки ковариации принятых сигналов (наблюдений)
                R_OO = R[np.ix_(O_i, O_i)]
                R_OO = R_OO + 1e-6 * np.eye(R_OO.shape[0])
                R_MO = R[np.ix_(M_i, O_i)]
                R_MM = R[np.ix_(M_i, M_i)]
                # Оцениваем параметры апостериорного распределения ненаблюдаемых данных и пропущенные значения
                Mu_Xm_cond[i] = R_MO @ np.linalg.inv(R_OO) @ X_modified[i, O_i]
                X_modified[i, M_i] = Mu_Xm_cond[i]
                K_Xm_cond_accum[np.ix_(M_i, M_i)] += R_MM - R_MO @ np.linalg.inv(R_OO) @ R_MO.conj().T
        # Вычисляем блоки совместной ковариации исходных и принятых сигналов
        K_XX = A @ P @ A.conj().T + Q + 1e-6 * np.eye(Q.shape[0])
        K_SS = P
        K_XS = A @ P
        K_SX = K_XS.conj().T
        Mu_S_cond = K_SX @ np.linalg.inv(K_XX) @ X_modified.T
        K_S_cond = K_SS - K_SX @ np.linalg.inv(K_XX) @ K_XS

        # Шаг максимизации
        R = complex_cov(X_modified) + K_Xm_cond_accum / G
        new_P = M_step(Mu_S_cond, K_S_cond)
        #print(f'diff of P is {np.sum((new_P-P)**2)} on iteration {EM_Iteration}')
        P = new_P
        lkhd = incomplete_lkhd(X_modified, theta, P, Q)
        print(f'likelihood is {lkhd} on iteration {EM_Iteration}')

        EM_Iteration += 1
    return P, lkhd


def multi_start_EM(theta: np.ndarray, X: np.ndarray, M: int, Q: np.ndarray, num_of_starts: int = 1, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    theta - угол;
    X - коллекция полученных сигналов;
    M - число источников;
    Q - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_lhd, best_P, best_start = -np.inf, None, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        P = initializer(X, M, seed=i * 100)
        est_P, est_lhd = EM(theta, P, X, Q, max_iter, eps)
        if est_lhd > best_lhd:
            best_lhd, best_P, best_start = est_lhd, est_P, i
    #print(f"best_start={best_start}")
    return best_P, best_lhd


def steering_ula(theta, M):
    """Steering vector для ULA: M датчиков, шаг d, длина волны lam."""
    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * 0.5 * m * np.sin(theta))


def ESPRIT_spectrum(X, K, left_bound, right_bound, num_points):
    """
    Строит пространственный спектр P(theta) от -pi/2 до pi/2.
    """
    G, M = X.shape
    X = X.T

    # Ковариационная матрица
    R = X @ X.conj().T / G

    # EVD
    eigvals, eigvecs = np.linalg.eig(R)
    idx = np.argsort(-eigvals.real)

    Us = eigvecs[:, idx[:K]]          # подпространство сигналов
    Un = eigvecs[:, idx[K:]]          # подпространство шума

    # Угловая сетка
    thetas = np.linspace(left_bound, right_bound, num_points)

    P = np.zeros(num_points)

    for i, th in enumerate(thetas):
        a = steering_ula(th, M)
        P[i] = 1.0 / np.abs(a.conj().T @ Un @ Un.conj().T @ a)

    return thetas, P