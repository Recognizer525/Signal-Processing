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


def gds(K: int, G: int,
        A: np.ndarray = None, f: np.ndarray = None, phi: np.ndarray = None, 
        seed: int = None) -> np.ndarray:
    """
    Генерирует детерминированные сигналы, 
    представляющие из себя комплексные нормальные вектора 
    с круговой симметрией.
    
    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    G: int
        Число наблюдений.
    A: np.ndarray
        Одномерный массив размера K, 
        в котором указаны амплитуды сигналов от источников.
    f: np.ndarray
        Одномерный массив размера K, 
        в котором указаны частоты сигналов от источников.
    phi: np.ndarray
        Одномерный массив размера K, 
        в котором указаны фазы сигналов от источников.
    seed: int
        Randomstate для генерации амплитуд, частот и фаз, 
        если таковые не указаны пользователем.

    Returns
    ---------------------------------------------------------------------------
    signals: np.ndarray
        Сгенерированные сигналы в формате двумерного массива размера (G,K).
    """ 
    if seed is None:
        seed = 10
    # G - размер выборки, K - число источников
    if A is None:
        A = np.random.RandomState(seed + 40).uniform(0.5, 1.5, K)         
    if f is None:
        f = np.random.RandomState(seed + 10).uniform(0.01, 0.1, K)        
    if phi is None:
        phi = np.random.RandomState(seed + 1).uniform(0, 2*np.pi, K)
    
    g = np.arange(G)
    signals = np.zeros((K, G), dtype=complex)
    for k in range(K):
        signals[k] = A[k] * np.exp(1j * (2 * np.pi * f[k] * g + phi[k]))
    signals = signals.T
    return signals


def init_est(X: np.ndarray, K: int, seed: int = None):
    if seed is None:
        seed = 100 
    S = gds(K, len(X), seed=seed+20)
    return S


def incomplete_lkhd(X: np.ndarray, 
                    theta: np.ndarray, 
                    S: np.ndarray, 
                    Q: np.ndarray, 
                    inv_Q: np.ndarray) -> np.float64:
    """
    Вычисляет неполное правдоподобие на основании доступных наблюдений 
    и текущей оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    theta: np.ndarray
        Одномерный массив размера (K,1). Соответствует оценке DoA.
    S: np.ndarray
        Двумерный массив, соответствующий оценке 
        последовательности исходных сигналов.
    Q: np.ndarray
        Ковариационная матрица шума.
    inv_Q: np.ndarray
        Матрица, обратная к Q.

    Returns
    ---------------------------------------------------------------------------
    res: np.float64
        Значение неполного правдоподобия.
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
            #print(f"log is {np.log(np.linalg.det(Q_o))}")
            res += (- np.log(np.linalg.det(Q_o)) - 
                    (X[i, O_i].T - A_o @ S[i].T).conj().T @ 
                    np.linalg.inv(Q_o) @ (X[i, O_i].T - A_o @ S[i].T))
        else:
            res += (- np.log(np.linalg.det(Q)) - 
                    (X[i].T - A @ S[i].T).conj().T @ 
                    inv_Q @ (X[i].T - A @ S[i].T))
    return res.real


def CM_step_S(X: np.ndarray, 
              A: np.ndarray, 
              Q: np.ndarray) -> np.ndarray:
    """
    Осуществляет условную максимизацию по исходным сигналам.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям 
        (с учетом оценок пропущенных значений).
    A: np.ndarray
        Двумерный массив, соответствующий матрице векторов направленности.
    Q: np.ndarray
        Ковариационная матрица шума.
    
    Returns
    ---------------------------------------------------------------------------
    S: np.ndarray
        Новая оценка детерминированных исходных сигналов.
    """
    inv_Q = np.linalg.inv(Q)
    A_H = A.conj().T
    return (np.linalg.inv(A_H @ inv_Q @ A) @ A_H @ inv_Q @ X).T


def ECM_kn(theta: np.ndarray, 
           S: np.ndarray, 
           X: np.ndarray, 
           Q: np.ndarray, 
           max_iter: int=50, 
           rtol: float=1e-6,
           method: str = 'L-BFGS-B') -> tuple[np.ndarray,
                                      np.ndarray,
                                      np.float64]:
    """
    Запускает ЕCМ-алгоритм для выбранной начальной оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    theta: np.ndarray
        Одномерный массив размера (K,1). Соответствует оценке DoA.
    S: np.ndarray
        Двумерный массив, соответствующий оценке 
        последовательности исходных сигналов.
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    Q: np.ndarray
        Ковариационная матрица шума.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.
    method: str
        Метод оптимизации функции потерь для DoA.
    
    Returns
    ---------------------------------------------------------------------------
    lkhd: np.float64
        Неполное правдоподобие для полученных параметров.
    """
    L = Q.shape[0]

    print(f'Initial theta = {theta}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    Mu_cond = {}
    X_modified = X.copy()
    ECM_Iteration = 0
    while ECM_Iteration < max_iter:
        A = A_ULA(L, theta)
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                A_o, A_m = A[O_i, :], A[M_i, :]
                K_OO = Q[np.ix_(O_i, O_i)]
                K_MO = Q[np.ix_(M_i, O_i)]
                Mu_cond[i] = (A_m @ S[i] + K_MO @ np.linalg.inv(K_OO) @ 
                              (X_modified[i, O_i] - A_o @ S[i]))
                X_modified[i, M_i] = Mu_cond[i]


        new_S = CM_step_S(X_modified.T, A, Q)
        lkhd = incomplete_lkhd(X, theta, new_S, 
                               Q, np.linalg.inv(Q))

        S = new_S
        print(f'incomplete likelihood is {lkhd} on iteration {ECM_Iteration}')

        ECM_Iteration += 1
    return lkhd


def multi_start_ECM(theta: np.ndarray,
                    X: np.ndarray, 
                    K: int, 
                    Q: np.ndarray, 
                    max_iter: int = 20, 
                    rtol: float = 1e-6) -> tuple[np.ndarray, 
                                                 np.ndarray,
                                                 np.float64]:
    """
    Реализует мультистарт для ЕCМ-алгоритма.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    K: int
        Число источников.
    Q: np.ndarray
        Двумерный массив, соответствующий ковариационной матрице шума.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol: float
        Величина, используемая для проверки сходимости итерационного процесса.

    Returns
    ---------------------------------------------------------------------------
    best_theta: np.ndarray
        Оценка DoA.
    best_S: np.ndarray
        Оценка детерминированных исходных сигналов.
    best_lhd: np.float64
        Неполное правдоподобие для полученных параметров.
    """
 

    S = init_est(X, K, seed=100)
    est_lhd = ECM_kn(theta, S, X, Q, max_iter, rtol)
    return est_lhd


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