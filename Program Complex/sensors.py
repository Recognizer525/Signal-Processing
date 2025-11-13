import numpy as np
from numpy.linalg import eig
from scipy.signal import find_peaks

DIST_RATIO = 0.5

def MCAR(X: np.ndarray,
         mis_cols: object,
         num_mv: object,
         rs: int = 42) -> np.ndarray:
    '''
    Реализует создание абсолютно случайных пропусков.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, представляет из себя выборку, 
        состоящую из наблюдений, каждому из них соответствует своя строка.
    mis_cols: object
        Целое число (int), либо список (list[int]). 
        Указывает на индексы столбцов, в которые следует добавить пропуски.
    num_mv: object 
        Целое число (int), либо список (list[int]). 
        Указывает на количество пропусков, которые следует добавить 
        в каждый столбец из числа указанных в mis_cols.
    rs: int
        Целое число (int), соответствует randomstate для выбора 
        конкретных позиций, где будут размещены пропуски.

    Возвращает:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, представляет собой выборку, 
        в которую добавлены абсолютно случайные пропуски.
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
    Генерирует детерминированные сигналы, 
    представляющие из себя комплексные нормальные вектора 
    с круговой симметрией.
    
    Параметры:
    ---------------------------------------------------------------------------
    M: int
        Число источников.
    G: int
        Число наблюдений.
    A: np.ndarray
        Одномерный массив размера M, 
        в котором указаны амплитуды сигналов от источников.
    f: np.ndarray
        Одномерный массив размера M, 
        в котором указаны частоты сигналов от источников.
    phi: np.ndarray
        Одномерный массив размера M, 
        в котором указаны фазы сигналов от источников.
    seed: int
        Randomstate для генерации амплитуд, частот и фаз, 
        если таковые не указаны пользователем.

    Возвращает:
    ---------------------------------------------------------------------------
    signals: np.ndarray
        Сгенерированные сигналы в формате двумерного массива размера (G,M).
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


def gss(K: int, G: int, Cov: np.ndarray, seed: int = None):
    """
    Генерирует детерминированные сигналы, 
    представляющие из себя комплексные нормальные вектора 
    с круговой симметрией.
    
    Параметры:
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    G: int
        Число наблюдений.
    Cov: np.ndarray
        Ковариационная матрица исходных сигналов.

    Возвращает:
    ---------------------------------------------------------------------------
    signals: np.ndarray
        Сгенерированные сигналы в формате двумерного массива размера (G,M).
    """
    if seed is None:
        seed = 70
    n = 2 * K # Размер ковариационной матрицы совместного распределения
    C = np.zeros((n,n), dtype=np.float64)
    C[:K,:K] = Cov.real
    C[K:,K:] = Cov.real
    C[:K,K:] = -Cov.imag
    C[K:,:K] = Cov.imag
    mu = np.zeros(n)
    B = np.random.RandomState(seed).multivariate_normal(mu, 0.5*C, G)
    signals = B[:,:K] + 1j * B[:, K:]
    return signals


def complex_cov(X: np.ndarray, ddof: int = 0):
    """
    Вычисляет оценку пространственной ковариационной матрицы.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив. Представляет собой коллекцию полученных сигналов. 
        Каждая строка соответствует одному вектору сигналов.
    ddof: int
        Поправка на число степеней свободы.
   
    Возвращает:
    ---------------------------------------------------------------------------
    cov: np.ndarray
        Двумерный массив. Представляет собой оценку ковариационной матрицы.
    """
    return np.einsum('ni,nj->ij', X, X.conj()) / (X.shape[0] - ddof)


def robust_complex_cov(X: np.ndarray, ddof: int = 0):
    """
    Вычисляет оценку пространственной ковариационной матрицы, таким образом, 
    чтобы обнулить мнимую часть диагональных элементов.

    Параметры:
    ---------------------------------------------------------------------------
    X: np.ndarray
        Коллекция полученных сигналов, представленная двумерным массивом. 
        Каждая строка соответствует одному вектору сигналов.
    ddof: int
        Поправка на число степеней свободы.

    Возвращает:
    ---------------------------------------------------------------------------
    K: np.ndarray
        Оценка ковариационной матрицы.
    """
    K = np.einsum('ni,nj->ij', X, X.conj()) / (X.shape[0] - ddof)
    K = (K + K.conj().T)/2
    return K


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


def A_ULA(L: int, theta:np.ndarray):
    """
    Создает матрицу векторов направленности для 
    равномерного линейного массива сенсоров (ULA).

    Параметры:
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    theta: np.ndarray
        Одномерный массив размера (M,1). Соответствует DoA.
    
    Возвращает:
    ---------------------------------------------------------------------------
    A: np.ndarray
        Двумерный массив размера (L,M). Матрица векторов направленности.
    """
    return np.exp(-2j * np.pi * DIST_RATIO * np.arange(L).reshape(-1,1) * np.sin(theta))


def MUSIC_DoA(R: np.ndarray, 
              num_sources: int, 
              scan_angles=np.arange(-90, 90.5, 0.5)):
    """
    Вычисляет оценку DoA через использование алгоритма MUSIC.

    Параметры:
    ---------------------------------------------------------------------------
    R: np.ndarray
        Пространственная ковариационная матрица.
    num_sources: int
        Число источников.
    scan_angles: np.ndarray
        Сетка углов, представлена одномерным массивом.

    Возвращает:
    ---------------------------------------------------------------------------
    doa_estimates: np.ndarray
        Оценка DoA, представлена одномерным массивом.
    """
    L = R.shape[0]
    eigvals, eigvecs = eig(R)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, :-num_sources] 
    P_music = []
    for theta in scan_angles:
        a = np.exp(-1j * 2 * np.pi * DIST_RATIO * np.arange(L) * np.sin(np.deg2rad(theta)))
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