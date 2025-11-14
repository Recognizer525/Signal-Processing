import numpy as np
from numpy.linalg import eig
from scipy.signal import find_peaks

DIST_RATIO = 0.5

def MCAR(X: np.ndarray,
         mis_cols: int|list,
         num_mv: int|list,
         rs: int = 42) -> np.ndarray:
    '''
    Реализует создание абсолютно случайных пропусков.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, представляет из себя выборку, 
        состоящую из наблюдений, каждому из них соответствует своя строка.
    mis_cols: int|list
        Целое число (int), либо список (list[int]). 
        Указывает на индексы столбцов, в которые следует добавить пропуски.
    num_mv: int|list
        Целое число (int), либо список (list[int]). 
        Указывает на количество пропусков, которые следует добавить 
        в каждый столбец из числа указанных в mis_cols.
    rs: int
        Randomstate для выбора 
        конкретных позиций, где будут размещены пропуски.

    Returns
    ---------------------------------------------------------------------------
    X1: np.ndarray
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


def gss(K: int, G: int, Cov: np.ndarray, seed: int = None) -> np.ndarray:
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
    Cov: np.ndarray
        Ковариационная матрица исходных сигналов.

    Returns
    ---------------------------------------------------------------------------
    signals: np.ndarray
        Сгенерированные сигналы в формате двумерного массива размера (G,K).
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


def complex_cov(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    Вычисляет оценку пространственной ковариационной матрицы.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Выборка, состоящая из реализаций комплексных случайных векторов.
        Предполагается, что математическое ожидание равно нулю, 
        расположенных построчно.
    ddof: int
        Поправка на число степеней свободы.
   
    Returns
    ---------------------------------------------------------------------------
    cov: np.ndarray
        Оценка ковариационной матрицы.
    """
    return np.einsum('ni,nj->ij', X, X.conj()) / (X.shape[0] - ddof)


def robust_complex_cov(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """
    Вычисляет оценку пространственной ковариационной матрицы, таким образом, 
    чтобы обнулить мнимую часть диагональных элементов, которая возникает из-за
    погрешностей вычислений.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Выборка, состоящая из реализаций комплексных случайных векторов.
        Предполагается, что математическое ожидание равно нулю, 
        расположенных построчно.
    ddof: int
        Поправка на число степеней свободы.

    Returns
    ---------------------------------------------------------------------------
    Cov: np.ndarray
        Оценка ковариационной матрицы.
    """
    Cov = np.einsum('ni,nj->ij', X, X.conj()) / (X.shape[0] - ddof)
    Cov = (Cov + Cov.conj().T)/2
    return Cov


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


def A_ULA(L: int, theta:np.ndarray) -> np.ndarray:
    """
    Создает матрицу векторов направленности для 
    равномерного линейного массива сенсоров (ULA).

    Parameters
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    theta: np.ndarray
        Оценка DoA, представлена в форме одномерного массива размера (K,1).
    
    Returns
    ---------------------------------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности. 
        Представлена в форме двумерного массива размера (L,K).
    """
    return (np.exp(-2j * np.pi * DIST_RATIO * 
                   np.arange(L).reshape(-1,1) * np.sin(theta)))


def random_complex_cov(n: int, max_real: float, seed: int|None = None):
    """
    Создаёт случайную эрмитову неотрицательно определённую матрицу размера n×n.
    
    Parameters
    ---------------------------------------------------
    n: int
        Размер матрицы.
    max_real: float
        Определяет максимальное значение действительной части
        элементов создаваемой матрицы.
    seed: int|None
        Фиксирует генератор случайных чисел.
    
    Returns
    ---------------------------------------------------
    C: np.ndarray
        Эрмитова, неотрицательно определенная матрица.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        seed = 42

    A = (np.random.RandomState(seed).randn(n, n) 
         + 1j * np.random.RandomState(seed+1).randn(n, n))
    C = A @ A.conj().T
    C /= np.trace(C).real

    if max_real is not None:
        C *= max_real
    return C


def MUSIC_DoA(R: np.ndarray, 
              num_sources: int, 
              scan_angles=np.arange(-90, 90.5, 0.5)) -> np.ndarray:
    """
    Вычисляет оценку DoA через использование алгоритма MUSIC.

    Parameters
    ---------------------------------------------------------------------------
    R: np.ndarray
        Пространственная ковариационная матрица.
    num_sources: int
        Число источников.
    scan_angles: np.ndarray
        Сетка углов, представлена одномерным массивом.

    Returns
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
        a = (np.exp(-1j * 2 * np.pi * DIST_RATIO * 
                    np.arange(L) * np.sin(np.deg2rad(theta))))
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


def is_diagonal(A: np.ndarray) -> bool:
    """
    Проверяет свойство диагональности для матрицы A.
    """
    return np.all(A == np.diag(np.diagonal(A)))


def is_spd(A: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Проверяет, что матрица A симметрична и положительно определена.
    """
    # Проверим эрмитовость
    if not np.allclose(A, A.conj().T, atol=tol):
        return False
    # Проверим положительную определённость
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        print('Not positive semi-definite', 'det=', np.linalg.det(A))
        return False