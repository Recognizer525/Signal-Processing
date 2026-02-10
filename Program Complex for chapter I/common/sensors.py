import numpy as np

DIST_RATIO = 0.5

def MCAR(X: np.ndarray,
         mis_cols: int|np.ndarray, 
         share_mv: float|np.ndarray,
         rs: int = 42) -> np.ndarray:
    '''
    Реализует создание абсолютно случайных пропусков.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, представляет из себя выборку, 
        состоящую из наблюдений, каждому из них соответствует своя строка.
    mis_cols: int|np.ndarray
        Целое число (int), либо np.ndarray. 
        Указывает на индексы столбцов, в которые следует добавить пропуски.
    share_mv: int|np.ndarray
        Вещественное число (float), либо np.ndarray. 
        Указывает на долю пропусков, которые следует добавить 
        в каждый столбец из числа указанных в mis_cols.
    rs: int
        Randomstate для выбора конкретных позиций, где будут размещены пропуски.

    Returns
    ---------------------------------------------------------------------------
    X1: np.ndarray
        Двумерный массив, представляет собой выборку, 
        в которую добавлены абсолютно случайные пропуски.
    '''
    #print(f"share_mv={share_mv}")
    #print(f"mis_cols={mis_cols}")

  
    # Проверяем длины списков
    assert len(mis_cols) == len(share_mv), \
    "mis_cols и num_mv должны быть одной длины"

    # Проверяем X
    assert isinstance(X, np.ndarray) and X.ndim == 2, \
    "X должен быть двумерным numpy-массивом"

    # Проверяем индексы столбцов
    n_cols = X.shape[1]
    assert all(isinstance(c, (int, np.int16, np.int32, np.int64)) for c in mis_cols), \
    "mis_cols должен содержать целые индексы"
    assert all(0 <= c < n_cols for c in mis_cols), \
    "Индекс столбца вне диапазона"

    # Проверяем число пропусков
    n_rows = X.shape[0]
    assert all(isinstance(n, np.float32) for n in share_mv), \
        "share_mv должен содержать вещественные числа"
    assert all(0 <= n <= 1 for n in share_mv), \
        "share_mv не может превышать 1"

    X1 = X.copy()
    n_rows = X.shape[0]
    for i, col in enumerate(mis_cols):
        rng = np.random.RandomState(rs + i)
        rows = rng.choice(n_rows, size=int(np.floor(share_mv[i]*n_rows)), replace=False)
        X1[:,mis_cols[i]][rows] = np.nan
    return X1


def signal_variance(signal_pressure: float, distance: float) -> float:
    """
    На основе давления сигнала и расстояния определяет ковариацию сигнала.

    Parameters
    ---------------------------------------------------------------------------
    signal_pressure: float
        Приведенное давление сигнала.
    distance: float
        Расстояние между источником сигнала и центром антенны.
    """
    return (signal_pressure / distance) ** 2


def noise_variance(noise_pressure: float) -> float:
    """
    На основе давления шума и расстояния определяет ковариацию шума.

    Parameters
    ---------------------------------------------------------------------------
    noise_pressure: float
        Приведенное давление шума.
    """
    return noise_pressure ** 2


def gss(K: int, 
        T: int, 
        Cov: np.ndarray, 
        rs: int|None = None) -> np.ndarray:
    """
    Генерирует детерминированные сигналы, 
    представляющие из себя комплексные нормальные вектора 
    с круговой симметрией.
    
    Parameters
    ---------------------------------------------------------------------------
    K: int
        Число источников.
    T: int
        Число наблюдений.
    Cov: np.ndarray
        Ковариационная матрица исходных сигналов.
    rs: int|None
        RandomState.

    Returns
    ---------------------------------------------------------------------------
    signals: np.ndarray
        Сгенерированные сигналы в формате двумерного массива размера (T,K).
    """
    if rs is None:
        rs = 70
    n = 2 * K # Длина столбца ковариационной матрицы
    C = np.zeros((n,n), dtype=np.float64)
    C[:K,:K] = Cov.real
    C[K:,K:] = Cov.real
    C[:K,K:] = -Cov.imag
    C[K:,:K] = Cov.imag
    mu = np.zeros(n)
    B = np.random.RandomState(rs).multivariate_normal(mu, 0.5*C, T)
    signals = B[:,:K] + 1j * B[:, K:]
    return signals


def complex_cov(X: np.ndarray, 
                ddof: int = 0, 
                reg: bool = True) -> np.ndarray:
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
    reg: bool
        Проводить ли преобразование (X+X^H)/2 для коррекции численных ошибок.

    Returns
    ---------------------------------------------------------------------------
    Cov: np.ndarray
        Оценка ковариационной матрицы.
    """
    Cov = np.einsum('ni,nj->ij', X, X.conj()) / (X.shape[0] - ddof)
    if reg:
        Cov = (Cov + Cov.conj().T)/2
    return Cov


def initial_Cov(X: np.ndarray):
    """
    Начальная оценка матрицы ковариации для набора наблюдений с пропусками.
    
    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Выборка. Число строк - число наблюдений, 
        число столбцов - число компонентов наблюдения.

    Returns
    ---------------------------------------------------------------------------
    R: ndarray
        Начальная оценка ковариации.
    """
    # Строки без пропусков
    observed_rows = np.where(~np.isnan(X).any(axis=1))[0]

    # если достаточно полных наблюдений
    if len(observed_rows) >= X.shape[1]:
        R = complex_cov(X[observed_rows, :])
    else:
        # Заполнение средним (mean imputation) по столбцам
        X_filled = X.copy()
        col_means = np.nanmean(X_filled, axis=0)
        inds = np.where(np.isnan(X_filled))
        X_filled[inds] = np.take(col_means, inds[1])
        R = complex_cov(X_filled)

    R += 1e-6 * np.eye(R.shape[0])
    return R


def angle_correcter(theta: np.ndarray, is_ULA: bool = True) -> np.ndarray:
    """
    Приводит углы к диапазону [-pi/2, pi/2], сохраняя то же значение синуса.

    Parameters
    ---------------------------------------------------
    theta: int
        Исходный вектор углов.
    is_ULA: bool
        Является ли антенная решетка равномерной линейной, если да,
        требуется приведение углов к [-pi/2; pi/2].
    
    Returns
    ---------------------------------------------------
    new_theta: np.ndarray
        Скорректированный вектор углов.
    """
    # Приведение к диапазону [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    if is_ULA:
        # Отражение значений, выходящих за пределы [-pi/2, pi/2]
        mask = theta > np.pi/2
        theta[mask] = np.pi - theta[mask]
        mask = theta < -np.pi/2
        theta[mask] = -np.pi - theta[mask]
    return theta


def is_diagonal(A: np.ndarray) -> bool:
    """
    Проверяет свойство диагональности для матрицы A.
    """
    return np.all(A == np.diag(np.diagonal(A)))


def is_pd(A: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Проверяет, является ли матрица A эрмитовой 
    и положительно определенной.

    Parameters
    ---------------------------------------------------
    A: np.ndarray
        Матрица, подлежащая проверке.
    tol: float
        Допустимая погрешность (когда речь идет о сравнении A и A.conj().T).
    
    Returns
    ans: bool
        Ответ (True/False).
    ---------------------------------------------------
    """
    # Проверка эрмитовости
    if not np.allclose(A, A.conj().T, atol=tol):
        return False
    # Проверка положительной определенности.
    evals = np.linalg.eigvalsh(A)
    res = evals.min() > 0
    #print(f'min={evals.min()}')
    return res


def cov_correcter(A: np.ndarray, 
                  reg_coef: float = 1e-3) -> np.ndarray:
    """
    Реализует операцию (A+A^H)/2 + reg_coef * E 
    для повышения численной стабильности результатов.

    A: np.ndarray
        Матрица, которую следует откорректировать.
    reg_coef: float
        Коэффициент регуляризации.
    """
    if A.ndim == 2:
        return 0.5 * (A + A.conj().T) + reg_coef * np.eye(A.shape[0])
    if A.ndim == 3:
        return 0.5 * (A + np.conj(np.transpose(A, axes=(0,2,1)))) \
            + reg_coef * np.eye(A.shape[1])
    

def SNR(A: np.ndarray, 
        P: np.ndarray, 
        Q: np.ndarray, 
        metrics: str = 'avg', 
        scale: str = 'linear') -> np.float64:
    """
    Вычисляет отношение сигнал-шум для всей антенной решетки.

    Parameters
    ---------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности.
    P: np.ndarray
        Ковариация сигналов.
    Q: np.ndarray
        Ковариация шума.
    metrics: str
        Метрика для агрегации SNR по сигналам (допускаются суммирование и усреднение).
    scale: str
        Тип шкалы для оценки SNR (допускаются линейная и логарифмическая).
    
    Returns
    ---------------------------------------------------
    ans: np.float64
        Отношение сигнал-шум (в линейной или логарифмической шкале).
    """
    T = A @ P @ A.conj().T
    ans = 0.0
    for i in range(A.shape[1]):
        a = A[:, i]
        ans += np.vdot(a, T @ a) / np.vdot(a, Q @ a)
    if metrics == 'avg':
        ans = ans / P.shape[0]
    elif metrics == 'total':
        pass
    else:
        raise ValueError(f"Указан неизвестный тип метрики {metrics}")
    
    ans = ans.real.astype(np.float64)
    
    if scale == 'log':
        return 10 * np.log10(ans)
    elif scale == 'linear':
        return ans
    else:
        raise ValueError(f"Указан неизвестный тип шкалы {scale}")
    

def db_to_var(dB_magnitude: float) -> float:
    """
    Сопоставляет децибелам (которые характеризуют мощность сигнала/шума)
    дисперсию.
    """
    return 10 ** (dB_magnitude / 10)

