import numpy as np
from numpy.linalg import eig, pinv
from scipy.signal import find_peaks

from . import sensors as sn

DIST_RATIO = 0.5


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



def ESPRIT_DoA(R: np.ndarray, num_sources: int) -> np.ndarray:
    """
    Вычисляет оценку DoA через использование алгоритма ESPRIT.

    Parameters
    ---------------------------------------------------------------------------
    R: np.ndarray
        Пространственная ковариационная матрица (LxL).
    num_sources: int
        Число источников.

    Returns
    ---------------------------------------------------------------------------
    doa_estimates: np.ndarray
        Оценка DoA в радианах.
    """
    L = R.shape[0]
    
    # Собственные значения и векторы
    eigvals, eigvecs = eig(R)
    idx = np.argsort(eigvals)[::-1]  # сортировка по убыванию
    eigvecs = eigvecs[:, idx]
    
    # Выделяем сигнальное подпространство
    Es = eigvecs[:, :num_sources]

    # Разделяем на верхнюю и нижнюю подматрицы (сдвиг на 1)
    Es1 = Es[:-1, :]
    Es2 = Es[1:, :]

    # Находим матрицу Phi через псевдообратную
    Phi = pinv(Es1) @ Es2

    # Считаем собственные значения Phi
    phi_eigvals, _ = eig(Phi)

    # Углы прихода
    doa_estimates = np.arcsin(np.angle(phi_eigvals) / (2 * np.pi * DIST_RATIO))
    
    return doa_estimates


def trunc_MUSIC(data: np.ndarray, num_sources: int) -> np.ndarray:
    """
    Из наблюдений исключаются столбцы с пропусками,
    по ковариации оставшихся данных реализуется MUSIC.

    Parameters
    ---------------------------------------------------------------------------
    data: np.ndarray
        Данные, по которым оценивается ковариация.
    num_sources: int
        Число источников.

    Returns
    ---------------------------------------------------------------------------
    DoA: np.ndarray
        Оценки угловых координат.
    """
    mask1 = ~np.isnan(data).any(axis=0)
    truncated_data = data[:, mask1]
    R = sn.complex_cov(truncated_data)
    DoA = MUSIC_DoA(R, num_sources)
    return DoA


def trunc_ESPRIT(data: np.ndarray, num_sources: int) -> np.ndarray:
    """
    Из наблюдений исключаются столбцы с пропусками,
    по ковариации оставшихся данных реализуется ESPRIT.

    Parameters
    ---------------------------------------------------------------------------
    data: np.ndarray
        Данные, по которым оценивается ковариация.
    num_sources: int
        Число источников.

    Returns
    ---------------------------------------------------------------------------
    DoA: np.ndarray
        Оценки угловых координат.
    """
    mask1 = ~np.isnan(data).any(axis=0)
    truncated_data = data[:, mask1]
    R = sn.complex_cov(truncated_data)
    res = ESPRIT_DoA(R, num_sources)
    return res


def mean_imput_MUSIC(data: np.ndarray, num_sources: int) -> np.ndarray:
    """
    Применяется mean imputation к столбцам с пропусками,
    оценивается ковариация полученного набора,
    по ковариации оставшихся данных реализуется MUSIC.

    Parameters
    ---------------------------------------------------------------------------
    data: np.ndarray
        Данные, по которым оценивается ковариация.
    num_sources: int
        Число источников.

    Returns
    ---------------------------------------------------------------------------
    DoA: np.ndarray
        Оценки угловых координат.
    """
    col_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data1 = data.copy()
    data1[inds] = np.take(col_means, inds[1])
    R = sn.complex_cov(data1)
    res = MUSIC_DoA(R, num_sources)
    return res


def mean_imput_ESPRIT(data: np.ndarray, num_sources: int) -> np.ndarray:
    """
    Применяется mean imputation к столбцам с пропусками,
    оценивается ковариация полученного набора,
    по ковариации оставшихся данных реализуется ESPRIT.

    Parameters
    ---------------------------------------------------------------------------
    data: np.ndarray
        Данные, по которым оценивается ковариация.
    num_sources: int
        Число источников.

    Returns
    ---------------------------------------------------------------------------
    DoA: np.ndarray
        Оценки угловых координат.
    """
    col_means = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data1 = data.copy()
    data1[inds] = np.take(col_means, inds[1])
    R = sn.complex_cov(data1)
    res = ESPRIT_DoA(R, num_sources)
    return res