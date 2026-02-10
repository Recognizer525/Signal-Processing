import numpy as np
import torch

DIST_RATIO = 0.5


def A_ULA(L: int, 
          theta: np.ndarray, 
          dist: float = DIST_RATIO) -> np.ndarray:
    """
    Создает матрицу векторов направленности для 
    равномерной линейной антенной решетки (ULA).

    Parameters
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    theta: np.ndarray
        Оценка DoA, размер (K,1) или (K,).
    dist: float
        Отношение расстояния между соседними элементами к длине волны.
    
    Returns
    ---------------------------------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности размера (L,K).
    """
    return (np.exp(-2j * np.pi * dist * 
                   np.arange(L).reshape(-1,1) * np.sin(theta)))


def A_custom(coords: np.ndarray, 
             theta: np.ndarray, 
             phi: np.ndarray) -> np.ndarray:
    """
    Создает матрицу векторов направленности для 
    произвольной антенной решетки.

    Parameters
    ---------------------------------------------------------------------------
    coords: np.ndarray
        Координаты сенсоров, массив размера (L,3).
    theta: np.ndarray
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: np.ndarray
        Оценка угла места (elevation), размер (K,1) или (K,).

    
    Returns
    ---------------------------------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности размера (L,K).
    """
    k = 2 * np.pi * np.vstack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    return np.exp(-1j * (coords @ k))


def A_ULA_torch(L: int, 
                u: torch.Tensor, 
                dist: float = DIST_RATIO) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для линейной равномерной антенной
    решетки (ULA).

    Parameters
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    u: torch.Tensor
        Направляющие косинусы, shape (K,) или (K,1), u ∈ [-1, 1].
    dist: float
        Отношение расстояния между сенсорами к длине волны.
    
    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности.
    """
    sensor_indices = torch.arange(L).reshape(-1, 1).float() 
    return torch.exp(-2j * torch.pi * dist * 
                     sensor_indices * u)


def A_custom_torch(coords: torch.Tensor, 
                   angles: torch.Tensor) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для 
    произвольной антенной решетки в PyTorch.

    Parameters
    ---------------------------------------------------------------------------
    coords: torch.Tensor
        Координаты сенсоров, размер (L,3).
    angles: torch.Tensor
        Оценка углов, размер (2K,).

    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности размера (L,K).
    """
    K = angles.numel() // 2
    theta = angles[:K]
    phi = angles[K:]

    k = 2 * torch.pi * torch.vstack((
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi)
    ))
    return torch.exp(-1j * (coords @ k))


def rectangular_array_coords(origin: np.ndarray,
                             Nx: int,
                             Ny: int,
                             dx: float,
                             dy: float) -> np.ndarray:
    """
    Создает координаты прямоугольной антенной решетки в плоскости XY.

    Parameters
    ---------------------------------------------------------------------------
    origin : np.ndarray
        Опорная точка решетки, форма (3,).
    Nx : int
        Количество сенсоров вдоль оси x.
    Ny : int
        Количество сенсоров вдоль оси y.
    dx : float
        Шаг между сенсорами по x.
    dy : float
        Шаг между сенсорами по y.

    Returns
    ---------------------------------------------------------------------------
    coords : np.ndarray
        Координаты сенсоров, форма (Nx*Ny, 3).
    """

    origin = np.asarray(origin).reshape(1, 3)

    x = np.arange(Nx) * dx
    y = np.arange(Ny) * dy

    xx, yy = np.meshgrid(x, y, indexing='ij')

    zz = np.zeros_like(xx)

    coords = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)

    return coords + origin