import numpy as np
import torch

DIST_RATIO = 0.5


def A_ULA(L: int, theta: np.ndarray, dist: float = DIST_RATIO) -> np.ndarray:
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


def A_URA(M: int, N: int, 
          theta: np.ndarray, phi: np.ndarray, 
          dx: float = DIST_RATIO, dy: float = DIST_RATIO) -> np.ndarray:
    """
    Создает матрицу векторов направленности для 
    равномерной прямоугольной антенной решетки (URA).

    Parameters
    ---------------------------------------------------------------------------
    M: int
        Число сенсоров по оси X.
    N: int
        Число сенсоров по оси Y.
    theta: np.ndarray
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: np.ndarray
        Оценка угла места (elevation), размер (K,1) или (K,).
    dx: float
        Расстояние между соседними элементами по горизонтали в долях длины волны.
        к длине волны.
    dy: float
        Расстояние между соседними элементами по вертикали в долях длины волны.
        к длине волны.
    
    
    Returns
    ---------------------------------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности размера (L,K).
    """
    m = np.arange(M).reshape(-1,1)
    n = np.arange(N).reshape(1,-1)
    return np.exp(-2j * np.pi * (dx * m * np.sin(phi) * np.cos(theta) +
                                 dy * n * np.sin(phi) * np.sin(theta))).reshape(-1,1)


def A_UCA(N: int, theta: np.ndarray, phi: np.ndarray, R: float = 0.5) -> np.ndarray:
    """
    Создает матрицу векторов направленности для 
    равномерной круговой антенной решетки (URA).

    Parameters
    ---------------------------------------------------------------------------
    N: int
        Число сенсоров.
    theta: np.ndarray
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: np.ndarray
        Оценка угла места (elevation), размер (K,1) или (K,).
    R: float
        Отношение радиуса антенной решетки к длине волны.
    
    Returns
    ---------------------------------------------------------------------------
    A: np.ndarray
        Матрица векторов направленности размера (L,K).
    """
    n = np.arange(N)
    theta_n = 2 * np.pi * n / N
    return np.exp(-2j * np.pi * R * np.sin(phi) * np.cos(theta - theta_n)).reshape(-1,1)


def A_custom(coords: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
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
    k = 2 * np.pi * np.array([np.sin(phi)*np.cos(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(phi)])
    return np.exp(-1j * coords @ k.reshape(-1,1))


def A_ULA_torch(L: int, theta: torch.Tensor, dist: float = DIST_RATIO) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для линейной равномерной антенной
    решетки (ULA).

    Parameters
    ---------------------------------------------------------------------------
    L: int
        Число сенсоров.
    theta: torch.Tensor
        Оценка DoA, размер (K,1) или (K,).
    
    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности.
    """
    device = theta.device
    sensor_indices = torch.arange(L, device=device).reshape(-1, 1).float() 
    return torch.exp(-2j * torch.pi * dist * 
                     sensor_indices * torch.sin(theta))


def A_URA_torch(M: int, N: int, 
                theta: torch.Tensor, phi: torch.Tensor, 
                dx: float = DIST_RATIO, dy: float = DIST_RATIO) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для 
    равномерной прямоугольной антенной решетки (URA) в PyTorch.

    Parameters
    ---------------------------------------------------------------------------
    M: int
        Число сенсоров по оси X.
    N: int
        Число сенсоров по оси Y.
    theta: torch.Tensor
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: torch.Tensor
        Оценка угла места (elevation), размер (K,1) или (K,).
    dx: float
        Расстояние между соседними элементами по горизонтали в долях длины волны.
    dy: float
        Расстояние между соседними элементами по вертикали в долях длины волны.

    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности, размер (M*N, K).
    """
    device = theta.device

    m = torch.arange(M, device=device).reshape(-1, 1).float() # (M,1)
    n = torch.arange(N, device=device).reshape(1, -1).float() # (1, N)

    exponent = -2j * torch.pi * (
        dx * m * torch.sin(phi) * torch.cos(theta) +
        dy * n * torch.sin(phi) * torch.sin(theta)
    )  # размер (M,N,K)

    # Преобразуем к размеру (M*N, K)
    return exponent.reshape(-1, theta.shape[0]).exp()


def A_UCA_torch(N: int, 
                theta: torch.Tensor, 
                phi: torch.Tensor, 
                R: float = DIST_RATIO) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для 
    равномерной круговой антенной решетки (UCA) в PyTorch.

    Parameters
    ---------------------------------------------------------------------------
    N: int
        Число сенсоров.
    theta: torch.Tensor
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: torch.Tensor
        Оценка угла места (elevation), размер (K,1) или (K,).
    R: float
        Отношение радиуса решетки к длине волны.

    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности размера (N, K).
    """
    device = theta.device
    n = torch.arange(N, device=device).float()  # (N,)
    theta_n = 2 * torch.pi * n / N             # (N,)
    exponent = -2j * torch.pi * R * torch.sin(phi) * torch.cos(theta - theta_n.reshape(-1, 1))

    return exponent.exp()  # размер (N, K)


def A_custom_torch(coords: torch.Tensor, 
                   theta: torch.Tensor, 
                   phi: torch.Tensor) -> torch.Tensor:
    """
    Создает матрицу векторов направленности для 
    произвольной антенной решетки в PyTorch.

    Parameters
    ---------------------------------------------------------------------------
    coords: torch.Tensor
        Координаты сенсоров, размер (L,3).
    theta: torch.Tensor
        Оценка азимута (azimuth), размер (K,1) или (K,).
    phi: torch.Tensor
        Оценка угла места (elevation), размер (K,1) или (K,).

    Returns
    ---------------------------------------------------------------------------
    A: torch.Tensor
        Матрица векторов направленности размера (L,K).
    """
    k = 2 * torch.pi * torch.stack([
        torch.sin(phi) * torch.cos(theta),  
        torch.sin(phi) * torch.sin(theta),  
        torch.cos(phi)                      
    ], dim=0) 
    exponent = -1j * coords @ k
    return exponent.exp()