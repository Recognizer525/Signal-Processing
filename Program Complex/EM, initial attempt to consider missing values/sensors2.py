import numpy as np
import scipy
import math
from functools import partial

dist_ratio = 0.5

def CN(size: int, number: int, Gamma: np.ndarray):
    """
    Генерирует комплексные нормальные вектора (circularly-symmetric case).
    size - размер вектора;
    number - количество векторов;
    Gamma - ковариационная матрица.
    """ 
    n = 2 * size # Размер ковариационной матрицы совместного распределения
    C = np.zeros((n,n), dtype=np.float64)
    C[:size,:size] = Gamma.real
    C[size:,size:] = Gamma.real
    C[:size,size:] = -Gamma.imag
    C[size:,:size] = Gamma.imag
    mu = np.zeros(n)
    B = np.random.RandomState(70).multivariate_normal(mu, 0.5*C, number)
    D = B[:,:size] + 1j * B[:, size:]
    return D

def space_covariance_matrix(X: np.ndarray):
    """
    Метод предназначен для формирования оценки матрицы пространственной ковариации.
    X - коллекция полученных сигналов.
    """
    N = len(X)
    ans = np.zeros((len(X[0]), len(X[0])), dtype = np.complex128)
    for i in range(len(X)):
        ans += X[i][:, None] @ X[i][:, None].conj().T
    return ans * (1/N)


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
        np.random.RandomState(rs).shuffle(h)
        X1[:,mis_cols[i]][np.where(h==1)] = np.nan
    return X1


def f(theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, DoS: dict):
    """
    Минимизируемая, на М-шаге, функция.
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    DoS - словарь вычислений статистик по каждому снимку.
    """
    ans = 0
    
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    Mv, Ov = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    
    M, L, G = Ga_s.shape[0], Ga_n.shape[0], X.shape[0]
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))
    for i in range(G):
        if set(Ov[i, ]) != set(col_numbers - 1):
            M_i, O_i = Mv[i, ][Mv[i, ] > -1], Ov[i, ][Ov[i, ] > -1]
            # Определяем матрицы управляющих векторов отдельно для исправных и отдельно для неисправных датчиков.
            A_o, A_m = A[O_i], A[M_i]
            # Аналогично определяем ковариационные матрицы шума для двух групп датчиков.
            Ga_o, Ga_m = Ga_n[np.ix_(O_i, O_i)], Ga_n[np.ix_(M_i, M_i)]
            # Определяем длину вектора латентных переменных
            L2 = len(M_i)
            
            K_Xo = A_o @ Ga_s @ A_o.conj().T + Ga_o
                
            K_Y = np.zeros((L2+M,L2+M), dtype=np.complex128)
            K_Y[L2:,L2:] = Ga_s
            K_Y[:L2,:L2] = A_m @ Ga_s @ A_m.conj().T + Ga_m
            K_Y[L2:,:L2] = Ga_s @ A_m.conj().T
            K_Y[:L2,L2:] = A_m @ Ga_s

            K_Xo_Y = np.zeros((L-L2,M+L2), dtype=np.complex128)
            #print('M, L, L2', M, L, L2)
            #print('K_Xo_Y',K_Xo_Y.shape)
            K_Xo_Y[:,:L2] = A_o @ Ga_s @ A_m.conj().T
            K_Xo_Y[:,L2:] = A_o @ Ga_s
            K_Y_Xo = K_Xo_Y.conj().T

            #print('K_Xo_Y',K_Xo_Y.shape)
            #print('np.linalg.inv(K_Y)',np.linalg.inv(K_Y).shape)
            #print("DoS[i]['Cond_mean']",DoS[i]['Cond_mean'].shape)

            K_Xo_cond_Y = K_Xo - K_Xo_Y @ np.linalg.inv(K_Y) @ K_Y_Xo
            Mu_Xo_cond_Y = K_Xo_Y @ np.linalg.inv(K_Y) @ DoS[i]['Cond_mean']
            
            ans += (X[i,O_i] - Mu_Xo_cond_Y).conj().T @ np.linalg.inv(K_Xo_cond_Y) @ (X[i,O_i] - Mu_Xo_cond_Y) + \
            np.trace(np.linalg.inv(K_Y) @ DoS[i]['Cond_cov']) + DoS[i]['Cond_mean'].conj().T @  np.linalg.inv(K_Y) @ DoS[i]['Cond_mean']

    return ans.real


def equation_solver(theta: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, X: np.ndarray, Dict_of_Snapshots: dict):
    """
    Нахождение оптимального вектора углов на М-шаге.
    theta - вектор углов, которые соответствуют DOA;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    X - коллекция полученных сигналов;
    Dict_of_Snapshots - словарь вычислений статистик по каждому снимку.
    """
    simplified_f = partial(f, Ga_s=Ga_s, Ga_n=Ga_n, X=X, DoS=Dict_of_Snapshots)
    ans = scipy.optimize.minimize(simplified_f, theta.reshape(-1,), method='Nelder-Mead', options={ 'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}).x
    return ans, simplified_f(ans)




def angle_correcter(theta: np.ndarray):
    """
    Набор углов преобразуется таким образом, чтобы все углы были в области [-pi/2; pi/2], для этого по мере необходимости добавляется/вычитается 2*pi 
    требуемое число раз, кроме того, учитывается тот факт, что синус симметричен относительно pi/2 и -pi/2.
    theta - вектор углов, которые соответствуют DOA.
    """
    for i in range(len(theta)):
        while theta[i] > np.pi:
            theta[i] -= 2*np.pi
        while theta[i] < -np.pi:
            theta[i] += 2*np.pi

    for i in range(len(theta)):
        if theta[i] > np.pi/2:
            theta[i] = np.pi - theta[i]
        elif theta[i] < -np.pi/2:
            theta[i] = - np.pi - theta[i]
    return theta

def EM(theta: np.ndarray, X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, max_iter: int = 20, eps: float = 1e-8) -> np.ndarray:
    '''
    Функция применяет алгоритм максимального правдоподобия к полученным данным для восстановления пропущенных значений.
    '''
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    no_conv = True
    Mv, Ov = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    L, M = Ga_n.shape[0], Ga_s.shape[0]
    A = np.exp(-2j * np.pi * dist_ratio * np.arange(L).reshape(-1,1) * np.sin(theta).reshape(1,-1))  
    X_modified = X.copy()
    EM_Iteration = 0
    while no_conv and EM_Iteration < max_iter:
        # Создаем ассоциативный массив для различных ковариационных матриц и векторов мат. ожидания;
        # это необходимо для того, чтобы на М-шаге не повторять вычисления с Е-шага.
        Dict_of_Snapshots = {}

        # Е-шаг
        for i in range(X.shape[0]):
            # Если в какой-то момент времени имеются пропуски на некоторых датчиках:
            if set(Ov[i, ]) != set(col_numbers - 1):
                M_i, O_i = Mv[i, ][Mv[i, ] > -1], Ov[i, ][Ov[i, ] > -1]
                # Определяем матрицы управляющих векторов отдельно для исправных и отдельно для неисправных датчиков.
                A_o, A_m = A[O_i], A[M_i]
                # Аналогично определяем ковариационные матрицы шума для двух групп датчиков.
                Ga_o, Ga_m = Ga_n[np.ix_(O_i, O_i)], Ga_n[np.ix_(M_i, M_i)]
                # Определяем длину вектора латентных переменных
                L2 = len(M_i)

                K_Xo = A_o @ Ga_s @ A_o.conj().T + Ga_o
                
                K_Y = np.zeros((L2+M,L2+M), dtype=np.complex128)
                K_Y[L2:,L2:] = Ga_s
                K_Y[:L2,:L2] = A_m @ Ga_s @ A_m.conj().T + Ga_m
                K_Y[L2:,:L2] = Ga_s @ A_m.conj().T
                K_Y[:L2,L2:] = A_m @ Ga_s

                K_Xo_Y = np.zeros((L-L2,L2+M), dtype=np.complex128)
                K_Xo_Y[:,:L2] = A_o @ Ga_s @ A_m.conj().T
                K_Xo_Y[:,L2:] = A_o @ Ga_s
                K_Y_Xo = K_Xo_Y.conj().T

                K_Y_cond_Xo = K_Y - K_Y_Xo @ np.linalg.inv(K_Xo) @ K_Xo_Y
                Mu_Y_cond_Xo = K_Y_Xo @ np.linalg.inv(K_Xo) @ X_modified[i, O_i]
                #print('K_Y_cond_Xo', K_Y_cond_Xo.shape)
                #print('Mu_Y_cond_Xo', Mu_Y_cond_Xo.shape)
                snapshot_dict = {'Cond_mean': Mu_Y_cond_Xo,
                                 'Cond_cov': K_Y_cond_Xo          
                                }
                Dict_of_Snapshots[i] = snapshot_dict


        theta_new, neg_likelihood = equation_solver(theta, Ga_s, Ga_n, X, Dict_of_Snapshots)
        print('theta_new', theta_new)
        print(f"Iteration={EM_Iteration}, theta_new={theta_new:}, -likelihood = {neg_likelihood:.5f}")
        no_conv = np.linalg.norm(theta - theta_new) >= eps
        if not no_conv:
            print(f"norm={np.linalg.norm(theta - theta_new)}")
        theta = theta_new
        EM_Iteration += 1
    return theta_new, neg_likelihood


def multi_start_EM(X: np.ndarray, Ga_s: np.ndarray, Ga_n: np.ndarray, num_of_starts: int = 20, max_iter: int = 20, eps: float = 1e-6):
    """
    Мультистарт для ЕМ-алгоритма.
    X - коллекция полученных сигналов;
    Ga_s - ковариация сигнала;
    Ga_n - ковариация шума;
    num_of_starts - число запусков;
    max_iter - предельное число итерация;
    eps - величина, используемая для проверки сходимости последних итераций.
    """
    best_neg_lhd, best_theta = np.inf, None
    for i in range(num_of_starts):
        print(f'{i}-th start')
        M = Ga_s.shape[0]
        theta = np.random.uniform(-np.pi, np.pi, M).reshape(M,1)
        est_theta, neg_lhd = EM(theta, X, Ga_s, Ga_n, max_iter, eps)
        if neg_lhd < best_neg_lhd:
            best_neg_lhd, best_theta = neg_lhd, est_theta
    best_theta = angle_correcter(best_theta)
    return best_theta, best_neg_lhd

