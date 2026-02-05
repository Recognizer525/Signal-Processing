import numpy as np
import time

from . import sensors as sn
#from common import estim_angles_pgd as ead
from . import optim_estim_angles as oea
from . import diff_sensor_structures as dss
from . import debug_funcs as df
from . import convergence as conv
from . import initialization as intl
from . import log_funcs as lf
from . import exact_maximizers as exm


def EM(angles: np.ndarray,
       P: np.ndarray,
       X: np.ndarray,
       Q: np.ndarray,
       Psi: np.ndarray,
       nu: int,
       max_iter: int = 50,
       rtol_params: float = 1e-3,
       rtol_lkhd: float = 1e-6,
       reg_coef: float = 0,
       show_lkhd: bool = True,
       show_params: bool = True,
       debug: bool = True) -> tuple[np.ndarray,
                                    np.ndarray,
                                    np.float64]:
    """
    Запускает MAP-ЕМ-алгоритм для выбранной начальной оценки параметров.

    Parameters
    ---------------------------------------------------------------------------
    angles: np.ndarray
        Начальная оценка вектора углов, которые соответствуют DOA.
    P: np.ndarray
        Начальная оценка ковариационной матрицы исходных сигналов.
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    Q: np.ndarray
        Ковариационная матрица шума.
    Psi: np.ndarray
        Априорная информация о ковариации сигналов.
    nu: int
        Степень уверенности в априорной информации.
    max_iter: int
        Предельное число итераций.
    rtol_params: float
        Величина, используемая для проверки сходимости итерационного процесса по параметрам.
    rtol_lkhd: float
        Величина, используемая для проверки сходимости итерационного процесса по правдоподобию.
    reg_coef: float
        Коэффициент регуляризации для смягчения численной нестабильности.
    show_lkhd: bool
        Показывать ли значение неполного правдоподобия после каждого обновления параметров.
    show_params: bool
        Показывать ли новую оценку параметров после каждого их обновления.
    debug: bool
        Проверять ли на ошибки массивы и матрицы, возникающие после Е-шага.

    Returns
    ---------------------------------------------------------------------------
    angles: np.ndarray
        Новая оценка DoA.
    P: np.ndarray
        Новая оценка ковариации исходных сигналов.
    lkhd: np.float64
        Новая оценка неполного правдоподобия.
    lkhd_list: list
        Список значений правдоподобия по итерациям.
    angles_list: list
        Список значений углов по итерациям.
    """
    Q_inv = np.linalg.inv(Q)
    Q_inv_sqrt = np.sqrt(Q_inv)

    lkhd = lf.log_posterior(X, angles, P, Q, Psi, nu)

    if show_lkhd:
        print(f"Inital log_posterior = {lkhd}")
    
    T = X.shape[0]
    L = Q.shape[0]
    K = P.shape[0]
    
    mask = ~np.isnan(X).any(axis=1)

    #print(f'Initial angles = {angles}')

    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1]+1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1

    R = sn.initial_Cov(X)
    A = dss.A_ULA(L, angles)

    K_Xm_cond = np.zeros((T, L, L), dtype=np.complex128)
    K_S_cond = np.zeros((T, K, K), dtype=np.complex128)

    Gap_based_Cov = np.zeros((T, K, K), dtype=np.complex128)
    Gap_based_Cross_cov = np.zeros((T, L, K), dtype=np.complex128)
    E_X_cond = X.copy()

    lkhd_list = list()
    angles_list = list()

    lkhd_list.append(lkhd)
    angles_list.append(angles)

    EM_Iteration = 1
    while EM_Iteration <= max_iter:
        print(f"Iteration={EM_Iteration}")
        R_inv_A_P = np.linalg.inv(R) @ A @ P
        R_inv_A_P_H = R_inv_A_P.conj().T
        Common_Cov_S = P - R_inv_A_P_H @ A @ P

        for i in range(T):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]

                R_OO = R[np.ix_(O_i, O_i)]
                R_OO = sn.cov_correcter(R_OO, reg_coef)
                R_MO = R[np.ix_(M_i, O_i)]
                R_MM = R[np.ix_(M_i, M_i)]

                E_X_cond[i, M_i] = R_MO @ np.linalg.inv(R_OO) @ E_X_cond[i, O_i]
                K_Xm_cond[np.ix_([i], M_i, M_i)] += (R_MM - R_MO @ 
                                                      np.linalg.inv(R_OO) @ 
                                                      R_MO.conj().T)


        E_X_E_X_H = np.einsum('li,lj -> lij', E_X_cond, E_X_cond.conj())
        Sigma_XX_arr = E_X_E_X_H + K_Xm_cond
        Sigma_XX_arr = sn.cov_correcter(Sigma_XX_arr, reg_coef)


        Gap_based_Cov[~mask] = R_inv_A_P_H @ Sigma_XX_arr[~mask] @ R_inv_A_P
        Gap_based_Cross_cov[~mask] = Sigma_XX_arr[~mask] @ R_inv_A_P

        Mu_S_cond = R_inv_A_P_H @ E_X_cond.T
        K_S_cond = Common_Cov_S + Gap_based_Cov

        E_X_E_S_H = np.einsum('li,lj -> lij', E_X_cond, Mu_S_cond.conj().T)
        E_S_E_S_H = np.einsum('li,lj -> lij', Mu_S_cond.T, Mu_S_cond.conj().T)

        Sigma_XS_arr = np.where(mask[:, None, None], E_X_E_S_H, Gap_based_Cross_cov)

        
        #Sigma_XX = np.mean(Sigma_XX_arr, axis=0)
        Sigma_XS = np.mean(Sigma_XS_arr, axis=0)
        Sigma_SS = np.mean(E_S_E_S_H + K_S_cond, axis=0)

        if debug:
            df.is_valid_result(E_X_E_X_H,'E_X_E_X_H', expected_shape=(T, L, L))
            df.is_valid_result(Sigma_XX_arr,'Sigma_XX_arr', expected_shape=(T, L, L), check_psd=True)
            df.is_valid_result(Mu_S_cond,'Mu_S_cond', expected_shape=(K,T))
            df.is_valid_result(K_S_cond,'K_S_cond', expected_shape=(T,K,K), check_psd=True)
            df.is_valid_result(E_X_E_S_H,'E_X_E_S_H', expected_shape=(T,L,K))
            df.is_valid_result(E_S_E_S_H,'E_S_E_S_H', expected_shape=(T,K,K), check_psd=True)
            df.is_valid_result(Sigma_XS,'Sigma_XS', expected_shape=(L, K))
            df.is_valid_result(Sigma_SS,'Sigma_SS', expected_shape=(K, K), check_psd=True)

        new_angles = oea.find_angles(Sigma_XS, angles, Sigma_SS, Q_inv_sqrt)
        idx = np.argsort(new_angles)
        new_angles[:] = new_angles[idx]

        new_P = exm.MAP_est_of_P(Sigma_SS, Psi, nu, X.shape[0])
        new_P = sn.cov_correcter(new_P, reg_coef)
        new_P[:] = new_P[np.ix_(idx, idx)]

        if show_params:
            print(f"new_angles={new_angles}")
            print(f"new_P:\n{new_P}")

        new_lkhd = lf.log_posterior(X, new_angles, new_P, Q, Psi, nu)

        if show_lkhd:
            print(f'Log_posterior is {new_lkhd} on iteration {EM_Iteration}.')

        if conv.if_params_converged(angles, new_angles, P, new_P, rtol_params):
            print("Parameters are converged!")
            break

        if conv.if_lkhd_converged(lkhd, new_lkhd, rtol_lkhd):
            print("Log_posterior is converged!")
            break

        if new_lkhd < lkhd:
            print(f"Accumulation of floating-point errors, log_posterior started to decrease!")
            break
        angles, P, lkhd = new_angles, new_P, new_lkhd
        angles_list.append(angles)
        lkhd_list.append(lkhd)

        A = dss.A_ULA(L, angles)
        R = A @ P @ A.conj().T + Q
        EM_Iteration += 1
        
    return angles, P, lkhd, lkhd_list, angles_list


def multistart_EM(X: np.ndarray,
                  K: int,
                  Q: np.ndarray,
                  theta_guess: np.ndarray,
                  Psi: np.ndarray,
                  nu: int,
                  num_of_starts: int = 10,
                  max_iter: int = 20,
                  rtol_params: float = 1e-6,
                  rtol_lkhd: float = 1e-6,
                  reg_coef: float = 0,
                  show_lkhd: bool = True,
                  show_params: bool = True,
                  show_time: bool = True,
                  debug: bool = True) -> tuple[np.ndarray,
                                                np.ndarray,
                                                np.float64]:
    """
    Реализует мультистарт для MAP-ЕМ-алгоритма.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    K: int
        Число источников.
    Q: np.ndarray
        Ковариационная матрица шума.
    theta_guess: np.ndarray
        Текущая начальная оценка углов.
    Psi: np.ndarray
        Априорная информация о ковариации сигналов.
    nu: int
        Степень уверенности в априорной информации.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol_params: float
        Величина, используемая для проверки сходимости итерационного процесса по параметрам.
    rtol_lkhd: float
        Величина, используемая для проверки сходимости итерационного процесса по правдоподобию.
    reg_coef: float
        Коэффициент регуляризации для смягчения численной нестабильности.
    show_lkhd: bool
        Показывать ли значение неполного правдоподобия после каждого обновления параметров.
    show_params: bool
        Показывать ли новую оценку параметров после каждого их обновления.
    debug: bool
        Проверять ли на ошибки массивы и матрицы, возникающие после Е-шага.

    Returns
    ---------------------------------------------------------------------------
    best_angles: np.ndarray
        Оценка DoA.
    best_P: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    best_lhd: np.float64
        Оценка неполного правдоподобия.
    best_lkhd_list: list
        Список значений правдоподобия по итерациям для лучшей итерации мультистарта.
    best_angles_list: list
        Список значений углов по итерациям для лучшей итерации мультистарта.
    """
    start_time = time.perf_counter() 
    best_lhd, best_angles, best_P, best_start = -np.inf, None, None, None
    best_lkhd_list, best_angles_list = None, None
    L = X.shape[1]
    R = sn.initial_Cov(X)
    for i in range(num_of_starts):
        print(f'{i}-th start')
        angles, P = intl.init_est_kn1(K, Q, R, theta_guess, L, iter=i, seed=i*12+70)
        est_angles, est_P, est_lhd, est_lkhd_list, est_angles_list  = EM(angles=angles, 
                                                                         P=P, 
                                                                         X=X, 
                                                                         Q=Q, 
                                                                         Psi=Psi,
                                                                         nu=nu,
                                                                         max_iter=max_iter, 
                                                                         rtol_params=rtol_params,
                                                                         rtol_lkhd=rtol_lkhd, 
                                                                         reg_coef=reg_coef,
                                                                         show_lkhd=show_lkhd,
                                                                         show_params=show_params,
                                                                         debug=debug)
        if est_lhd > best_lhd:
            best_lhd, best_start = est_lhd, i
            best_P, best_angles = est_P, est_angles
            best_lkhd_list, best_angles_list = est_lkhd_list, est_angles_list

    print(f"best_start={best_start}")
    end_time = time.perf_counter()
    if show_time:
        elapsed_time = end_time - start_time
        print(f"Время выполнения функции: {elapsed_time:.6f} секунд")
    return best_angles, best_P, best_lhd, best_lkhd_list, best_angles_list


def multistart_EM2(X: np.ndarray,
                   K: int,
                   Q: np.ndarray,
                   Psi: np.ndarray,
                   nu: int,
                   num_of_starts: int = 10,
                   max_iter: int = 20,
                   rtol_params: float = 1e-6,
                   rtol_lkhd: float = 1e-6,
                   reg_coef: float = 0,
                   show_lkhd: bool = True,
                   show_params: bool = True,
                   show_time: bool = True,
                   debug: bool = True) -> tuple[np.ndarray,
                                                np.ndarray,
                                                np.float64]:
    """
    Реализует мультистарт для MAP-ЕМ-алгоритма.

    Parameters
    ---------------------------------------------------------------------------
    X: np.ndarray
        Двумерный массив, соответствующий наблюдениям.
    K: int
        Число источников.
    Q: np.ndarray
        Ковариационная матрица шума.
    Psi: np.ndarray
        Априорная информация о ковариации сигналов.
    nu: int
        Степень уверенности в априорной информации.
    num_of_starts: int
        Число запусков.
    max_iter: int
        Предельное число итераций.
    rtol_params: float
        Величина, используемая для проверки сходимости итерационного процесса по параметрам.
    rtol_lkhd: float
        Величина, используемая для проверки сходимости итерационного процесса по правдоподобию.
    reg_coef: float
        Коэффициент регуляризации для смягчения численной нестабильности.
    show_lkhd: bool
        Показывать ли значение неполного правдоподобия после каждого обновления параметров.
    show_params: bool
        Показывать ли новую оценку параметров после каждого их обновления.
    debug: bool
        Проверять ли на ошибки массивы и матрицы, возникающие после Е-шага.

    Returns
    ---------------------------------------------------------------------------
    best_angles: np.ndarray
        Оценка DoA.
    best_P: np.ndarray
        Оценка ковариационной матрицы исходных сигналов.
    best_lhd: np.float64
        Оценка неполного правдоподобия.
    best_lkhd_list: list
        Список значений правдоподобия по итерациям для лучшей итерации мультистарта.
    best_angles_list: list
        Список значений углов по итерациям для лучшей итерации мультистарта.
    """
    start_time = time.perf_counter() 
    best_lhd, best_angles, best_P, best_start = -np.inf, None, None, None
    best_lkhd_list, best_angles_list = None, None
    L = X.shape[1]
    R = sn.initial_Cov(X)
    for i in range(num_of_starts):
        print(f'{i}-th start')
        angles, P = intl.init_est_kn2(K, Q, R, L, seed=i*100)
        est_angles, est_P, est_lhd, est_lkhd_list, est_angles_list  = EM(angles=angles, 
                                                                         P=P, 
                                                                         X=X, 
                                                                         Q=Q,
                                                                         Psi=Psi,
                                                                         nu=nu, 
                                                                         max_iter=max_iter, 
                                                                         rtol_params=rtol_params,
                                                                         rtol_lkhd=rtol_lkhd, 
                                                                         reg_coef=reg_coef,
                                                                         show_lkhd=show_lkhd,
                                                                         show_params=show_params,
                                                                         debug=debug)
        if est_lhd > best_lhd:
            best_lhd, best_start = est_lhd, i
            best_P, best_angles = est_P, est_angles
            best_lkhd_list, best_angles_list = est_lkhd_list, est_angles_list

    print(f"best_start={best_start}")
    end_time = time.perf_counter()
    if show_time:
        elapsed_time = end_time - start_time
        print(f"Время выполнения функции: {elapsed_time:.6f} секунд")
    return best_angles, best_P, best_lhd, best_lkhd_list, best_angles_list


