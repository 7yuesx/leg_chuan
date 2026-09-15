"""
离线预计算：TinyMPC 语义的 cache 求解 + LESO 增益。

本模块只吃 numpy 矩阵，不依赖 sympy / mujoco / 文件 IO，可以单独做单元测试。
对应 TinyMPC 的 tiny_precompute_and_set_cache。

算法（只有一套，与 TinyMPC 一致）：
    权重先做 rho 正则化： Q1 = Q + rho*I,  R1 = R + rho*I
    解 (Q1, R1) 的 Riccati 方程得到 Pinf、Kinf：
        Kinf = (R1 + B'Pinf B)^-1 B'Pinf A
        Pinf = Q1 + A'Pinf (A - B*Kinf)
    再由同一个 Pinf 派生：
        Quu_inv = (R1 + B'Pinf B)^-1
        AmBKt   = (A - B*Kinf)^T
        APf     = AmBKt*Pinf*fdyn,  BPf = B'*Pinf*fdyn
    写进 coeff 文件时符号取 +（cache_sign="positive"，与 TinyMPC 相同）。

为什么 Q/R 要先加 rho*I：ADMM 子问题的代价里状态与输入的 Hessian 各多一个 rho*I，
子问题的最优控制律天生就是 (Q+rho*I, R+rho*I) 的 LQR。所以这不是调参选择。

求解方式用 scipy 的 solve_discrete_are（直接法，一次到机器精度），而不是 TinyMPC 的
固定点迭代。两者解的是同一个方程：这里实测把 (Q+rho*I, R+rho*I) 交给直接法，
与 TinyMPC 的迭代结果相差 P 6.3e-5 / K 7.6e-5 / Quu_inv 1.1e-6 / AmBKt 8.4e-5（迭代容差）。
选直接法的原因：本系统可控性 Gram 最小奇异值约 9e-8，固定点迭代到 tol=1e-5 需要
2800~4000 次（TinyMPC 自己写死的 1000 次上限完全不够），直接法快十倍且没有收敛余量问题。
riccati_fixed_point() 保留为参考实现/交叉校验用，不参与默认流程。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are

from .types import CacheMatrices

# riccati_fixed_point 默认参数（TinyMPC 的取值：1000 次、1e-5）
RICCATI_MAX_ITER = 1000
RICCATI_TOL = 1e-5


def dare_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
              R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接法求解离散 Riccati 方程，返回 (P, K)，两者都是正号约定。

    传入的 Q、R 应当已经是正则化后的权重（Q+rho*I、R+rho*I）。
    Q 必须半正定，否则离散黎卡提方程可能无稳定解（scipy 会报 symplectic pencil 错误）。
    """
    qmin = float(np.linalg.eigvalsh(Q)[0])
    if qmin < -1e-9:
        raise ValueError(
            f"Q 不是半正定（最小特征值 {qmin:.3g}）：θ_l/θ_r 交叉项过大，"
            f"需满足 |offdiag| <= sqrt(q_l*q_r)")

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return P, K


def riccati_fixed_point(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                        rho: float,
                        max_iter: int = RICCATI_MAX_ITER,
                        tol: float = RICCATI_TOL,
                        ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """
    TinyMPC 的 Riccati 固定点迭代（参考实现，不参与默认流程）。

    与 tiny_api.cpp 的 tiny_precompute_and_set_cache 逐行一致：
    初值 Ktp1 = 0、Ptp1 = rho*I，迭代
        K = (R1 + B' P B)^-1 B' P A
        P = Q1 + A' P (A - B K)
    直到 max|K - K_prev| < tol，返回 (Pinf, Kinf, iters, converged)。

    注意本系统收敛慢：tol=1e-5 需要 2800~4000 次，超过 TinyMPC 写死的 1000 次上限
    （TinyMPC 那种情况会静默返回未收敛值）。默认流程改用 dare_gain()。
    """
    nx, nu = B.shape
    Q1 = Q + rho * np.eye(nx)
    R1 = R + rho * np.eye(nu)

    K_prev = np.zeros((nu, nx))
    P = rho * np.eye(nx)
    K = np.zeros((nu, nx))

    for it in range(1, max_iter + 1):
        K = np.linalg.solve(R1 + B.T @ P @ B, B.T @ P @ A)
        P = Q1 + A.T @ P @ (A - B @ K)
        if np.max(np.abs(K - K_prev)) < tol:
            return P, K, it, True
        K_prev = K
    return P, K, max_iter, False


def cache_from_positive(P_pos: np.ndarray, K_pos: np.ndarray,
                        A: np.ndarray, B: np.ndarray, R: np.ndarray,
                        rho: float,
                        cache_sign: str = "positive",
                        f: Optional[np.ndarray] = None) -> CacheMatrices:
    """
    由正定解 (P_pos, K_pos) 派生 cache 字段。

    Quu_inv = (R + rho*I + B'P_pos B)^-1（TinyMPC 的 Quu_inv / C1）
    AmBKt = (A - B*K_pos)^T（即 TinyMPC 的 C2）
    f 非 None 时再算 APf = AmBKt*P_pos*f、BPf = B'*P_pos*f。
    """
    if cache_sign == "positive":
        sgn = 1.0
    elif cache_sign == "negative":
        sgn = -1.0
    else:
        raise ValueError(f"未知的 cache_sign: {cache_sign!r}（可选 positive / negative）")

    nu = B.shape[1]
    Quu_inv = np.linalg.inv(R + rho * np.eye(nu) + B.T @ P_pos @ B)
    AmBKt = (A - B @ K_pos).T
    APf = BPf = None
    if f is not None:
        # TinyMPC 的仿射项预计算（tiny_precompute_and_set_cache 最后两行）
        fdyn = np.asarray(f, dtype=float).reshape(-1, 1)
        APf = AmBKt @ P_pos @ fdyn
        BPf = B.T @ P_pos @ fdyn
    return CacheMatrices(P=sgn * P_pos, K=sgn * K_pos, Quu_inv=Quu_inv, AmBKt=AmBKt,
                         sign=cache_sign, APf=APf, BPf=BPf)


def precompute_cache(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                     rho: float, cache_sign: str = "positive",
                     f: Optional[np.ndarray] = None) -> CacheMatrices:
    """
    TinyMPC 语义的 cache：权重先加 rho*I，再用 scipy 直接法解 Riccati。

    Q、R 传未正则化的权重，rho 由调用方给出；内部按 Q+rho*I、R+rho*I 求解。
    """
    nx, nu = B.shape
    Q1 = Q + rho * np.eye(nx)
    R1 = R + rho * np.eye(nu)
    P_pos, K_pos = dare_gain(A, B, Q1, R1)
    return cache_from_positive(P_pos, K_pos, A, B, R, rho,
                               cache_sign=cache_sign, f=f)


def leso_gain(A: np.ndarray, B: np.ndarray,
              obs_Qo_state: float, obs_Qo_d: float, obs_Ro: float,
              nx: int, nu: int) -> np.ndarray:
    """
    LESO 观测器增益 Le。

    增广模型：
        X = [x(nx); d(nu)]，d 为输入扰动，u_actual = u + d
        Ae = [[A, B], [0, I_nu]]，Ce = [I_nx, 0]
    用离散 LQE（对偶 DARE）而非极点配置：极点配置得到的 Le 随腿长不连续、无法拟合，
    LQE 增益平滑可拟合。
    """
    E_aug = B
    Ae = np.block([[A, E_aug],
                   [np.zeros((nu, nx)), np.eye(nu)]])
    Ce = np.block([np.eye(nx), np.zeros((nx, nu))])

    Qo = np.zeros((nx + nu, nx + nu))
    Qo[:nx, :nx] = np.eye(nx) * obs_Qo_state
    Qo[nx:, nx:] = np.eye(nu) * obs_Qo_d
    Ro = np.eye(nx) * obs_Ro

    P_obs = solve_discrete_are(Ae.T, Ce.T, Qo, Ro)             # 对偶 DARE
    return P_obs @ Ce.T @ np.linalg.inv(Ro + Ce @ P_obs @ Ce.T)


def check_cache_consistency(cache: CacheMatrices, A: np.ndarray, B: np.ndarray,
                            R: np.ndarray, rho: float) -> Optional[str]:
    """
    自洽性校验：Quu_inv 与 AmBKt 必须能由 P、K 复现（TinyMPC 里它们同源于 Pinf）。
    返回 None 表示通过，否则返回描述问题的字符串。
    """
    sgn = 1.0 if cache.sign == "positive" else -1.0
    P_pos = sgn * cache.P
    K_pos = sgn * cache.K
    nu = B.shape[1]
    Quu_inv_ref = np.linalg.inv(R + rho * np.eye(nu) + B.T @ P_pos @ B)
    AmBKt_ref = (A - B @ K_pos).T

    def _relerr(x, y):
        denom = max(np.max(np.abs(y)), 1e-12)
        return float(np.max(np.abs(x - y)) / denom)

    err_s = _relerr(cache.Quu_inv, Quu_inv_ref)
    err_a = _relerr(cache.AmBKt, AmBKt_ref)
    if err_s > 1e-8 or err_a > 1e-8:
        return (f"Quu_inv/AmBKt 与 P/K 不自洽"
                f"（relerr Quu_inv={err_s:.2e}, AmBKt={err_a:.2e}）")
    return None
