"""
平衡点线性化：连续时间 A_c/B_c → ZOH 离散 A/B。

要点（与原 compute_fit.py 第 8 节完全一致）：
  1) 在平衡点 p 处取 F 对状态、加速度、力矩的雅可比；
  2) 线性化点一般不是精确平衡点（该点 F(p) != 0），因此完整模型是
         dz/dt = A_c dz + B_c du + f_c,  f_c = -F_a^-1 * F(p)
     对应 TinyMPC 的 fdyn。A_c 只保留纯雅可比 -F_a^-1 * F_z，偏置全部进 f_c。
     （原 MATLAB 写法是把补偿项 M_eq^-1 * d(M_eq*u)/dz 折进 A_c；本模型的质量矩阵
      与状态无关，该项恒为 0，所以两种写法在这里数值相同，但分离 f 才能对接
      TinyMPC 的 APf/BPf 预计算。）
  3) F_a = dF_da 在平衡点处是对称正定（质量矩阵），用 solve 而不是 inv。

ZOH 离散化同时给出 A, B, f：f_d 用增广矩阵指数
    expm([[A_c, f_c], [0, 0]] * Ts) = [[A_d, f_d], [0, 1]]
计算，与 cont2discrete 的 A/B 一致。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.signal import cont2discrete

from .equilibrium import solve_equilibrium
from .geometry import geometry
from .types import Context, LinearPoint


def linearize_point(l_l: float, l_r: float, ctx: Context) -> LinearPoint:
    """在 (l_l, l_r) 工作点线性化并离散化，返回 LinearPoint。"""
    p = ctx.params
    cfg = ctx.cfg
    model = ctx.model
    gv = geometry(l_l, l_r, p)

    if ctx.cfg.linearization_point == "equilibrium":
        theta_body, theta_l, theta_r, notes = solve_equilibrium(l_l, l_r, ctx)
    elif ctx.cfg.linearization_point == "zero":
        theta_body = p.theta_p
        theta_l = theta_r = 0.0
        notes = []
    else:
        raise ValueError(f"未知的 linearization_point: "
                         f"{ctx.cfg.linearization_point!r}（可选 equilibrium / zero）")

    # 平衡点状态：位置全部为 0，姿态取平衡角，速度全 0
    state0 = np.array([0.0, 0.0, theta_body, theta_l, theta_r,
                       0.0, 0.0, 0.0, 0.0, 0.0])
    M0 = np.zeros(4)          # 平衡点处力矩为 0
    a0 = np.zeros(5)          # 平衡点处加速度为 0

    F_z = model.dF_dz(state0, a0, M0, gv)      # 5x10
    F_a = model.dF_da(state0, a0, M0, gv)      # 5x5
    F_M = model.dF_dM(state0, a0, M0, gv)      # 5x4
    F_rest = model.F_fun(state0, a0, M0, gv)   # 该点残差（不为 0）

    A_c = np.zeros((cfg.nx, cfg.nx))
    A_c[:5, 5:] = np.eye(5)
    A_c[5:, :] = -np.linalg.solve(F_a, F_z)

    B_c = np.zeros((cfg.nx, cfg.nu))
    B_c[5:, :] = -np.linalg.solve(F_a, F_M)

    # 仿射项：线性化点不是精确平衡点时 F(p) != 0
    f_c = np.zeros((cfg.nx, 1))
    f_c[5:, 0] = -np.linalg.solve(F_a, np.asarray(F_rest, dtype=float).ravel())

    C_c = np.eye(cfg.nx)
    D_c = np.zeros((cfg.nx, cfg.nu))

    A, B, _, _, _ = cont2discrete((A_c, B_c, C_c, D_c), cfg.Ts, method='zoh')
    f = zoh_affine(A_c, f_c, cfg.Ts)
    return LinearPoint(A=np.asarray(A, dtype=float), B=np.asarray(B, dtype=float),
                       theta_body=theta_body, theta_l=theta_l, theta_r=theta_r,
                       A_c=A_c, B_c=B_c, f_c=f_c, f=f, notes=list(notes))


def zoh_affine(A_c: np.ndarray, f_c: np.ndarray, Ts: float) -> np.ndarray:
    """ẋ = A x + f 的 ZOH 离散仿射项：f_d = ∫_0^Ts e^{A(Ts-t)} dt · f。"""
    nx = A_c.shape[0]
    aug = np.zeros((nx + 1, nx + 1))
    aug[:nx, :nx] = A_c * Ts
    aug[:nx, nx] = np.asarray(f_c, dtype=float).ravel() * Ts
    return expm(aug)[:nx, nx].reshape(nx, 1)


def discrete_ss(A_c: np.ndarray, B_c: np.ndarray,
                cfg) -> Tuple[np.ndarray, np.ndarray]:
    """单独的 ZOH 离散化工具（便于对 precompute 做纯矩阵单测）。"""
    A, B, _, _, _ = cont2discrete(
        (A_c, B_c, np.eye(A_c.shape[0]), np.zeros((A_c.shape[0], B_c.shape[1]))),
        cfg.Ts, method='zoh')
    return np.asarray(A, dtype=float), np.asarray(B, dtype=float)
