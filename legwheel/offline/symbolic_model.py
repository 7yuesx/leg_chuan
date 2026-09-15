"""
符号动力学模型：动能 T、势能 V、拉格朗日方程、隐式残差 F 及其雅可比。

纯符号层，不涉及任何数值求解、几何实测或文件 IO。一次 build_model() 之后
所有 lambdify 结果缓存在 SymbolicModel 上，后续所有采样点复用。

状态 z = [x, w, theta, theta_l, theta_r, x_dot, w_dot, theta_dot,
          theta_dot_l, theta_dot_r]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import sympy as sp

from .types import Params


@dataclass(eq=False)          # eq=False: 保持按对象身份哈希，便于放进缓存键
class SymbolicModel:
    """符号模型的 lambdify 结果集合。"""

    F_fun: Any               # (state, accs, Ms, geom) -> (5, 1) 残差
    dF_dz: Any               # -> (5, 10)
    dF_da: Any               # -> (5, 5)
    dF_dM: Any               # -> (5, 4)
    dM_eq_u_fun: Any         # (state, accs, Ms, geom, u) -> (5, 10) 仿射修正项
    V_balance: Any           # (theta, theta_l, theta_r, geom) -> 标量
    dV_dq: Any               # (theta, theta_l, theta_r, geom) -> (3,)：
                             # (dV/dtheta, dV/dtheta_l, dV/dtheta_r)


def build_model(p: Params = None) -> SymbolicModel:
    """建立符号模型。p 为 None 时使用 config.PARAMS。"""
    if p is None:
        from .config import PARAMS
        p = PARAMS

    # ---- 符号变量 ----
    x, w, th, thl, thr = sp.symbols('x w theta theta_l theta_r')
    vx, vw, vth, vthl, vthr = sp.symbols('x_dot w_dot theta_dot theta_dot_l theta_dot_r')
    ax, aw, ath, athl, athr = sp.symbols('x_ddot w_ddot theta_ddot theta_ddot_l theta_ddot_r')
    M_l2, M_r2, M_l3, M_r3 = sp.symbols('M_l2 M_r2 M_l3 M_r3')
    l_l, l_r = sp.symbols('l_l l_r')
    c_h1, c_h2, c_x1, c_x2, thp1, thp2 = sp.symbols('c_h1 c_h2 c_x1 c_x2 theta_p1 theta_p2')

    coords = [x, w, th, thl, thr]
    vels = [vx, vw, vth, vthl, vthr]
    accs = [ax, aw, ath, athl, athr]
    state = coords + vels
    Ms = [M_l2, M_r2, M_l3, M_r3]
    geom_syms = [l_l, l_r, c_h1, c_h2, c_x1, c_x2, thp1, thp2]

    # ---- 动能 T、势能 V ----
    T = (
        sp.Rational(1, 2) * p.m_w
        * (vx - sp.sqrt(p.d**2 + (sp.sin(thl) * 2 * l_l)**2) * vw
           + sp.cos(thl) * 2 * l_l * vthl)**2
        + sp.Rational(1, 2) * p.m_w
        * (vx + sp.sqrt(p.d**2 + (sp.sin(thr) * 2 * l_r)**2) * vw
           + sp.cos(thr) * 2 * l_r * vthr)**2
        + sp.Rational(1, 2) * p.j_w_y
        * ((vx - sp.sqrt(p.d**2 + (sp.sin(thl) * 2 * l_l)**2) * vw
            + sp.cos(thl) * 2 * l_l * vthl) / p.r)**2
        + sp.Rational(1, 2) * p.j_w_y
        * ((vx + sp.sqrt(p.d**2 + (sp.sin(thr) * 2 * l_r)**2) * vw
            + sp.cos(thr) * 2 * l_r * vthr) / p.r)**2
        + sp.Rational(1, 2) * p.j_w_z * vw**2 * 2
        + sp.Rational(1, 2) * p.j_l_y * vthl**2
        + sp.Rational(1, 2) * p.m_l
        * ((vx - sp.sqrt(p.d**2 + (sp.sin(thl) * (2 * l_l - c_h1))**2) * vw
            + sp.cos(thl) * (2 * l_l - c_h1) * vthl)**2
           + (c_h1 * vthl * sp.sin(thl))**2)
        + sp.Rational(1, 2) * p.j_l_z * vw**2
        + sp.Rational(1, 2) * p.j_r_y * vthr**2
        + sp.Rational(1, 2) * p.m_r
        * ((vx + sp.sqrt(p.d**2 + (sp.sin(thr) * (2 * l_r - c_h2))**2) * vw
            + sp.cos(thr) * (2 * l_r - c_h2) * vthr)**2
           + (c_h2 * vthr * sp.sin(thr))**2)
        + sp.Rational(1, 2) * p.j_r_z * vw**2
        + sp.Rational(1, 2) * p.j_y * vth**2
        + sp.Rational(1, 2) * p.m
        * (vx**2
           + (p.l * vth * sp.sin(th))**2
           + ((l_l + l_r) * (vthl + vthr) / 2 * sp.sin((thl + thr) / 2))**2)
        + sp.Rational(1, 2) * p.J_z * vw**2)

    V = (p.m_l * p.g * sp.sqrt(c_h1**2 + c_x1**2) * sp.cos(thl - thp1)
         + p.m_r * p.g * sp.sqrt(c_h2**2 + c_x2**2) * sp.cos(thr - thp2)
         + p.m * p.g * ((l_l + l_r) * sp.cos((thl + thr) / 2)
                        + p.l * sp.cos(th + 0.0)))

    L = T - V

    # ---- 拉格朗日方程 ----
    def euler_lagrange(i: int):
        v = vels[i]
        pv = sp.diff(L, v)
        dpv = 0
        for j in range(5):
            dpv += sp.diff(pv, coords[j]) * vels[j]   # 位置项
            dpv += sp.diff(pv, vels[j]) * accs[j]     # 加速度项
        return dpv - sp.diff(L, coords[i])

    EL = [euler_lagrange(i) for i in range(5)]

    # 与原 MATLAB 的 5 个方程一一对应（写成 F = 0）
    F = [
        (M_l3 + M_r3) / p.r        - EL[0],   # eq1: x
        (-M_l3 + M_r3) / p.r * p.d - EL[1],   # eq2: w
        -(M_l2 + M_r2)             - EL[2],   # eq3: theta
        (M_l2 - M_l3)              - EL[3],   # eq4: theta_l
        (M_r2 - M_r3)              - EL[4],   # eq5: theta_r
    ]

    # ---- 平衡点势能：对 (θ, θl, θr) 的驻点 ----
    # 注意 V_full 里 θ 只出现在机身重力项 m*g*l*cos(θ)：
    #   l != 0 → dV/dθ = -m*g*l*sin(θ)，θ 由平衡方程确定（0 或 π）；
    #   l == 0 → dV/dθ ≡ 0，θ 是自由坐标，模型不给出唯一值（见 equilibrium.py）。
    th1s, thl1, thr1 = sp.symbols('theta_1 theta_l1 theta_r1')
    V_full = (p.m_l * p.g * sp.sqrt(c_h1**2 + c_x1**2) * sp.cos(thl1 - thp1)
              + p.m_r * p.g * sp.sqrt(c_h2**2 + c_x2**2) * sp.cos(thr1 - thp2)
              + p.m * p.g * ((l_l + l_r) * sp.cos((thl1 + thr1) / 2)
                             + p.l * sp.cos(th1s + 0.0)))

    # ---- 隐式形式 d(M_eq * u)/dz（仿射修正项）----
    u_syms: List[sp.Symbol] = list(sp.symbols('u1 u2 u3 u4 u5'))
    M_eq = sp.Matrix(F).jacobian(accs)                      # 5x5
    dM_eq_u_dz = (M_eq * sp.Matrix(u_syms)).jacobian(state)  # 5x10

    # ---- 一次性 lambdify（cse 展开，后续采样点直接调用）----
    return SymbolicModel(
        F_fun=sp.lambdify((state, accs, Ms, geom_syms), F,
                          modules='numpy', cse=True),
        dF_dz=sp.lambdify((state, accs, Ms, geom_syms),
                          sp.Matrix(F).jacobian(state), modules='numpy', cse=True),
        dF_da=sp.lambdify((state, accs, Ms, geom_syms),
                          sp.Matrix(F).jacobian(accs), modules='numpy', cse=True),
        dF_dM=sp.lambdify((state, accs, Ms, geom_syms),
                          sp.Matrix(F).jacobian(Ms), modules='numpy', cse=True),
        dM_eq_u_fun=sp.lambdify((state, accs, Ms, geom_syms, u_syms),
                                dM_eq_u_dz, modules='numpy', cse=True),
        V_balance=sp.lambdify((th1s, thl1, thr1, geom_syms), V_full,
                              modules='numpy', cse=True),
        dV_dq=sp.lambdify((th1s, thl1, thr1, geom_syms),
                          [sp.diff(V_full, th1s),
                           sp.diff(V_full, thl1), sp.diff(V_full, thr1)],
                          modules='numpy', cse=True),
    )
