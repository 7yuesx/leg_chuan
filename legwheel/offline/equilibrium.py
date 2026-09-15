"""
平衡点求解。

平衡点 = 势能 V(θ, θl, θr) 的驻点，三个角一起解：
    ∂V/∂θ = ∂V/∂θl = ∂V/∂θr = 0
其中 ∂V/∂θ 来自机身重力项 m*g*l*cos(θ)。当前 config 里 l = 0（机身质心在髋轴线
上），该项恒为 0，θ 是自由坐标，模型不给出唯一值，此时取配置值 theta_p；
l != 0 时 θ 被真正解出（从 theta_p 出发得到 0 或 π）。腿角部分与旧实现一致，
数值求解失败（残差超阈值）时退回实测拟合曲线 theta_balance_fit。

与原 compute_fit.py 的区别只有一处：偏导改由符号求导（Sp.diff）后 lambdify 得到，
不再用 eps=1e-6 的有限差分。原实现的差分噪声约 1e-9，逼得残差阈值只能放在 1e-7；
符号梯度是精确的，阈值保持 1e-7 不变即可稳定通过。

数值求解失败（残差超阈值）时退回实测拟合曲线 theta_balance_fit。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import fsolve

from .config import PARAMS
from .geometry import geometry
from .types import Context, Params

# ==================== 实测平衡腿角兜底 ====================
# 来自仿真测量（sim_leg_angle.py）：机器人静止时的腿角按 Leg.forward 定义
#   theta = -atan2(x_c, y_c) + pitch
# 对 8 个腿长特征点测得的稳态均值做 4 次加权拟合，系数单位弧度（高次到低次）。
_THETA_MEAS_RANGE = (0.1187, 0.3515)
_THETA_FIT_COEFF = np.array([
    87.88538943, -110.36430963, 52.12496719, -11.17598776, 0.97426381])

# 残差阈值：符号梯度下 1e-7 可以稳定满足
RESIDUAL_TOL = 1e-7

# 判断 ∂V/∂θ 是否恒为 0（θ 自由）时用的探测角
_THETA_PROBES = (-0.3, 0.0, 0.3)


def theta_balance_fit(L: float) -> float:
    """实测平衡腿角 θ(L)，弧度。范围外取端点。"""
    L = float(L)
    if L < _THETA_MEAS_RANGE[0]:
        L = _THETA_MEAS_RANGE[0]
    elif L > _THETA_MEAS_RANGE[1]:
        L = _THETA_MEAS_RANGE[1]
    return float(np.polyval(_THETA_FIT_COEFF, L))


def _theta_is_free(theta_l: float, theta_r: float, gv, ctx: Context) -> bool:
    """
    ∂V/∂θ 是否恒为 0。

    机身重力项是 m*g*l*cos(θ)，l = 0 时该项整体消失，θ 成为自由坐标（模型既
    不把它推向 0 也不推向 π）。这里在几个探测角上求值判断，不依赖 l 这个具体
    参数名，模型以后若多出别的 θ 耦合项也能自动识别。
    """
    dV = ctx.model.dV_dq
    worst = 0.0
    for t in _THETA_PROBES:
        grad = np.asarray(dV(t, theta_l, theta_r, gv), dtype=float).ravel()
        worst = max(worst, abs(float(grad[0])))
    return worst < 1e-9


def _solve_body_pitch(theta_p: float, theta_l: float, theta_r: float,
                      gv, ctx: Context, notes: List[str]) -> float:
    """
    机体俯仰平衡角 θ。

    ∂V/∂θ ≡ 0（l = 0）时模型无法确定 θ，取配置值 theta_p（与旧版线性化点一致，
    也让 theta_coeff 是干净的零多项式而不是求解噪声）；否则与腿角联立求解。
    """
    if _theta_is_free(theta_l, theta_r, gv, ctx):
        notes.append(f"提示：∂V/∂θ ≡ 0（机身质心无水平偏置，l={ctx.params.l:g}），"
                     f"θ 为自由坐标，取 theta_p={theta_p:g} 作为平衡俯仰角")
        return float(theta_p)

    dV = ctx.model.dV_dq

    def grad(q_a: np.ndarray) -> np.ndarray:
        return np.asarray(dV(q_a[0], q_a[1], q_a[2], gv), dtype=float).reshape(3)

    sol, info, ier, msg = fsolve(grad, [theta_p, theta_l, theta_r], maxfev=400,
                                 xtol=1e-12, full_output=True)
    res = float(np.max(np.abs(np.asarray(info.get("fvec", grad(sol))).ravel())))
    if res <= RESIDUAL_TOL:
        return float(sol[0])

    notes.append(f"WARNING: 机体俯仰平衡角求解残差未达标 max|dV|={res:.3e}，"
                 f"取 theta_p={theta_p:g}")
    return float(theta_p)


def solve_equilibrium(l_l: float, l_r: float,
                      ctx: Context) -> Tuple[float, float, float, List[str]]:
    """
    求解 (θ, θl, θr)。返回 (theta, theta_l, theta_r, notes)。
    结果按 (l_l, l_r) 缓存在 ctx 上，同一工作点只解一次。
    """
    key = (float(l_l), float(l_r))
    cached = ctx._balance_cache.get(key)
    if cached is not None:
        theta, theta_l, theta_r, note = cached
        return theta, theta_l, theta_r, ([note] if note else [])

    p: Params = ctx.params
    gv = geometry(l_l, l_r, p)
    dV = ctx.model.dV_dq                       # 精确符号梯度
    theta_p = p.theta_p
    notes: List[str] = []

    def grad(q_a: np.ndarray) -> np.ndarray:
        # 先只解腿角（θ 固定为 theta_p），再单独确定 θ：θ 与腿角在模型里
        # 不耦合时这样不会引入 θ 求解噪声，耦合时 _solve_body_pitch 会联立修正。
        return np.asarray(dV(theta_p, q_a[0], q_a[1], gv),
                          dtype=float).ravel()[1:]

    sol, info, ier, msg = fsolve(grad, [0.0, 0.0], maxfev=200,
                                 xtol=1e-12, full_output=True)
    res = float(np.max(np.abs(np.asarray(info.get("fvec", grad(sol))).ravel())))

    if res <= RESIDUAL_TOL:
        theta_l, theta_r = float(sol[0]), float(sol[1])
        if ier != 1:
            # fsolve 的返回码只描述迭代过程有没有"进展"，不描述解的质量。
            # 残差已达判据时（常见于 ier=5：十次迭代无改善）按残差接受，仅作提示。
            notes.append(f"提示：fsolve 返回码 ier={ier}，但残差 max|dV|={res:.1e} "
                         f"已达判据，按残差接受该解（L=({l_l:.4f}, {l_r:.4f})）")
    else:
        # 兜底：残差未达标时退回实测拟合值，并告警
        theta_l = theta_balance_fit(l_l)
        theta_r = theta_balance_fit(l_r)
        notes.append(f"WARNING: 平衡点数值求解残差未达标 L=({l_l:.4f}, {l_r:.4f}) "
                     f"max|dV|={res:.3e}，退回实测拟合值 "
                     f"theta=({theta_l:.4f}, {theta_r:.4f})")

    theta = _solve_body_pitch(theta_p, theta_l, theta_r, gv, ctx, notes)

    ctx._balance_cache[key] = (theta, theta_l, theta_r,
                               notes[-1] if notes else None)
    return theta, theta_l, theta_r, notes
