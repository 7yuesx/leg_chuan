"""
腿机构几何：由腿长 L 解析求出腿质心在髋→轮轴坐标系下的位置。

解析两连杆方法（原 compute_fit.py 的 _geometry_analytic）：
  1) 由 l_b、l_s 与 L 的余弦定理求髋关节角 H；
  2) c_x = l_b*l_s*sin(H) / (2L)，c_h 由连杆质量加权得到；
  3) thp = atan2(c_x, c_h)。

本包不依赖 MuJoCo。L 超出机构可达范围（三角不等式不成立）时对 arccos 做裁剪，
并按唯一腿长计数，供流程末尾报告。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from .config import PARAMS
from .types import LegGeom, Params

# 超出可达范围的唯一腿长计数（诊断用，与工作区状态无关的纯计数）
_CLAMPED: List[float] = []


@lru_cache(maxsize=None)
def _leg_geometry_cached(L: float, l_b: float, l_s: float,
                         m_b: float, m_s: float) -> LegGeom:
    """按 (L, 机构参数) 缓存的解析几何；参数都是浮点，可直接做缓存键。"""
    ratio = (l_b**2 + l_s**2 - L**2) / (2.0 * l_b * l_s)
    clamped = not (-1.0 <= ratio <= 1.0)
    if clamped:
        if L not in _CLAMPED:
            _CLAMPED.append(L)
        ratio = min(1.0, max(-1.0, ratio))

    H = np.arccos(ratio)
    c_x = l_b * l_s * np.sin(H) / (2.0 * L)
    # c_h = L/2 / (1 + m_s/m_b) + (l_s^2 + L^2 - l_b^2) / (4L)
    c_h = (L / 2.0) / (1.0 + m_s / m_b) + (l_s**2 + L**2 - l_b**2) / (4.0 * L)
    thp = np.arctan2(c_x, c_h)
    return LegGeom(c_h=float(c_h), c_x=float(c_x), thp=float(thp), clamped=clamped)


def leg_geometry(L: float, p: Params = PARAMS) -> LegGeom:
    """单条腿长 L 的解析几何。"""
    return _leg_geometry_cached(float(L), p.l_b, p.l_s, p.m_b, p.m_s)


def geometry(l_l: float, l_r: float, p: Params = PARAMS) -> np.ndarray:
    """
    返回符号模型需要的 8 元素几何向量：
        [l_l, l_r, c_h1, c_h2, c_x1, c_x2, thp1, thp2]
    其中前两项是"腿长的一半"（符号模型里 2*l - c_h 的写法依赖这一点），
    与 compute_fit.py 的 geometry() 返回值定义保持一致。
    """
    gl = leg_geometry(l_l, p)
    gr = leg_geometry(l_r, p)
    out = np.empty(8)
    out[0] = float(l_l) / 2.0
    out[1] = float(l_r) / 2.0
    out[2], out[3] = gl.c_h, gr.c_h
    out[4], out[5] = gl.c_x, gr.c_x
    out[6], out[7] = gl.thp, gr.thp
    return out


def clamped_leg_lengths() -> tuple:
    """返回发生裁剪的唯一腿长（用于流程末尾告警）。"""
    return tuple(_CLAMPED)


def reset_diagnostics() -> None:
    _CLAMPED.clear()
