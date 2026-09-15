"""
数据结构契约。对应 TinyMPC 的 types.hpp：把模块之间传递的数据形状固定下来，
每个模块只依赖这里的类型，不依赖别的模块的内部实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:                      # 仅用于类型标注，避免运行期循环 import
    from .symbolic_model import SymbolicModel


@dataclass(frozen=True)
class Params:
    """物理参数（对应 compute_fit.py 第 1 节）。"""

    m: float          # 机身质量
    m_l: float        # 左腿连杆总质量
    m_r: float        # 右腿连杆总质量
    m_w: float        # 轮质量
    j_z: float
    j_y: float
    j_l_z: float
    j_l_y: float
    j_r_z: float
    j_r_y: float
    d: float          # 轮距半宽
    j_w_z: float
    j_w_y: float
    J_z: float
    l: float          # 机身质心到髋的水平距离
    r: float          # 轮半径
    g: float
    l_b: float        # 连杆长
    l_s: float        # 连杆长
    m_b: float
    m_s: float
    theta_p: float


@dataclass(frozen=True)
class MpcConfig:
    """MPC / 观测器配置 + 采样与拟合设置（对应 TinyMPC 的 tiny_set_default_settings）。"""

    nx: int                                  # 状态维数
    nu: int                                  # 输入维数
    N: int                                   # 预测步数
    Ts: float                                # 采样周期
    rho: float                               # ADMM 罚参数
    Q: np.ndarray                            # (nx, nx)
    R: np.ndarray                            # (nu, nu)
    obs_Qo_state: float                      # LESO：状态通道过程噪声
    obs_Qo_d: float                          # LESO：扰动通道过程噪声
    obs_Ro: float                            # LESO：测量噪声
    fit_degree: int                          # 拟合多项式总次数
    l_min: float                             # 采样区间起点（含）
    l_max: float                             # 采样区间终点（含）
    l_step: float                            # 采样步长
    normalize_features: bool = False         # 特征基是否做中心化/归一化
    cache_sign: str = "positive"             # "positive" = TinyMPC 正号约定；"negative" = 旧负号约定
    # 线性化点："equilibrium" = 解平衡腿角；"zero" = 直接用 theta_l = theta_r = 0
    linearization_point: str = "equilibrium"


@dataclass(frozen=True)
class LegGeom:
    """单条腿的解析几何量（腿质心在髋→轮轴坐标系下的投影）。"""

    c_h: float        # 沿髋→轮轴线
    c_x: float        # 垂直髋→轮轴线
    thp: float        # 腿质心相位角
    clamped: bool = False   # 该腿长超出机构可达范围（已做裁剪）时为 True


@dataclass
class LinearPoint:
    """单个腿长工作点线性化 + 离散化的结果。"""

    A: np.ndarray            # (nx, nx) 离散状态矩阵
    B: np.ndarray            # (nx, nu) 离散输入矩阵
    theta_body: float
    theta_l: float
    theta_r: float
    A_c: np.ndarray          # (nx, nx) 连续时间状态矩阵
    B_c: np.ndarray          # (nx, nu) 连续时间输入矩阵
    f_c: np.ndarray          # (nx, 1) 连续时间仿射项 -Fa^-1 * F(p)
    f: np.ndarray            # (nx, 1) 离散仿射项（ZOH），即 TinyMPC 的 fdyn
    notes: List[str] = field(default_factory=list)


@dataclass
class CacheMatrices:
    """
    对应 TinyMPC 的 TinyCache：Pinf / Kinf / Quu_inv / AmBKt。

    符号约定与现有 compute_fit.py 一致（沿用负号约定）：
    cache_sign="positive"（默认，与 TinyMPC 一致）：P = Pinf、K = Kinf（u = -K x）
    cache_sign="negative"（旧工程约定）：P = -Pinf、K = -Kinf
    Quu_inv（TinyMPC 的 C1）与 AmBKt（C2）在两种约定下相同。详见 precompute.py。
    """

    P: np.ndarray            # (nx, nx) 正定（positive）或负定（negative）
    K: np.ndarray            # (nu, nx) 正号增益（positive）或负号增益（negative）
    Quu_inv: np.ndarray      # (nu, nu)，即 TinyMPC 的 Quu_inv / C1
    AmBKt: np.ndarray        # (nx, nx)
    sign: str = "positive"   # 落盘符号约定
    APf: np.ndarray = None   # (nx, 1) TinyMPC 的 APf = AmBKt * Pinf * fdyn
    BPf: np.ndarray = None   # (nu, 1) TinyMPC 的 BPf = B' * Pinf * fdyn


@dataclass
class Sample:
    """一个采样网格点上的全部待拟合矩阵。"""

    l_l: float
    l_r: float
    A: np.ndarray
    B: np.ndarray
    P: np.ndarray
    K: np.ndarray
    Quu_inv: np.ndarray      # 即 TinyMPC 的 Quu_inv / C1
    AmBKt: np.ndarray
    Le: np.ndarray           # (nx+nu, nx)
    theta: float             # 机体俯仰平衡角
    theta_l: float
    theta_r: float
    f: np.ndarray            # (nx, 1) 离散仿射项
    APf: np.ndarray          # (nx, 1)
    BPf: np.ndarray          # (nu, 1)
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureSpec:
    """拟合特征基的定义：指数表 + 可选的归一化常数。"""

    degree: int
    exponents: Tuple[Tuple[int, int], ...]   # ((i, j), ...)，对应 x^i * y^j
    labels: Tuple[str, ...]
    center: float = 0.0
    scale: float = 1.0

    @property
    def nfeat(self) -> int:
        return len(self.exponents)


@dataclass
class FitResult:
    """拟合结果：系数（float32 落盘 / float64 内部校验）+ 维度信息。"""

    coeffs: Dict[str, np.ndarray]            # name -> (dim, nfeat) float32
    coeffs_f64: Dict[str, np.ndarray]        # name -> (dim, nfeat) float64
    mat_info: Dict[str, Tuple[int, int, int]]  # name -> (dim, rows, cols)
    spec: FeatureSpec
    samples: List[Sample]


@dataclass
class Context:
    """一次离线计算的全部上下文（取代原先的模块级全局变量）。"""

    params: Params
    cfg: MpcConfig
    model: "SymbolicModel"
    notes: List[str] = field(default_factory=list)
    _balance_cache: Dict[Tuple[float, float],
                         Tuple[float, float, float, Optional[str]]] = field(
        default_factory=dict)

    def add_note(self, msg: str) -> None:
        self.notes.append(msg)
