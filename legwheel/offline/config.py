"""
固定参数与 MPC 配置。对应 TinyMPC 的 tiny_set_default_settings：
所有"旋钮"集中在这一处，其它模块只读取，不再各自硬编码常量。
"""

from __future__ import annotations

import numpy as np

from .types import MpcConfig, Params

# ==================== 物理参数 ====================
# 与 compute_fit.py 第 1 节完全一致；注释里保留了原文件中的备选数值。
PARAMS = Params(
    m=16.877574000000,          # 备选: 20.275132741229 / 17.275132741229
    m_l=1.113200000000 + 0.306873000000 + 0.058433000000
        + 0.042999000000 + 0.065877000000 + 0.010945000000,
    m_r=1.113200000000 + 0.306873000000 + 0.058433000000
        + 0.042999000000 + 0.065877000000 + 0.010945000000,
    m_w=0.490400000000,         # 备选: 0.353429173529
    j_z=0.359350481645,
    j_y=0.457285642113,
    j_l_z=0.00468513 + 2.48016e-03 + 9.09634e-05
        + 7.61359e-05 + 2.05523e-04 + 9.8695e-06,
    j_l_y=0.00641833 + 2.54050e-03 + 1.01653e-04
        + 7.83158e-05 + 2.11040e-04 + 1.0337e-05,
    j_r_z=0.00468513 + 2.48016e-03 + 9.09634e-05
        + 7.61359e-05 + 2.05523e-04 + 9.8695e-06,
    j_r_y=0.00641833 + 2.54050e-03 + 1.01653e-04
        + 7.83158e-05 + 2.11040e-04 + 1.0337e-05,
    d=0.2,                      # 备选: 0.165
    j_w_z=0.00034133,
    j_w_y=0.0005817,
    J_z=0.359350481645,         # = j_z
    l=0.0,
    r=0.06,
    g=9.81,
    l_b=0.21,                   # 备选: 0.215
    l_s=0.2462962962962,        # 备选: 0.258
    m_b=0.134820000000 + 0.155400000000 + 0.230811000000
        + 0.117180000000 + 0.236250000000,
    m_s=0.349772111111 + 0.353429173529,
    theta_p=0.0,
)

# ==================== 权重与配置 ====================
# 状态: [x, w, theta, theta_l, theta_r, x_dot, w_dot, theta_dot,
#        theta_dot_l, theta_dot_r]
_Q_DIAG = np.array([10.0, 10.0, 8000.0, 6000.0, 6000.0,
                    10.0, 10.0, 1.0, 1.0, 1.0])
_R_DIAG = np.array([5.0, 5.0, 10.0, 10.0])
MPC = MpcConfig(
    nx=10,
    nu=4,
    N=100,
    Ts=0.002,
    rho=2.0,
    Q=np.diag(_Q_DIAG),
    R=np.diag(_R_DIAG),
    obs_Qo_state=10.0,
    obs_Qo_d=10.0,
    obs_Ro=1000.0,
    fit_degree=3,               # 2 = 原 MATLAB 的二元二次；3 为当前使用值
    l_min=0.10,
    l_max=0.40,          # 采样区间上界（含端点）；机构解析可达范围 0.036~0.456
    l_step=0.02,         # 步长 → 0.10, 0.12, ..., 0.40 共 16 档
    normalize_features=False,   # True 会改变 runtime 特征构造，需同步改 user_class
    # 符号约定：positive = TinyMPC 的正号（P 正定、K 为正号增益，u = -K x）。
    # 必须与运行期 user_class.Mpc 的写法一致，改这里要同步改那里。
    cache_sign="positive",
    # 线性化点："equilibrium" 解平衡腿角（推荐）；"zero" 直接取 theta_l=theta_r=0。
    # 选 zero 且几何带质心偏置时，残差 F(p) != 0，仿射项 f 会明显非零。
    linearization_point="equilibrium",
)


def leg_length_grid(cfg: MpcConfig = MPC) -> np.ndarray:
    """
    采样腿长网格，**包含 l_max 端点**。

    用 linspace 而不是 np.arange：arange 不含右端点，会把区间偷偷缩短一档
    （旧版 l_max=0.30 实际只到 0.28）。点数由步长推出：
        n = round((l_max - l_min)/l_step) + 1
    """
    n = int(round((cfg.l_max - cfg.l_min) / cfg.l_step)) + 1
    return np.linspace(cfg.l_min, cfg.l_max, n)
