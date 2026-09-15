"""
腿足式平衡轮 MPC 离线计算包。

流水线（依赖方向单向，无环）：

    config ─┬─> symbolic_model ─┬─> equilibrium ──> linearization ─┐
            ├─> geometry ───────┘                                  ├─> sampling ─> fitting ─> codegen
            └─> (Q/R/rho/N/Ts) ─────────────────> precompute ──────┘            (pipeline 串起来)

各模块职责见同目录下对应文件的模块 docstring。对外只需要这两个入口：

    from offline import build_context, compute_system, run

几何模型使用解析两连杆公式（本包不依赖 mujoco）。
"""

from .config import PARAMS, MPC, leg_length_grid
from .pipeline import build_context, compute_system, run, main
from .types import (Params, MpcConfig, LegGeom, LinearPoint, CacheMatrices,
                    Sample, FitResult, FeatureSpec, Context)

__all__ = [
    "PARAMS", "MPC", "leg_length_grid",
    "build_context", "compute_system", "run", "main",
    "Params", "MpcConfig", "LegGeom", "LinearPoint", "CacheMatrices",
    "Sample", "FitResult", "FeatureSpec", "Context",
]
