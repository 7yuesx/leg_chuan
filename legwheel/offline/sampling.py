"""
采样：在 (l_l, l_r) 网格上逐个求 Sample。

一个样本 = 线性化 → 预计算 cache → LESO 增益 → 收集全部待拟合矩阵。
支持可选并行（Linux 下用 fork，子进程通过 initializer 继承 ctx，不需要 pickle
lambdify 出来的函数）。默认串行，输出顺序确定、便于和旧结果逐点对比。
"""

from __future__ import annotations

import multiprocessing as _mp
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .precompute import leso_gain, precompute_cache
from .types import Context, Sample


def compute_sample(l_l: float, l_r: float, ctx: Context) -> Sample:
    """单个工作点的全部待拟合矩阵。"""
    from .linearization import linearize_point      # 延迟导入，避免符号层开销提前触发

    lp = linearize_point(l_l, l_r, ctx)
    cache = precompute_cache(lp.A, lp.B, ctx.cfg.Q, ctx.cfg.R, ctx.cfg.rho,
                             cache_sign=ctx.cfg.cache_sign, f=lp.f)
    Le = leso_gain(lp.A, lp.B, ctx.cfg.obs_Qo_state, ctx.cfg.obs_Qo_d,
                   ctx.cfg.obs_Ro, ctx.cfg.nx, ctx.cfg.nu)
    return Sample(l_l=float(l_l), l_r=float(l_r),
                  A=lp.A, B=lp.B, P=cache.P, K=cache.K, Quu_inv=cache.Quu_inv,
                  AmBKt=cache.AmBKt, Le=Le,
                  theta=lp.theta_body,
                  theta_l=lp.theta_l, theta_r=lp.theta_r,
                  f=lp.f, APf=cache.APf, BPf=cache.BPf,
                  notes=tuple(lp.notes))


# ---- 并行 worker：ctx 由 initializer 注入子进程，避免 pickle lambdify 结果 ----
_WORKER_CTX: Optional[Context] = None


def _init_worker(ctx: Context) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _worker(task: Tuple[float, float]) -> Optional[Sample]:
    try:
        return compute_sample(task[0], task[1], _WORKER_CTX)
    except Exception:                     # 失败点在复查阶段统一报告
        return None


def sample_grid(l_vals: Sequence[float], ctx: Context, n_jobs: int = 1,
                on_error: Optional[Callable[[float, float, Exception], None]] = None,
                ) -> Tuple[List[Sample], List[Tuple[float, float]]]:
    """
    遍历网格，返回 (成功样本列表, 缺失点列表)。
    与原实现不同：失败点会被记录下来而不是静默跳过。
    """
    tasks = [(float(ll), float(lr)) for ll in l_vals for lr in l_vals]
    samples: List[Sample] = []
    missing: List[Tuple[float, float]] = []

    if n_jobs and n_jobs > 1:
        mp_ctx = _mp.get_context("fork")
        with mp_ctx.Pool(processes=n_jobs, initializer=_init_worker,
                         initargs=(ctx,)) as pool:
            results = pool.map(_worker, tasks, chunksize=1)
        for task, res in zip(tasks, results):
            if res is None:
                missing.append(task)
            else:
                samples.append(res)
        return samples, missing

    for ll, lr in tasks:
        try:
            samples.append(compute_sample(ll, lr, ctx))
        except Exception as exc:          # noqa: BLE001 —— 离线批处理，收集而不是中断
            missing.append((ll, lr))
            if on_error is not None:
                on_error(ll, lr, exc)
    return samples, missing


def check_completeness(samples: Sequence[Sample], l_vals: Sequence[float],
                       missing: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    """采样完整性校验：期望点数 vs 实际点数。"""
    expected = len(l_vals) ** 2
    return {
        "expected": expected,
        "success": len(samples),
        "missing": list(missing),
        "complete": len(missing) == 0 and len(samples) == expected,
    }
