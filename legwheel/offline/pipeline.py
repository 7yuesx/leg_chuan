"""
流程编排：把 config / symbolic_model / geometry / equilibrium / linearization /
precompute / sampling / fitting / codegen 串起来。原 compute_fit.py 的 main() 在这里。

对外门面：
    build_context()                 构造一次离线计算的全部上下文
    compute_system(l_l, l_r, ctx)   单个工作点的全部矩阵（保留旧函数名做兼容）
    run(...)                        跑完整条流水线并落盘
    main(argv)                      命令行入口（python -m offline）
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .codegen import write_all
from .config import MPC, PARAMS, leg_length_grid
from .fitting import (MATRIX_ORDER, make_feature_spec, residual_report,
                      self_check)
from .fitting import fit_matrices
from .geometry import clamped_leg_lengths, reset_diagnostics
from .sampling import check_completeness, compute_sample, sample_grid
from .symbolic_model import build_model
from .types import Context, FitResult, MpcConfig, Params, Sample

DEFAULT_OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_context(cfg: MpcConfig = MPC, params: Params = PARAMS) -> Context:
    """构造上下文：符号模型只建一次，后续所有采样点复用。"""
    return Context(params=params, cfg=cfg, model=build_model(params))


def compute_system(l_l: float, l_r: float,
                   ctx: Optional[Context] = None) -> Sample:
    """兼容旧版 compute_fit.compute_system(l_l, l_r)：返回该工作点的 Sample。"""
    if ctx is None:
        ctx = build_context()
    return compute_sample(l_l, l_r, ctx)


def _selfcheck_points(l_vals: Sequence[float], n: int) -> List[Tuple[float, float]]:
    """取相邻采样点的中点作为回代自检点（不在采样网格上）。"""
    mids = [0.5 * (l_vals[i] + l_vals[i + 1]) for i in range(len(l_vals) - 1)]
    if not mids:
        return []
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        a = mids[i % len(mids)]
        b = mids[(i * 2 + 1) % len(mids)]
        pts.append((round(a, 6), round(b, 6)))
    return pts


def run(cfg: MpcConfig = MPC, params: Params = PARAMS,
        out_dir: Optional[str] = None, n_jobs: int = 1,
        degree: Optional[int] = None, header: bool = True,
        n_check_points: int = 8, verbose: bool = True,
        write: bool = True) -> FitResult:
    """跑完整条流水线。out_dir 为 None 时写到仓库根目录。"""
    out_dir = out_dir or DEFAULT_OUT_DIR
    if degree is not None and degree != cfg.fit_degree:
        cfg = MpcConfig(**{**cfg.__dict__, "fit_degree": int(degree)})

    reset_diagnostics()
    ctx = build_context(cfg, params)
    l_vals = leg_length_grid(cfg)

    if verbose:
        print(f"采样区间: l_l, l_r = {l_vals[0]} ~ {l_vals[-1]} "
              f"(步长 {l_vals[1] - l_vals[0]})，共 {len(l_vals)} 档")
        print(f"拟合次数: {cfg.fit_degree}（特征数 {len(make_feature_spec(cfg.fit_degree, cfg.l_min, cfg.l_max).exponents)}）")
        print("预计算: TinyMPC 形式（权重 Q+rho*I、R+rho*I，scipy DARE 直接精确解）")
        print(f"符号约定: {cfg.cache_sign}（positive = P 正定、K 正号，u = -K x）")
        print(f"线性化点: {cfg.linearization_point}")
        print(f"几何模型: 解析两连杆（不使用 MuJoCo）")
        print("正在逐个样本计算 A, B, f, P, K, Quu_inv, AmBKt, Le, theta ...")

    def _on_error(ll, lr, exc):
        print(f"  跳过 ({ll:.4f}, {lr:.4f}): {exc}")

    samples, missing = sample_grid(l_vals, ctx, n_jobs=n_jobs, on_error=_on_error)
    completeness = check_completeness(samples, l_vals, missing)

    if verbose:
        print(f"成功样本数: {len(samples)} / {completeness['expected']}")
        fmax = max(float(np.max(np.abs(s.f))) for s in samples)
        apf = max(float(np.max(np.abs(s.APf))) for s in samples)
        bpf = max(float(np.max(np.abs(s.BPf))) for s in samples)
        print(f"仿射项量级: |f|max={fmax:.3e}, |APf|max={apf:.3e}, |BPf|max={bpf:.3e}")
        if not completeness["complete"]:
            print(f"  WARNING: 缺失采样点 {len(missing)} 个: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        clamped = clamped_leg_lengths()
        if clamped:
            print(f"  WARNING: 腿长超出机构可达范围并已裁剪（arccos 裁剪）: {clamped}")
        for note in dict.fromkeys(n for s in samples for n in s.notes):
            print(f"  {note}")
        print("")

    spec = make_feature_spec(cfg.fit_degree, cfg.l_min, cfg.l_max,
                             normalize=cfg.normalize_features)
    res = fit_matrices(samples, spec)

    if verbose:
        print(f"# 特征顺序: {', '.join(spec.labels)} (x = l_l, y = l_r)")
        print(f"# 系数数组按行主序排列，每行 {spec.nfeat} 个系数，"
              f"可 reshape 成 (rows, cols, {spec.nfeat})\n")

        # 逐元素残差报告（拟合质量的可观测性）
        rep = residual_report(res, samples)
        print("拟合残差（逐元素相对误差）:")
        for name in MATRIX_ORDER:
            r = rep[name]
            if r["scale"] < 1e-9:
                # 仿射项在平衡点线性化下是数值零，相对残差没有意义
                print(f"  {name:>8s}: ≈0（量级 {r['scale']:.1e}，相对残差无意义）")
            else:
                print(f"  {name:>8s}: max_rel={r['max_rel']:.3e} "
                      f"(展开下标 {r['worst_index']}), matrix_rel={r['matrix_rel']:.3e}")
        print("")

    if n_check_points:
        pts = _selfcheck_points(l_vals, n_check_points)
        errs = self_check(res, ctx, pts)
        if verbose and errs:
            print(f"回代自检（{len(pts)} 个非网格点，解析值 vs 拟合重建）:")
            for name in MATRIX_ORDER:
                print(f"  {name:>8s}: rel_err={errs[name]:.3e}")
            print("")

    if write:
        paths = write_all(out_dir, res, cfg, params, len(samples), header=header)
        if verbose:
            for key, path in paths.items():
                print(f"已生成 {key}: {path}")
            print("")

    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m offline",
        description="腿足式平衡轮 MPC 离线矩阵拟合（解析几何，不依赖 MuJoCo）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="产物输出目录（默认：仓库根目录）")
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行进程数，1 为串行（默认 1）")
    ap.add_argument("--degree", type=int, default=None,
                    help="拟合多项式总次数（默认取 config.MPC.fit_degree）")
    ap.add_argument("--check-points", type=int, default=8,
                    help="回代自检点数，0 表示跳过")
    ap.add_argument("--no-header", action="store_true",
                    help="不生成 mpc_offline_data.hpp")
    ap.add_argument("--no-write", action="store_true",
                    help="只计算与报告，不写任何文件")
    ap.add_argument("--quiet", action="store_true", help="减少输出")
    args = ap.parse_args(argv)

    run(out_dir=args.out_dir, n_jobs=args.jobs, degree=args.degree,
        header=not args.no_header, n_check_points=args.check_points,
        verbose=not args.quiet, write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
