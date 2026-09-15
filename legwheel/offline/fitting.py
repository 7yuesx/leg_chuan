"""
拟合：把每个矩阵元素拟合成 (l_l, l_r) 的二元多项式。

特征基与原实现完全一致，保证 coeff.py 的系数顺序不变：
    [1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3, ...]
（按总次数分组，组内 x 的指数从高到低）

本模块只依赖 numpy：输入 Sample 列表，输出 FitResult。
额外提供两项原实现没有的能力：
  * residual_report：逐元素相对残差（A 里 1.0 与 1e-13 共存时，全矩阵范数看不出问题）
  * self_check：在非采样网格点上回代重建矩阵，与解析值比对
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .types import Context, FeatureSpec, FitResult, Sample

# 落盘顺序，必须与 coeff.py / user_class.py 的消费顺序一致
MATRIX_ORDER: Tuple[str, ...] = (
    "A", "B", "f", "P", "K", "Quu_inv", "AmBK_T", "APf", "BPf",
    "Le", "theta", "theta_l", "theta_r")


def make_feature_spec(degree: int, l_min: float, l_max: float,
                      normalize: bool = False) -> FeatureSpec:
    """构造特征基定义。normalize=True 会引入中心化/归一化（需 runtime 同步）。"""
    # 常数项在前，随后按总次数分组、组内 x 的指数从高到低
    exponents: List[Tuple[int, int]] = [(0, 0)]
    labels: List[str] = ["1"]
    for deg in range(1, degree + 1):
        for i in range(deg, -1, -1):        # x 的指数从高到低
            exponents.append((i, deg - i))
            if i == 0:
                labels.append(f'y^{deg}' if deg > 1 else 'y')
            elif i == deg:
                labels.append(f'x^{deg}' if deg > 1 else 'x')
            else:
                labels.append(f'x^{i}*y^{deg-i}')

    center, scale = 0.0, 1.0
    if normalize:
        center = 0.5 * (l_min + l_max)
        half = 0.5 * (l_max - l_min)
        scale = half if half > 0 else 1.0
    return FeatureSpec(degree=degree, exponents=tuple(exponents),
                       labels=tuple(labels), center=center, scale=scale)


def build_features(spec: FeatureSpec, l_l, l_r) -> np.ndarray:
    """特征矩阵。l_l / l_r 可以是标量或数组，返回 (N, nfeat)。"""
    x = (np.asarray(l_l, dtype=float) - spec.center) / spec.scale
    y = (np.asarray(l_r, dtype=float) - spec.center) / spec.scale
    # 单项式按指数表逐个求值（(0,0) 即常数 1 列），返回 (N, nfeat)
    cols = [(x ** i) * (y ** j) for i, j in spec.exponents]
    return np.column_stack(cols).astype(float)


def sample_matrix(name: str, s: Sample) -> np.ndarray:
    """取某个样本里指定矩阵的 (rows, cols) 形式。"""
    if name == "theta":
        return np.array([[s.theta]])
    if name == "theta_l":
        return np.array([[s.theta_l]])
    if name == "theta_r":
        return np.array([[s.theta_r]])
    return np.asarray(getattr(s, {"AmBK_T": "AmBKt"}.get(name, name)), dtype=float)


def fit_matrices(samples: Sequence[Sample], spec: FeatureSpec) -> FitResult:
    """对每个矩阵的每个元素做最小二乘，返回 FitResult。"""
    if not samples:
        raise RuntimeError("没有成功样本，无法拟合")

    x = np.array([s.l_l for s in samples], dtype=float)
    y = np.array([s.l_r for s in samples], dtype=float)
    X = build_features(spec, x, y)                 # (N, nfeat)

    coeffs_f64: Dict[str, np.ndarray] = {}
    coeffs_f32: Dict[str, np.ndarray] = {}
    mat_info: Dict[str, Tuple[int, int, int]] = {}

    for name in MATRIX_ORDER:
        m0 = sample_matrix(name, samples[0])
        rows, cols = m0.shape
        dim = rows * cols
        # 行主序展平 → Y: (dim, N)
        Y = np.array([sample_matrix(name, s).reshape(-1, order='C')
                      for s in samples], dtype=float).T
        coeff = np.linalg.lstsq(X, Y.T, rcond=None)[0].T      # (dim, nfeat)
        coeffs_f64[name] = coeff
        coeffs_f32[name] = coeff.astype(np.float32)
        mat_info[name] = (dim, rows, cols)

    return FitResult(coeffs=coeffs_f32, coeffs_f64=coeffs_f64,
                     mat_info=mat_info, spec=spec, samples=list(samples))


def residual_report(res: FitResult, samples: Sequence[Sample] = None) -> Dict[str, Dict[str, float]]:
    """
    逐元素相对残差报告。

    返回 {矩阵名: {"max_rel": 该矩阵所有元素里最差的相对残差,
                   "worst_index": 行主序下标,
                   "matrix_rel": 用矩阵最大范数归一化的整体相对残差}}
    """
    samples = list(samples if samples is not None else res.samples)
    x = np.array([s.l_l for s in samples], dtype=float)
    y = np.array([s.l_r for s in samples], dtype=float)
    X = build_features(res.spec, x, y)

    report: Dict[str, Dict[str, float]] = {}
    for name in MATRIX_ORDER:
        Y = np.array([sample_matrix(name, s).reshape(-1, order='C')
                      for s in samples], dtype=float).T
        pred = res.coeffs_f64[name] @ X.T                      # (dim, N)
        err = np.abs(pred - Y)
        scale = np.maximum(np.max(np.abs(Y), axis=1), 1e-12)   # 逐元素尺度
        per_elem = np.max(err / scale[:, None], axis=1)
        worst = int(np.argmax(per_elem))
        report[name] = {
            "max_rel": float(per_elem[worst]),
            "worst_index": worst,
            "matrix_rel": float(np.max(err) / max(np.max(np.abs(Y)), 1e-12)),
            "scale": float(np.max(np.abs(Y))),
        }
    return report


def self_check(res: FitResult, ctx: Context, points: Iterable[Tuple[float, float]],
               ) -> Dict[str, float]:
    """
    回代自检：在给定（通常不在采样网格上的）工作点重新解析计算，与拟合重建比对。
    返回 {矩阵名: 相对误差}。
    """
    from .sampling import compute_sample           # 延迟导入避免环

    points = list(points)
    if not points:
        return {}
    feats = np.column_stack([build_features(res.spec, ll, lr)[0] for ll, lr in points])
    out: Dict[str, float] = {}
    for name in MATRIX_ORDER:
        exact = np.array([sample_matrix(name, compute_sample(ll, lr, ctx)).reshape(-1, order='C')
                          for ll, lr in points], dtype=float).T        # (dim, P)
        pred = res.coeffs_f64[name] @ feats                            # (dim, P)
        denom = max(np.max(np.abs(exact)), 1e-12)
        out[name] = float(np.max(np.abs(pred - exact)) / denom)
    return out
