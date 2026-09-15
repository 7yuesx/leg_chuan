"""
代码生成：把拟合系数落盘。

三种产物：
  * write_coeff_py    → coeff.py（numpy 数组，供 leg_control.py 直接 import）
  * write_c_arrays    → coeff_c_arrays.txt（#define 形式的一维 float32 数组）
  * write_offline_header → mpc_offline_data.hpp（结构体风格，对齐 TinyMPC 的 tiny_data.hpp）

前两者的格式与旧 compute_fit.py 输出保持逐字一致，便于替换后直接 diff。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from .fitting import MATRIX_ORDER
from .types import FitResult, MpcConfig, Params


def _fmt_row(coeff_row: np.ndarray) -> str:
    """与旧实现一致：repr(float(float32 值))，保证是合法且无损的浮点字面量。"""
    return ", ".join(repr(float(v)) for v in coeff_row)


def write_coeff_py(path: str, res: FitResult, n_samples: int) -> str:
    """写 numpy 系数文件（coeff.py 的格式）。"""
    spec = res.spec
    feat_line = ", ".join(spec.labels)

    lines: List[str] = []
    lines.append("import numpy as np")
    lines.append("")
    lines.append(f"成功样本数: {n_samples}")
    lines.append("")
    lines.append(f"# 特征顺序: {feat_line} (x = l_l, y = l_r)")
    lines.append(f"# 系数数组按行主序排列，每行 {spec.nfeat} 个系数，"
                 f"可 reshape 成 (rows, cols, {spec.nfeat})")
    lines.append("")

    for name in MATRIX_ORDER:
        coeff = res.coeffs[name]
        dim, rows, cols = res.mat_info[name]
        lines.append(f"{name}_coeff = np.array([")
        for i in range(dim):
            comma = "," if i < dim - 1 else ""
            lines.append(f"    [{_fmt_row(coeff[i])}]{comma}")
        lines.append(f"], dtype=np.float32)  # ({rows}, {cols}, {spec.nfeat})，行主序")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines).rstrip("\n") + "\n")
    return path


def write_c_arrays(path: str, res: FitResult) -> str:
    """写 C 语言一维数组（行主序）文本。"""
    spec = res.spec
    feat_line = ", ".join(spec.labels)

    with open(path, 'w') as f:
        f.write("/* 系数矩阵，float32，按 C 语言一维数组（行主序）保存。\n")
        f.write(" * 特征顺序 (x = l_l, y = l_r): " + feat_line + "\n")
        f.write(f" * 每个数组元素个数 = rows*cols*nfeat，可 reshape 成 "
                f"(rows, cols, {spec.nfeat})。\n */\n\n")
        for name in MATRIX_ORDER:
            coeff = res.coeffs[name]
            dim, rows, cols = res.mat_info[name]
            f.write(f"/* {name}_coeff: {coeff.size} 个 float32 */\n")
            f.write(f"#define {name}_Coeffs_Def \\\n")
            f.write("    { \\\n")
            for i in range(dim):
                vals = ", ".join(f"{float(v)!r}f" for v in coeff[i])
                f.write(f"        {vals}, \\\n")
            f.write("    }\n\n")
    return path


def write_offline_header(path: str, res: FitResult, cfg: MpcConfig, params: Params) -> str:
    """
    结构体风格的 C++ 头文件（对齐 TinyMPC 的 tiny_data.hpp）。

    把维数、特征定义、归一化常数、以及各系数数组放在同一个文件里，
    C 端只需要包含它，不会和 Python 侧的常量脱节。
    """
    spec = res.spec
    with open(path, 'w') as f:
        f.write("/*\n * 由 offline 包自动生成的 MPC 离线系数（TinyMPC tiny_data 风格）。\n")
        f.write(" * 特征基: " + ", ".join(spec.labels) + "\n")
        f.write(f" * 采样点: {len(res.samples)} 个，(l_l, l_r) 二元 {spec.degree} 次\n */\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write("typedef float mpc_real_t;\n\n")
        f.write(f"#define MPC_NX {cfg.nx}\n")
        f.write(f"#define MPC_NU {cfg.nu}\n")
        f.write(f"#define MPC_HORIZON {cfg.N}\n")
        f.write(f"#define MPC_RHO {float(cfg.rho)!r}f\n")
        f.write(f"#define MPC_TS {float(cfg.Ts)!r}f\n")
        f.write(f"#define MPC_NFEAT {spec.nfeat}\n")
        f.write(f"#define MPC_FIT_DEGREE {spec.degree}\n")
        f.write(f"#define MPC_FEAT_CENTER {float(spec.center)!r}f\n")
        f.write(f"#define MPC_FEAT_SCALE {float(spec.scale)!r}f\n\n")
        f.write("/* 特征顺序（指数表）: (i, j) -> l_l^i * l_r^j */\n")
        f.write("static const int MPC_FEAT_EXPONENTS[MPC_NFEAT][2] = {\n")
        f.write("    " + ", ".join("{%d, %d}" % (i, j) for i, j in spec.exponents) + "\n};\n\n")

        for name in MATRIX_ORDER:
            coeff = res.coeffs[name]
            dim, rows, cols = res.mat_info[name]
            macro = f"MPC_{name.upper()}_COEFF"
            f.write(f"/* {name}_coeff: ({rows}, {cols}, {spec.nfeat})，行主序展开 */\n")
            f.write(f"static const mpc_real_t {macro}[{coeff.size}] = {{\n")
            for i in range(dim):
                vals = ", ".join(f"{float(v)!r}f" for v in coeff[i])
                f.write(f"    {vals},\n")
            f.write("};\n\n")

        f.write("/*\n"
                " * 用法：运行时按 MPC_FEAT_EXPONENTS 构造特征向量 f (长度 MPC_NFEAT)，\n"
                " * 对每个系数矩阵做 coeff(dim, NFEAT) * f 后 reshape 成 (rows, cols)。\n"
                " */\n")
    return path


def write_all(out_dir: str, res: FitResult, cfg: MpcConfig, params: Params,
              n_samples: int, header: bool = True) -> Dict[str, str]:
    """按约定的文件名写出全部产物，返回 {产物名: 路径}。"""
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}
    out["coeff.py"] = write_coeff_py(os.path.join(out_dir, "coeff.py"), res, n_samples)
    out["coeff_c_arrays.txt"] = write_c_arrays(
        os.path.join(out_dir, "coeff_c_arrays.txt"), res)
    if header:
        out["mpc_offline_data.hpp"] = write_offline_header(
            os.path.join(out_dir, "mpc_offline_data.hpp"), res, cfg, params)
    return out
