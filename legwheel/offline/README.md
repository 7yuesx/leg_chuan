# offline —— 腿足式平衡轮 MPC 离线矩阵拟合

把原 `compute_fit.py` 那份单文件脚本按功能拆成这样一组模块：符号建模 → 几何 → 平衡点 →
线性化 → 预计算 → 采样 → 拟合 → 代码生成。参考对象是 TinyMPC 的离线矩阵形式
（`tiny_setup` → `tiny_precompute_and_set_cache` → `tiny_codegen`）。

几何模型使用**解析两连杆公式**，本包不依赖 MuJoCo。

## 快速开始

```bash
# 默认写到仓库根目录（会覆盖 coeff.py 与 coeff_c_arrays.txt，先备份）
python3 -m offline

# 写到别处、并行、少打点自检
python3 -m offline --out-dir /tmp/offline_out --jobs 4 --check-points 4

# 只计算并看报告，不落盘
python3 -m offline --no-write
```

作为库使用：

```python
from offline import build_context, compute_system, run

ctx = build_context()                 # 符号模型只建一次，之后所有采样点复用
s   = compute_system(0.20, 0.20, ctx) # 单个工作点的 A/B/P/K/Quu_inv/AmBKt/Le/theta
print(s.A.shape, s.K.shape, s.theta_l)

res = run(out_dir="/tmp/offline_out") # 跑完整条流水线
```

命令行参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--out-dir` | 仓库根目录 | 产物输出目录（自动创建） |
| `--jobs` | 1 | 并行进程数；1 为串行 |
| `--degree` | `config.MPC.fit_degree` | 拟合多项式总次数 |
| `--check-points` | 8 | 回代自检点数，0 跳过 |
| `--no-header` | 关 | 不生成 `mpc_offline_data.hpp` |
| `--no-write` | 关 | 只计算与报告 |
| `--quiet` | 关 | 减少输出 |

## 模块划分与依赖

依赖方向单向、无环：

```
config ─┬─> symbolic_model ─┬─> equilibrium ──> linearization ─┐
        ├─> geometry ───────┘                                  ├─> sampling ─> fitting ─> codegen
        └─> (Q/R/rho/N/Ts) ─────────────────> precompute ──────┘            (pipeline 串起来)
```

| 文件 | 职责 | 依赖 |
|---|---|---|
| `types.py` | 数据结构契约（对应 TinyMPC 的 `types.hpp`） | 无 |
| `config.py` | 物理参数、`Q/R/rho/Ts/N`、采样与拟合设置 | 无 |
| `symbolic_model.py` | T/V/L、拉格朗日方程、隐式残差 F 及雅可比、`lambdify` | config, sympy |
| `geometry.py` | 解析两连杆腿几何，按腿长缓存 | config |
| `equilibrium.py` | 平衡点 (θ, θl, θr)（符号梯度求解）+ 实测拟合兜底 | symbolic_model, geometry |
| `linearization.py` | 平衡点线性化 + 仿射修正 + ZOH 离散 | equilibrium, geometry |
| `precompute.py` | Riccati 求解、cache 派生、LESO 增益（纯矩阵，无 sympy/mujoco） | 无（只吃 numpy/scipy） |
| `sampling.py` | 网格扫描、可选并行、样本完整性校验 | linearization, precompute |
| `fitting.py` | 特征基、最小二乘、逐元素残差、回代自检 | numpy |
| `codegen.py` | 写 `coeff.py` / C 数组 / `mpc_offline_data.hpp` | fitting, config |
| `pipeline.py` | 流程编排 + 命令行入口 | 全部 |

对外门面只有三个：`build_context()`、`compute_system(l_l, l_r, ctx)`、`run(...)`。
`compute_system` 保留了旧函数名，方便对比脚本与调试直接替换。

## 产物

`run()` / `--out-dir` 会写出三个文件，文件名与旧实现一致：

| 文件 | 内容 |
|---|---|
| `coeff.py` | 12 个 numpy 数组，供 `leg_control.py` 直接 `import` |
| `coeff_c_arrays.txt` | `#define XXX_Coeffs_Def { ... }` 形式的一维 float32 数组 |
| `mpc_offline_data.hpp` | 结构体风格头文件，对齐 TinyMPC 的 `tiny_data.hpp`：维数、rho、Ts、特征指数表、归一化常数、各系数数组 |

`coeff.py` 里的系数数组（行主序展平，可 `reshape(rows, cols, nfeat)`，`nfeat = 10`）：

| 变量 | 形状 | 含义 |
|---|---|---|
| `A_coeff` | (10, 10, 10) | 离散状态矩阵 |
| `B_coeff` | (10, 4, 10) | 离散输入矩阵 |
| `f_coeff` | (10, 1, 10) | 离散仿射项（TinyMPC 的 `fdyn`），线性化点非精确平衡点时非零 |
| `P_coeff` | (10, 10, 10) | 终端代价 `Pinf`（Riccati 解，**正定**） |
| `K_coeff` | (4, 10, 10) | LQR 增益 `Kinf`（**正号**，控制律 `u = -K x`） |
| `Quu_inv_coeff` | (4, 4, 10) | `(R + rho*I + B'PB)^-1`，即 TinyMPC 的 `Quu_inv`（`C1`）；旧名 `S` |
| `AmBK_T_coeff` | (10, 10, 10) | `(A - B*Kinf)^T`（即 TinyMPC 的 `AmBKt`） |
| `APf_coeff` | (10, 1, 10) | TinyMPC 的 `APf = AmBKt*Pinf*fdyn` |
| `BPf_coeff` | (4, 1, 10) | TinyMPC 的 `BPf = B'*Pinf*fdyn` |
| `Le_coeff` | (14, 10, 10) | LESO 观测器增益 |
| `theta_coeff` | (1, 1, 10) | 机体俯仰平衡角 θ* |
| `theta_l_coeff` / `theta_r_coeff` | (1, 1, 10) | 平衡腿角 |

特征基顺序固定为 `1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3`（`x = l_l`，`y = l_r`），
与 `user_class.Mpc.get_matrix` 里手工构造的 10 项特征一一对应。

### 关于 `theta_coeff`

平衡点是势能 `V(θ, θl, θr)` 的驻点，三个角一起解：`∂V/∂θ = ∂V/∂θl = ∂V/∂θr = 0`。
θ 的方程来自机身重力项 `m*g*l*cos(θ)`，即 `∂V/∂θ = -m*g*l*sin(θ)`：

| `PARAMS.l` | θ 的处理 |
|---|---|
| `l != 0` | `(θ, θl, θr)` 三未知量联立 `fsolve`，θ* 被真正解出并参与线性化点 |
| `l == 0`（当前值） | `∂V/∂θ ≡ 0`：机身质心落在髋轴线上，θ 是自由坐标，物理上任何俯仰角都平衡。此时取 `theta_p`（=0）作为平衡俯仰角，`theta_coeff` 就是零多项式 |

所以当前产物里 `theta_coeff` 全为 0 是模型结论而不是拟合失败；想要非零的俯仰目标，
得给 `PARAMS.l` 一个非零的质心水平偏置，或者像 `theta_balance_fit` 那样接入实测曲线。

## 配置项（`config.py`）

| 字段 | 当前值 | 说明 |
|---|---|---|
| `fit_degree` | 3 | 拟合多项式总次数，3 对应 10 个特征 |
| `l_min/l_max/l_step` | 0.10 / 0.40 / 0.02 | 采样网格，**含两端**：0.10 ~ 0.40 共 16 档（256 个采样点） |
| `rho` | 1.0 | ADMM 罚参数 |
| `Ts` | 0.002 | 离散化周期 |
| `N` | 100 | 预测步数 |
| `Q/R` | 见文件 | 状态/输入权重 |
| `obs_Qo_state / obs_Qo_d / obs_Ro` | 10 / 10 / 1000 | LESO 三个旋钮 |
| — | — | 预计算算法固定为 TinyMPC 形式，没有风格开关：权重先加 `rho*I`，再用 scipy DARE 直接解 |
| `linearization_point` | `"equilibrium"` | `equilibrium` = 解平衡点 (θ, θl, θr)；`zero` = 直接取 `theta=theta_p`、`theta_l=theta_r=0` |
| `normalize_features` | `False` | `True` 会改变 runtime 特征构造，需同步改 `user_class` |

物理参数集中在 `PARAMS`，原文件里被注释掉的备选数值以注释形式保留在旁边。

## 几何模型：解析式 vs MuJoCo

`geometry.py` 用原 `_geometry_analytic` 的两连杆公式：

```
H    = arccos((l_b^2 + l_s^2 - L^2) / (2*l_b*l_s))
c_x  = l_b*l_s*sin(H) / (2L)
c_h  = (L/2)/(1 + m_s/m_b) + (l_s^2 + L^2 - l_b^2) / (4L)
thp  = atan2(c_x, c_h)
```

按腿长缓存；`L` 超出三角不等式范围时对 `arccos` 做裁剪并计数，流程末尾统一告警。

实测两者差异不小（旧实现用 MuJoCo 按连杆质量加权测腿质心）：

| L | MuJoCo `c_h` | 解析 `c_h` | `thp` 差 (rad) |
|---|---|---|---|
| 0.10 | 0.016877 | 0.094118 | -0.487 |
| 0.20 | 0.051413 | 0.126130 | -0.154 |
| 0.28 | 0.071857 | 0.162386 | -0.026 |

解析式给出的腿质心离髋→轮轴更远、`thp` 更小，因此平衡腿角更大：L=0.2 处
`theta` 从 0.0504 变为 0.0850，系数矩阵 A 变化 20%~36%。这是几何模型切换本身的
代价，不是重构误差；若控制器原先照 MuJoCo 版 `coeff.py` 整定，换几何后需要重新整定。

## 预计算（TinyMPC 形式，唯一一套）

只有一套算法，与 TinyMPC 一致：

```
Q1 = Q + rho*I,  R1 = R + rho*I
Pinf, Kinf  由 (Q1, R1) 的 Riccati 方程给出
Quu_inv = (R1 + B'Pinf B)^-1        → 落盘为 Quu_inv
AmBKt   = (A - B*Kinf)^T            → 落盘为 AmBK_T
APf     = AmBKt*Pinf*fdyn,  BPf = B'*Pinf*fdyn
```

权重先加 `rho*I` 不是调参选择：ADMM 子问题的代价里状态和输入的 Hessian 各多一个
`rho*I`，子问题的最优控制律天生就是 `(Q+rho*I, R+rho*I)` 的 LQR。所以
`K/P/Quu_inv/AmBKt` 全部随 `rho` 变（实测 rho=0.1/10 时 K 变 27%/45%），
而不是像旧写法那样只有 `Quu_inv` 变。

### 求解方式：scipy DARE 直接法

解同一个 Riccati 方程有两种办法，本包用**直接法**（`scipy.linalg.solve_discrete_are`）：

| | 直接法（当前） | TinyMPC 的固定点迭代（`riccati_fixed_point()`，保留做参考） |
|---|---|---|
| 精度 | 一次到机器精度 | 停在 tol=1e-5，实测 P/K 差 3.5e-5~1.2e-4 |
| 速度 | 全流程约 2.5 秒 | 全流程约 20 秒（每点需 2800~4000 次迭代） |
| 失败模式 | Q 非半正定时抛异常（有断言兜底） | 不收敛时返回未到不动点的迭代值 |

两者的等价性实测：把 `(Q+rho*I, R+rho*I)` 交给直接法，与 TinyMPC 的迭代结果
相差 `P 6.3e-5 / K 7.6e-5 / Quu_inv 1.1e-6 / AmBKt 8.4e-5`——即迭代容差本身。
这套系统的可控性 Gram 最小奇异值约 `8.9e-8`，所以迭代收敛特别慢
（tol=1e-5 要 2807~3958 次，中位 3098），TinyMPC 写死的 1000 次上限一个点都不够，
它那种情况会静默返回未收敛的 `Kinf`。直接法没有这个问题，所以默认流程改用它；
`riccati_fixed_point()` 逐行照抄 `tiny_api.cpp`，保留用于交叉校验。

### 仿射项（TinyMPC 的 fdyn / APf / BPf）

线性化点一般不是精确平衡点，此时完整线性模型是

```
dz/dt = A_c dz + B_c du + f_c,      f_c = -F_a^-1 * F(p)
```

`A_c` 只保留纯雅可比 `-F_a^-1 * F_z`，偏置全部进 `f_c`；ZOH 离散化用增广矩阵指数
`expm([[A_c, f_c], [0, 0]] * Ts)` 同时得到 `A`、`f`。预计算再按 TinyMPC 派生
`APf = AmBKt*Pinf*f`、`BPf = B'*Pinf*f`，与 `A/B/f/K/P/Quu_inv/AmBKt/Le/theta` 一起
在腿长网格上拟合、落盘。

（原 MATLAB 写法是把补偿项 `M_eq^-1 * d(M_eq*u)/dz` 折进 `A_c`。本模型的质量矩阵与
状态无关，该项恒等于 0，所以两种写法在数值上一致，但只有分离出 `f` 才能对接
TinyMPC 的 `APf/BPf`。相关中间量 `dM_eq_u_fun` 仍保留在符号模型里备查。）

实测仿射项量级（`L=0.2` 附近）：

| 配置 | `|F(p)|` | `|f_c|` | `|f|`（每步） | `|APf|` | `|BPf|` |
|---|---|---|---|---|---|
| 质心偏置 + 平衡点线性化（默认，`linearization_point="equilibrium"`） | 2.2e-16 | 1.1e-17 | 1.1e-17 | 1.2e-14 | 2.5e-17 |
| 质心偏置 + `theta=0` 线性化（`"zero"`） | 1.58 | 41.19 | 0.0824 | 86.5 | 0.187 |
| 无质心偏置 + `theta=0` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

也就是说：只有在"几何带质心偏置、却又不在平衡点上线性化"时仿射项才真正非零——那正是
旧写法里补偿项为 0、漂移完全没被建模的情况。默认配置（解平衡腿角）下 `f ≈ 0`，
`f_coeff/APf_coeff/BPf_coeff` 会全为数值零。

离散化校验：每步 `|f| = 0.0824` 与 `|f_c|*Ts = 41.19*0.002 = 0.0824` 一致。

与 TinyMPC 的对应关系（`tiny_api.cpp` / `types.hpp`）：

| TinyMPC | offline | 是否一致 |
|---|---|---|
| `Q1 = Q + rho*I`、`R1 = R + rho*I` | 同 | 一致 |
| `Kinf/Pinf` 固定点迭代 | `dare_gain()`（等价解，精度/速度更优；`riccati_fixed_point()` 保留为逐行参考） | 相差迭代容差 3.5e-5~1.2e-4 |
| `Quu_inv`（`C1`） | `Quu_inv`（旧名 `S`） | 命名与数值都一致 |
| `AmBKt`（`C2`） | `AmBK_T` | 一致 |
| `fdyn` 仿射向量 | `f`（线性化残差单列，不再折进 A） | 一致 |
| `APf = AmBKt*Pinf*fdyn`、`BPf = B'*Pinf*fdyn` | `APf` / `BPf` | 一致 |

符号约定由 `config.cache_sign` 控制，默认 `"positive"`，与 TinyMPC 一致：

* `positive`（当前）：`P = Pinf`（正定），`K = Kinf`（正号增益，`u = -K x`）；
* `negative`（旧工程约定，仅供对比）：`P = -Pinf`，`K = -Kinf`。

两种约定下 `Quu_inv = (R + rho*I + B'P B)^-1`、`AmBKt = (A - B*K)^T`（都基于正定解）。
`precompute.check_cache_consistency()` 可以校验 Quu_inv/AmBKt 与 P/K 自洽。

> **运行期必须配套。** `positive` 对应 `user_class.Mpc` 里 `u = -K @ e ...`；
> `negative` 对应旧写法 `u = K @ e ...`。本仓库运行期已切到 `positive`
> （`update_forward` 与 `Admm` 各一行），换约定时两处要同时改，否则闭环会变成正反馈。

备注：rho 进权重对结果影响不小（相对未正则化的版本，`P` 16%、`K` 33%、`Quu_inv` 5%、`AmBKt` 38%
@ L=0.10~0.40；L=0.2 处 `||K||max` 55.7 → 42.0），所以从旧写法切过来时运行期需要重新整定。

## 可观测性

跑完会打印逐元素拟合残差与非网格点回代自检（`fitting.residual_report` / `self_check`）：

```
拟合残差（逐元素相对误差）:
         A: max_rel=6.991e-02 (展开下标 63), matrix_rel=5.955e-03
         K: max_rel=6.759e-03 (展开下标 18), matrix_rel=8.047e-04
   theta_l: max_rel=1.562e-02 (展开下标 0),  matrix_rel=1.562e-02

回代自检（8 个非网格点，解析值 vs 拟合重建）:
         A: rel_err=4.547e-03
         K: rel_err=5.996e-04
```

逐元素残差比全矩阵范数更有用：A 的 `(6, 3)` 元素（`∂w_dot/∂theta_l`）有约 7% 误差，
而矩阵整体只有 6e-3。要针对这类块单独提高阶数，看这个报告即可。

## 验证结论

* 单点交叉验证（旧 `compute_fit.py` 强制走解析兜底 vs 本包）：几何向量逐位一致；
  A/B/P/K/Quu_inv/AmBKt/Le 相对差约 `1e-7`。
* 平衡点精度：旧实现有限差分梯度 `max|dV| = 5.2e-6`，本包符号梯度 `2.2e-16`；
  上面那 `1e-7` 的矩阵差异来自旧求解器误差而非重构。
* 全流程 256/256 样本约 2.5 秒（scipy DARE 直接法；若改用 `riccati_fixed_point()` 迭代约 20 秒）。
* `--jobs 4` 与串行产物逐字节一致。
* 生成的 `coeff.py` 变量名与形状与原文件完全相同（只多出 `theta_coeff`），
  `user_class.Mpc` 可直接消费。
* **符号翻转等价性**：同一系统的两套系数（`+P/+K` 与 `-P/-K`）分别配 `user_class` 的
  新/旧公式，跑同样的初值与 ADMM 迭代，`u / z / lambda / e` 序列相对差为 `0.0`
  （逐位一致）；`P` 的特征值由 `[-2.68e5, -14.7]` 变成 `[14.7, 2.68e5]`，
  即符号翻转本身不改变任何控制行为。
* **正号产物自洽**：各工作点 `P` 全部正定（特征值 `6.3 ~ 2.8e5`）；`K` 与由重建的
  `A/B/P` 反算出的 `K_lqr` 相差 0.4%~1%（即独立拟合误差）；闭环 `|eig|max ≈ 0.9987`；
  `AmBK_T` 与 `(A-BK)^T`、`Quu_inv` 与 `(R + rho*I + B'PB)^-1` 各自在 `5e-4` 内吻合。
* **与 TinyMPC 等价**：`riccati_fixed_point` 与 `tiny_api.cpp` 的
  `tiny_precompute_and_set_cache` 逐行一致（同初值 `K=0`、`P=rho*I`，同判据 `1e-5`）；
  默认流程用的直接法与它在各工作点相差 `P/K` 3.5e-5~1.2e-4，即迭代容差。

## 已知限制

* `coeff.py` 会被 `codegen` 覆盖，重新生成前先备份当前版本。
* `cache_sign` 必须与 `user_class.Mpc` 的写法一致，改一边而不改另一边会破坏稳定性。
* `user_class.Mpc.get_matrix` 里特征数写死为 10，改 `fit_degree`（或开 `normalize_features`）
  必须同步改 runtime。
* `Quu_inv`/`AmBKt` 目前与 `A/B/P/K` 一样独立拟合；若要换成"由拟合后的 P/K 派生"，
  需同时调整 runtime 与 C 端数据结构。
* 非精确平衡点导致的仿射项仍折进等效 `A_c`（对应 TinyMPC 的 `fdyn/APf/BPf` 位置），
  当前没有单列成 `f_coeff`。
* 采样网格用 `linspace` 生成、含右端点（旧的 `np.arange` 会把区间少取一档）。改 `l_max`
  后拟合精度会随区间变宽而下降——同一批 256 个样本上实测：

  | 阶数 | 特征数 | A max_rel | K max_rel | theta max_rel | 回代 A | 回代 K |
  |---|---|---|---|---|---|---|
  | 3（默认） | 10 | 1.41e-1 | 1.25e-2 | 3.43e-2 | 7.75e-3 | 1.11e-3 |
  | 4 | 15 | 6.05e-2 | 3.44e-3 | 1.16e-2 | 4.85e-3 | 4.81e-4 |
  | 5 | 21 | 1.52e-2 | 9.70e-4 | 4.55e-3 | 3.77e-3 | 3.19e-4 |

  提高阶数必须同步改 `user_class.Mpc.get_matrix` 里写死的 10 项特征构造。
* 极端腿长（如 `L=0.30/0.32`）偶尔出现 `fsolve ier=5` 告警，但残差已是 2.2e-16
  （机器精度），属于"解已收敛、返回码不达标"，代码按残差判定并继续使用该解。
* 根目录的 `compute_fit.py`（旧实现）仍保留作为对照，尚未改成调用本包的薄壳。
* 切换前的产物备份在根目录：`coeff.py.bak_negative`、`coeff_c_arrays.txt.bak_negative`
  （负号约定 + MuJoCo 几何版本）、`coeff.py.bak_legacy`、`coeff_c_arrays.txt.bak_legacy`
  （正号 + 解析几何 + 未正则化预计算），确认无误后可删除。
