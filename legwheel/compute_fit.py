"""
MATLAB -> Python 转换

对应原 MATLAB 代码：
  - compute_system(l_l_val, l_r_val)
  - 对 (l_l, l_r) 采样网格逐个求系统矩阵
  - 对每个矩阵元素做二元二次拟合 [1, x, y, x^2, x*y, y^2]
  - 不再保存 .mat 文件，直接把拟合系数按 numpy 数组格式打印到屏幕

系数数组的行序为“行主序”(row-major)，即可以直接用
    coeff.reshape(rows, cols, 6)
还原成 (rows, cols, 6) 的系数张量（与 user_class.py 中 Mpc.get_matrix 的用法一致）。
"""

import numpy as np
import sympy as sp
from scipy.optimize import fsolve, least_squares
from scipy.signal import cont2discrete
from scipy.linalg import solve_discrete_are
import os as _os

# ==================== 1. 固定参数（与原 MATLAB 完全一致） ====================
m   = 16.877574000000#20.275132741229 #17.275132741229
m_l = (1.113200000000+0.306873000000+0.058433000000
       +0.042999000000+0.065877000000+0.010945000000) #(0.134820000000 + 0.155400000000 + 0.230811000000
       #+ 0.117180000000 + 0.349772111111 + 0.236250000000)
m_r = (1.113200000000+0.306873000000+0.058433000000
       +0.042999000000+0.065877000000+0.010945000000)#(0.134820000000 + 0.155400000000 + 0.230811000000
       #+ 0.117180000000 + 0.349772111111 + 0.236250000000)
m_w = 0.490400000000#0.353429173529
j_z = 0.359350481645
j_y = 0.457285642113
j_l_z = (0.00468513+2.48016e-03+9.09634e-05
         +7.61359e-05+2.05523e-04+9.8695e-06)#4 * (2.96625000e-04 + 7.109375000e-06)
j_l_y = (0.00641833+2.54050e-03+1.01653e-04
         +7.83158e-05+2.11040e-04+1.0337e-05)#4 * (2.98265625e-04 + 1.372109375e-03)
j_r_z = (0.00468513+2.48016e-03+9.09634e-05
         +7.61359e-05+2.05523e-04+9.8695e-06)#4 * (2.96625000e-04 + 7.109375000e-06)
j_r_y = (0.00641833+2.54050e-03+1.01653e-04
         +7.83158e-05+2.11040e-04+1.0337e-05)#4 * (2.98265625e-04 + 1.372109375e-03)
d   = 0.2#0.165#
j_w_z = 0.00034133
j_w_y = 0.0005817
J_z = j_z

l   = 0.0
r   = 0.06
g   = 9.81

l_b = 0.21#0.215
l_s = 0.2462962962962#0.258
m_b = 0.134820000000 + 0.155400000000 + 0.230811000000 + 0.117180000000 + 0.236250000000
m_s = 0.349772111111+0.353429173529
theta_p = 0.0

# ==================== 2. 符号变量（函数外一次性建立） ====================
# 状态：z = [x, w, theta, theta_l, theta_r, x_dot, w_dot, theta_dot, theta_dot_l, theta_dot_r]
x, w, th, thl, thr = sp.symbols('x w theta theta_l theta_r')
vx, vw, vth, vthl, vthr = sp.symbols('x_dot w_dot theta_dot theta_dot_l theta_dot_r')
ax, aw, ath, athl, athr = sp.symbols('x_ddot w_ddot theta_ddot theta_ddot_l theta_ddot_r')
M_l2, M_r2, M_l3, M_r3 = sp.symbols('M_l2 M_r2 M_l3 M_r3')

# 几何/长度相关符号（每个样本代入数值）
l_l, l_r = sp.symbols('l_l l_r')
c_h1, c_h2, c_x1, c_x2, thp1, thp2 = sp.symbols('c_h1 c_h2 c_x1 c_x2 theta_p1 theta_p2')

coords = [x, w, th, thl, thr]
vels   = [vx, vw, vth, vthl, vthr]
accs   = [ax, aw, ath, athl, athr]
state  = coords + vels
Ms     = [M_l2, M_r2, M_l3, M_r3]
geom_syms = [l_l, l_r, c_h1, c_h2, c_x1, c_x2, thp1, thp2]

# ==================== 3. 动能 T、势能 V（完全复制原代码） ====================
T = (
    sp.Rational(1, 2) * m_w
     * (vx - sp.sqrt(d**2 + (sp.sin(thl) * 2 * l_l)**2) * vw
        + sp.cos(thl) * 2 * l_l * vthl)**2
     + sp.Rational(1, 2) * m_w
     * (vx + sp.sqrt(d**2 + (sp.sin(thr) * 2 * l_r)**2) * vw
        + sp.cos(thr) * 2 * l_r * vthr)**2
     + sp.Rational(1, 2) * j_w_y
     * ((vx - sp.sqrt(d**2 + (sp.sin(thl) * 2 * l_l)**2) * vw
         + sp.cos(thl) * 2 * l_l * vthl) / r)**2
     + sp.Rational(1, 2) * j_w_y
     * ((vx + sp.sqrt(d**2 + (sp.sin(thr) * 2 * l_r)**2) * vw
         + sp.cos(thr) * 2 * l_r * vthr) / r)**2
     + sp.Rational(1, 2) * j_w_z * vw**2 * 2
     + sp.Rational(1, 2) * j_l_y * vthl**2
     + sp.Rational(1, 2) * m_l
     * ((vx - sp.sqrt(d**2 + (sp.sin(thl) * (2 * l_l - c_h1))**2) * vw
         + sp.cos(thl) * (2 * l_l - c_h1) * vthl)**2
        + (c_h1 * vthl * sp.sin(thl))**2)
     + sp.Rational(1, 2) * j_l_z * vw**2
     + sp.Rational(1, 2) * j_r_y * vthr**2
     + sp.Rational(1, 2) * m_r
     * ((vx + sp.sqrt(d**2 + (sp.sin(thr) * (2 * l_r - c_h2))**2) * vw
         + sp.cos(thr) * (2 * l_r - c_h2) * vthr)**2
        + (c_h2 * vthr * sp.sin(thr))**2)
     + sp.Rational(1, 2) * j_r_z * vw**2
     + sp.Rational(1, 2) * j_y * vth**2
     + sp.Rational(1, 2) * m
     * (vx**2
        + (l * vth * sp.sin(th))**2
        + ((l_l + l_r) * (vthl + vthr) / 2 * sp.sin((thl + thr) / 2))**2)
        + sp.Rational(1, 2) * J_z * vw**2)

V = (m_l * g * sp.sqrt(c_h1**2 + c_x1**2) * sp.cos(thl - thp1)
     + m_r * g * sp.sqrt(c_h2**2 + c_x2**2) * sp.cos(thr - thp2)
     + m * g * ((l_l + l_r) * sp.cos((thl + thr) / 2) + l * sp.cos(th + 0.0)))

L = T - V

# ==================== 4. 拉格朗日方程 ====================
def euler_lagrange(i):
    """d/dt(dL/dv_i) - dL/dq_i，其中 v_i 是 q_i 的广义速度。"""
    v = vels[i]
    pv = sp.diff(L, v)
    dpv = 0
    for j in range(5):
        dpv += sp.diff(pv, coords[j]) * vels[j]   # 位置项
        dpv += sp.diff(pv, vels[j]) * accs[j]     # 加速度项
    return dpv - sp.diff(L, coords[i])

EL = [euler_lagrange(i) for i in range(5)]

# 与原 MATLAB 的 5 个方程一一对应（把方程写成 F = 0 的形式）
F = [
    (M_l3 + M_r3) / r        - EL[0],   # eq1: x
    (-M_l3 + M_r3) / r * d   - EL[1],   # eq2: w
    -(M_l2 + M_r2)           - EL[2],   # eq3: theta
    (M_l2 - M_l3)            - EL[3],   # eq4: theta_l
    (M_r2 - M_r3)            - EL[4],   # eq5: theta_r
]

# ==================== 5. 平衡点（与 linearize.py force_horizontal 同方法） ====================
# 势能 V：腿质心势能 + 机身势能。机身俯仰 θ 固定为 theta_p（=0，水平），
# 平衡腿角 = 使 ∂V/∂θl = ∂V/∂θr = 0 的 (θl, θr)，即“腿角偏移保证机身水平”。
th1s, thl1, thr1 = sp.symbols('theta_1 theta_l1 theta_r1')
V_full = (m_l * g * sp.sqrt(c_h1**2 + c_x1**2) * sp.cos(thl1 - thp1)
          + m_r * g * sp.sqrt(c_h2**2 + c_x2**2) * sp.cos(thr1 - thp2)
          + m * g * ((l_l + l_r) * sp.cos((thl1 + thr1) / 2)
                     + l * sp.cos(th1s + 0.0)))

# ==================== 6. 一次性 lambdify ====================
# 势能 V 数值函数：以 (θ, θl, θr, 几何参数) 为输入，
# 物理参数（m、m_l、m_r、l_b、l_s 等）使用本文件的 legwheel 数值。
V_balance = sp.lambdify((th1s, thl1, thr1, geom_syms), V_full,
                        modules='numpy', cse=True)

# 动力学残差 F 及其雅可比：以 (状态, 加速度, 力矩, 几何参数) 为输入
F_fun  = sp.lambdify((state, accs, Ms, geom_syms), F,
                     modules='numpy', cse=True)
dF_dz  = sp.lambdify((state, accs, Ms, geom_syms),
                     sp.Matrix(F).jacobian(state), modules='numpy', cse=True)
dF_da  = sp.lambdify((state, accs, Ms, geom_syms),
                     sp.Matrix(F).jacobian(accs), modules='numpy', cse=True)
dF_dM  = sp.lambdify((state, accs, Ms, geom_syms),
                     sp.Matrix(F).jacobian(Ms), modules='numpy', cse=True)

# MATLAB 的 taylor 是在“非精确平衡点”p 处展开的（p 处 F != 0），
# 因此加速度对状态的导数除了隐式项 -F_a^-1 * dF_dz 之外，还有一项：
#     M_eq^-1 * d(M_eq*u)/dz ，其中 u = M_eq^-1 * F(p)
# 这里预先符号化 d(M_eq*u)/dz（u 为 5 个占位符号）。
u1, u2, u3, u4, u5 = sp.symbols('u1 u2 u3 u4 u5')
u_syms = [u1, u2, u3, u4, u5]
M_eq = sp.Matrix(F).jacobian(accs)                 # 5x5
dM_eq_u_dz = (M_eq * sp.Matrix(u_syms)).jacobian(state)   # 5x10
dM_eq_u_fun = sp.lambdify((state, accs, Ms, geom_syms, u_syms),
                          dM_eq_u_dz, modules='numpy', cse=True)


# ==================== 7. 几何参数（照 linearize.py 用 MuJoCo 实测） ====================
# 思路与 linearize.py 的 LegPoseSolver 一致：用 MuJoCo 模型按腿长摆位后，
# 按连杆实际质量加权测腿质心。legwheel 的前后链通过弹簧连接（非刚性闭合），
# 因此摆位规则为：
#   1) 前链（lf0→lf1→轮）按 2 连杆逆解：轮子在髋正下方、髋→轮距 = L
#      （膝角由 l_b、l_s 与 L 的余弦定理自然确定，取参考位形的解分支）；
#   2) 后链（l20~l23）保持参考位形；
#   3) 按 6 个连杆的实际质量加权求质心，投影到髋→轮轴线，
#      得到 (c_h 轴向, c_x 切向, thp)。
# 原解析两连杆公式保留为 _geometry_analytic，作为机构不可达时的兜底。

_MJ_MODEL = None
_MJ_DATA = None
_GEO_FALLBACKS = 0


def _mujoco_geometry(L):
    """MuJoCo 实测单个腿长 L 的 (c_h, c_x, thp)；机构不可达时返回 None。"""
    import mujoco
    global _MJ_MODEL, _MJ_DATA
    if _MJ_MODEL is None:
        _MJ_MODEL = mujoco.MjModel.from_xml_path(
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'mjmodel.xml'))
        _MJ_DATA = mujoco.MjData(_MJ_MODEL)
    m, d = _MJ_MODEL, _MJ_DATA

    hip_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'lf0_Link')
    wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'l_wheel_Link')
    # 前链两个驱动关节：qpos 下标（base 自由关节占前 7 个）与关节 id
    # leftjoint6（髋/摆动，无限位）、lf1_Joint（膝，有限位）
    jidx = [7, 8]
    jids = [1, 2]

    lo, hi = [], []
    for jid in jids:
        if m.jnt_limited[jid]:
            lo.append(m.jnt_range[jid, 0]); hi.append(m.jnt_range[jid, 1])
        else:
            lo.append(-np.inf); hi.append(np.inf)

    d.qpos[:] = m.qpos0
    d.qvel[:] = 0; d.ctrl[:] = 0

    def res(q):
        d.qpos[jidx] = q
        d.qvel[:] = 0; d.ctrl[:] = 0
        mujoco.mj_forward(m, d)
        w = d.body(wheel_id).xpos - d.body(hip_id).xpos
        # 轮子在髋正下方（x 对齐），髋→轮竖直距离 = L
        return np.asarray([20 * w[0], 20 * (w[2] + L)])

    sol = least_squares(res, m.qpos0[jidx], bounds=(lo, hi),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
    if np.max(np.abs(res(sol.x))) > 1e-5:
        return None

    d.qpos[jidx] = sol.x
    mujoco.mj_forward(m, d)
    hip   = d.body(hip_id).xpos
    wheel = d.body(wheel_id).xpos
    axis  = (wheel - hip) / np.linalg.norm(wheel - hip)

    com = np.zeros(3); tot = 0.0
    for name in ['lf0_Link', 'lf1_Link', 'l20_Link', 'l21_Link', 'l22_Link', 'l23_Link']:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        tot += m.body_mass[bid]
        com += m.body_mass[bid] * d.xipos[bid]
    com /= tot

    dvec = com - hip
    c_h = float(np.dot(dvec, axis))
    t_axis = np.array([axis[2], 0.0, -axis[0]])
    t_axis /= np.linalg.norm(t_axis)
    c_x = float(np.dot(dvec, t_axis))
    thp = float(np.arctan2(c_x, c_h))
    return c_h, c_x, thp


def _geometry_analytic(l_l_val, l_r_val):
    """原解析两连杆腿质心公式（兜底/参考）。"""
    l_lh = l_l_val / 2
    l_rh = l_r_val / 2

    H1 = np.arccos((l_b**2 + l_s**2 - l_l_val**2) / (2 * l_b * l_s))
    c_x1 = l_b*l_s * np.sin(H1) / (2 * l_l_val)
    c_h1 = l_lh / (1 + m_s / m_b)+(l_s / 2 * (l_s**2 + l_l_val**2 - l_b**2) / (2 * l_s * l_l_val))
    thp1 = np.arctan2(c_x1, c_h1)

    H2 = np.arccos((l_b**2 + l_s**2 - l_r_val**2) / (2 * l_b * l_s))
    c_x2 = l_b*l_s * np.sin(H2) / (2 * l_r_val)
    c_h2 = l_rh / (1 + m_s / m_b)+(l_s / 2 * (l_s**2 + l_r_val**2 - l_b**2) / (2 * l_s * l_r_val))
    thp2 = np.arctan2(c_x2, c_h2)

    return np.array([l_lh, l_rh, c_h1, c_h2, c_x1, c_x2, thp1, thp2])


def geometry(l_l_val, l_r_val):
    """输入 l_l_val / l_r_val，输出 [l_l, l_r, c_h1, c_h2, c_x1, c_x2, theta_p1, theta_p2]。
    几何照 linearize.py 用 MuJoCo 实测；机构不可达的腿长退回解析公式。"""
    global _GEO_FALLBACKS
    out = np.empty(8)
    for i, L in enumerate((float(l_l_val), float(l_r_val))):
        got = _mujoco_geometry(L)
        if got is not None:
            c_h, c_x, thp = got
        else:
            _GEO_FALLBACKS += 1
            l_h = L / 2
            H = np.arccos(np.clip((l_b**2 + l_s**2 - L**2) / (2 * l_b * l_s), -1, 1))
            c_x = l_b * l_s * np.sin(H) / (2 * L)
            c_h = l_h / (1 + m_s / m_b) + (l_s / 2 * (l_s**2 + L**2 - l_b**2) / (2 * l_s * L))
            thp = np.arctan2(c_x, c_h)
        out[2 + i] = c_h
        out[4 + i] = c_x
        out[6 + i] = thp
    out[0] = float(l_l_val) / 2
    out[1] = float(l_r_val) / 2
    return out


# ==================== 7.5 实测平衡腿角 ====================
# 来自仿真测量（sim_leg_angle.py）：机器人静止时的腿角按 Leg.forward 定义
#   theta = -atan2(x_c, y_c) + pitch
# 对 8 个腿长特征点测得的稳态均值做 4 次加权拟合，系数单位弧度（高次到低次）。
# 现在 compute_system 改为数值求解平衡方程（与 linearize.py force_horizontal
# 一致：机身水平 + 腿角偏移），本拟合值仅作为数值求解未收敛时的兜底。
# 测量范围 0.1187~0.3515 m，范围外取端点值（控制器腿长本身限制在 0.12~0.35）。
_THETA_MEAS_RANGE = (0.1187, 0.3515)
_THETA_FIT_COEFF = np.array([
    87.88538943, -110.36430963, 52.12496719, -11.17598776, 0.97426381])


def theta_balance_fit(L):
    """实测平衡腿角 θ(L)，弧度。范围外取端点。"""
    L = float(L)
    if L < _THETA_MEAS_RANGE[0]:
        L = _THETA_MEAS_RANGE[0]
    elif L > _THETA_MEAS_RANGE[1]:
        L = _THETA_MEAS_RANGE[1]
    return float(np.polyval(_THETA_FIT_COEFF, L))


# ==================== 8. compute_system 的 Python 版 ====================
def compute_system(l_l_val, l_r_val):
    """对应 MATLAB compute_system(l_l_val, l_r_val)。"""
    gv = geometry(l_l_val, l_r_val)

    # ---- 求平衡点 theta_l1, theta_r1（linearize.py force_horizontal 同款）----
    # 用势能 V 的数值梯度 + fsolve：机身俯仰固定为 theta_p（=0，水平），
    # 求解 ∂V/∂θl = 0、∂V/∂θr = 0，得到保证机身水平的腿角偏移。
    # 物理参数仍用本文件的 legwheel 数值（m、m_l、m_r、几何等）。
    def dV(q_a):
        qf = [theta_p, q_a[0], q_a[1]]     # [θ, θl, θr]，θ 固定为 theta_p
        eps = 1e-6; grad = np.zeros(2)
        V0 = V_balance(qf[0], qf[1], qf[2], gv)
        for i in range(2):
            qq = qf.copy(); qq[i+1] += eps
            grad[i] = (V_balance(qq[0], qq[1], qq[2], gv) - V0) / eps
        return grad

    sol, info, ier, msg = fsolve(dV, [0.0, 0.0], maxfev=200,
                                 xtol=1e-12, full_output=True)
    res = np.max(np.abs(info.get("fvec", dV(sol))))
    if ier != 1:
        # fsolve 有时返回 ier=5（步长很小但误差估计不达标），此时解往往已收敛，
        # 以残差为准而不是以返回码为准。
        print(f"  WARNING: 平衡点 fsolve ier={ier} L=({l_l_val:.4f}, {l_r_val:.4f}) "
              f"max|dV|={res:.3e}: {msg}")
    # 有限差分梯度（eps=1e-6）的残差噪声约 1e-9，阈值放宽到 1e-7
    if res <= 1e-7:
        theta_leg_l, theta_leg_r = float(sol[0]), float(sol[1])
    else:
        # 兜底：残差未达标时退回实测拟合值（7.5 节），并告警
        theta_leg_l = theta_balance_fit(l_l_val)
        theta_leg_r = theta_balance_fit(l_r_val)
        print(f"  WARNING: 平衡点数值求解残差未达标 L=({l_l_val:.4f}, {l_r_val:.4f}) "
              f"max|dV|={res:.3e}，退回实测拟合值 "
              f"theta=({theta_leg_l:.4f}, {theta_leg_r:.4f})")
    theta_body = theta_p

    # ---- 在平衡点处线性化 ----
    p = np.array([0.0, 0.0, theta_body, theta_leg_l, theta_leg_r,
                  0.0, 0.0, 0.0, 0.0, 0.0])
    M0 = np.zeros(4)
    a0 = np.zeros(5)

    F_z   = dF_dz(p, a0, M0, gv)     # 5x10
    F_a   = dF_da(p, a0, M0, gv)     # 5x5
    F_M   = dF_dM(p, a0, M0, gv)     # 5x4
    F_rest = F_fun(p, a0, M0, gv)    # 平衡点处的残差 F(z,0,0) != 0
    Cmat  = np.linalg.inv(F_a)       # = M_eq^-1
    u_vec = Cmat @ F_rest            # 5 维
    H_u   = dM_eq_u_fun(p, a0, M0, gv, u_vec)   # 5x10

    A_c = np.zeros((10, 10))
    A_c[:5, 5:] = np.eye(5)
    A_c[5:, :] = -np.linalg.solve(F_a, F_z) + Cmat @ H_u

    B_c = np.zeros((10, 4))
    B_c[5:, :] = -np.linalg.solve(F_a, F_M)

    C_c = np.eye(10)
    D_c = np.zeros((10, 4))

    # ---- 离散化（ZOH, Ts = 0.002）----
    Ts = 0.002
    A, B, C, D, _ = cont2discrete((A_c, B_c, C_c, D_c), Ts, method='zoh')
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # ---- LQR ----
    Q = np.array([[10,0,0,0,0,0,0,0,0,0],
                    [0,1,0,0,0,0,0,0,0,0],
                    [0,0,5000,0,0,0,0,0,0,0],
                    [0,0,0,4000,0,0,0,0,0,0],
                    [0,0,0,0,4000,0,0,0,0,0],
                    [0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0,1,0],
                    [0,0,0,0,0,0,0,0,0,1]])
        
    R = np.array([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    # Q 必须半正定，否则离散黎卡提方程可能无稳定解（scipy 会报 symplectic pencil 错误）
    qmin = np.linalg.eigvalsh(Q)[0]
    if qmin < -1e-9:
        raise ValueError(
            f"Q 不是半正定（最小特征值 {qmin:.3g}）：θ_l/θ_r 交叉项过大，"
            f"需满足 |offdiag| <= sqrt(q_l*q_r)")
    P = -solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(B.T @ -P @ B + R, B.T @ -P @ A)
    #直接拟合负矩阵
    # ---- S 和 AmBK ----
    rho = 1
    S = np.linalg.inv(R + rho * np.eye(4) + B.T @ -P @ B)
    # 注意：MPC 反向传播用的是 (A-BK)^T（用户确认），
    # 因此这里输出转置而不是 MATLAB 里的 inv(A-BK)。
    AmBK = (A - B @ -K).T

    # ---- LESO 观测器增益 Le（14 维增广：10 状态 + 4 输入扰动）----
    # 对应你的结构：
    #   X = [x(10); d(4)]，d 为输入扰动，u_actual = u + d
    #   Ae = [[A, B], [0, I4]]，Be = [B; 0]
    #   观测器：X[k+1] = Ae X[k] + Be u[k] + Le (y - Ce X[k])
    # 注意：极点配置(place)得到的 Le 随参数不连续，无法做多项式拟合（实测相邻腿长
    # 增益跳变 100+，拟合误差 ~50%）。改用离散 LQE（对偶 DARE），增益平滑可拟合。
    # Qo/Ro 是观测器调参（三个旋钮）：
    #   obs_Qo_state：状态通道过程噪声，越大观测器整体越快；
    #   obs_Qo_d    ：扰动通道过程噪声，越大扰动估计 d_hat 越猛（但更吃噪声）；
    #   obs_Ro      ：测量噪声权重，越大观测器越慢越平滑。
    E_aug = B                                     # 10x4：扰动加在输入通道
    Ae = np.block([[A, E_aug], [np.zeros((4, 10)), np.eye(4)]])   # 14x14
    Ce = np.block([np.eye(10), np.zeros((10, 4))])                # 10x14（测全部 10 个状态，可改）
    obs_Qo_state = 10.0
    obs_Qo_d = 10.0
    obs_Ro = 1000.0
    Qo = np.zeros((14, 14))
    Qo[:10, :10] = np.eye(10) * obs_Qo_state
    Qo[10:, 10:] = np.eye(4) * obs_Qo_d
    Ro = np.eye(10) * obs_Ro
    P_obs = solve_discrete_are(Ae.T, Ce.T, Qo, Ro)                # 对偶 DARE
    Le = P_obs @ Ce.T @ np.linalg.inv(Ro + Ce @ P_obs @ Ce.T)     # 14x10

    return (A, B, C, D, K, P, S, AmBK, Le,
            theta_body, theta_leg_l, theta_leg_r)


# ==================== 9. 采样 + 拟合 ====================
def main():
    # 采样区间：MuJoCo 实测几何可达范围（mjmodel.xml 前链 + lf1 关节限位）。
    # 实测覆盖 0.10~0.30，超出会退回解析公式；为保证 100% 实测，限定在此区间。
    l_vals = np.arange(0.10, 0.30, 0.02)
    # 拟合多项式总次数：2 = 原 MATLAB 的二元二次拟合；
    # 想要更高精度可改成 4（见 README 说明/末尾的实验结论）。
    FIT_DEGREE = 3

    print(f"采样区间: l_l, l_r = {l_vals[0]} ~ {l_vals[-1]} (步长 {l_vals[1]-l_vals[0]})")
    print(f"拟合次数: {FIT_DEGREE}")
    print("正在逐个样本计算 A, B, K, P, S, AmBK ...")

    samples = []
    for ll in l_vals:
        for lr in l_vals:
            try:
                res = compute_system(float(ll), float(lr))
            except Exception as exc:
                print(f"  跳过 ({ll:.4f}, {lr:.4f}): {exc}")
                continue
            A, B, C, D, K, P, S, AmBK, Le, tb, tll, tlr = res
            if not all(np.all(np.isfinite(v)) for v in (A, B, K, P, S, AmBK, Le)):
                continue
            samples.append((float(ll), float(lr), A, B, K, P, S, AmBK, Le,
                            float(tll), float(tlr)))

    N = len(samples)
    if N == 0:
        raise RuntimeError("没有成功样本，无法拟合")
    print(f"成功样本数: {N}\n")
    if _GEO_FALLBACKS:
        print(f"几何兜底（解析公式）次数: {_GEO_FALLBACKS} "
              f"（对应腿长超出 MuJoCo 机构可达范围）\n")

    x = np.array([s[0] for s in samples])
    y = np.array([s[1] for s in samples])
    # 特征：常数项 + 所有总次数 <= FIT_DEGREE 的单项式 x^i * y^j，
    # 顺序固定为 [1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3, ...]。
    feats = [[1.0] * N]
    for deg in range(1, FIT_DEGREE + 1):
        for i in range(deg, -1, -1):          # x 的指数从高到低
            feats.append(x**i * y**(deg - i))
    X = np.column_stack(feats)   # N x nfeat

    mat_info = {
        'A':    (100, 10, 10),
        'B':    (40,  10, 4),
        'P':    (100, 10, 10),
        'K':    (40,  4, 10),
        'S':    (16,  4, 4),
        'AmBK_T': (100, 10, 10),
        'Le':   (140, 14, 10),
        'theta_l': (1, 1, 1),
        'theta_r': (1, 1, 1),
    }

    feat_labels = ['1']
    for deg in range(1, FIT_DEGREE + 1):
        for i in range(deg, -1, -1):
            if i == 0:
                feat_labels.append(f'y^{deg}' if deg > 1 else 'y')
            elif i == deg:
                feat_labels.append(f'x^{deg}' if deg > 1 else 'x')
            else:
                feat_labels.append(f'x^{i}*y^{deg-i}')
    print(f"# 特征顺序: {', '.join(feat_labels)} (x = l_l, y = l_r)")
    print(f"# 系数数组按行主序排列，每行 {X.shape[1]} 个系数，可 reshape 成 (rows, cols, {X.shape[1]})\n")

    coeffs = {}
    for name, (dim, rows, cols) in mat_info.items():
        idx = {'A': 2, 'B': 3, 'K': 4, 'P': 5, 'S': 6, 'AmBK_T': 7,
               'Le': 8, 'theta_l': 9, 'theta_r': 10}[name]
        # 行主序展平每个矩阵，Y: (dim, N)
        Y = np.array([np.asarray(s[idx]).reshape(-1, order='C') for s in samples]).T
        coeff = np.linalg.lstsq(X, Y.T, rcond=None)[0].T.astype(np.float32)  # (dim, 10)
        coeffs[name] = coeff

        print(f"{name}_coeff = np.array([")
        for i in range(dim):
            row = ", ".join(repr(float(v)) for v in coeff[i])
            comma = "," if i < dim - 1 else ""
            print(f"    [{row}]{comma}")
        print(f"], dtype=np.float32)  # ({rows}, {cols}, {X.shape[1]})，行主序\n")

    # 再保存一份 C 语言一维数组（float32）格式的 txt
    out_txt = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'coeff_c_arrays.txt')
    with open(out_txt, 'w') as f:
        f.write("/* 系数矩阵，float32，按 C 语言一维数组（行主序）保存。\n")
        f.write(" * 特征顺序 (x = l_l, y = l_r): "
                + ", ".join(feat_labels) + "\n")
        f.write(" * 每个数组元素个数 = rows*cols*nfeat，可 reshape 成 "
                f"(rows, cols, {X.shape[1]})。\n */\n\n")
        for name, (dim, rows, cols) in mat_info.items():
            coeff = coeffs[name]
            f.write(f"/* {name}_coeff: {coeff.size} 个 float32 */\n")
            f.write(f"#define {name}_Coeffs_Def \\\n")
            f.write("    { \\\n")
            for i in range(dim):
                # repr(float32->float) 总带小数点或指数，保证是合法的 C 浮点字面量
                vals = ", ".join(f"{float(v)!r}f" for v in coeff[i])
                f.write(f"        {vals}, \\\n")
            f.write("    }\n")
            f.write("\n")
    print(f"已保存 C 语言一维数组（float32）到: {out_txt}")


if __name__ == '__main__':
    main()
