import scipy.io
import numpy as np
import os
print(os.path.getmtime('sys_matrices.mat'))
from scipy.signal import cont2discrete
# 加载文件
data = scipy.io.loadmat('sys_matrices.mat')
print(data.keys())
# 提取矩阵 (注意：loadmat 返回的是字典)


n = 100000
n=1
z = np.zeros((n, 24, 1))
u = np.zeros((n, 4, 1))
u_best = np.zeros((n, 4, 1))

P = np.zeros((n, 24, 24))
F = np.zeros((n, 4, 24))

q_diag = np.array([
    10,    # x: 100
    1,     # w: 10
    10000,   # theta: 5000 (身体俯仰是最重要的！)
    4000,    # theta_l: 500
    4000,    # theta_r: 500
    1,     # x_dot
    1,      # w_dot
    1,    # theta_dot
    1,     # theta_dot_l
    1      # theta_dot_r
])

S = np.diag(q_diag)
# np.array(
#             [[1,0,0,0,0,0,0,0,0,0],
#              [0,1,0,0,0,0,0,0,0,0],
#              [0,0,1,0,0,0,0,0,0,0],
#              [0,0,0,1,0,0,0,0,0,0],
#              [0,0,0,0,1,0,0,0,0,0],
#              [0,0,0,0,0,1,0,0,0,0],
#              [0,0,0,0,0,0,1,0,0,0],
#              [0,0,0,0,0,0,0,1,0,0],
#              [0,0,0,0,0,0,0,0,1,0],
#              [0,0,0,0,0,0,0,0,0,1]])



Q = S
# np.array(
#             [[1,0,0,0,0,0,0,0,0,0],
#              [0,1,0,0,0,0,0,0,0,0],
#              [0,0,1,0,0,0,0,0,0,0],
#              [0,0,0,1,0,0,0,0,0,0],
#              [0,0,0,0,1,0,0,0,0,0],
#              [0,0,0,0,0,1,0,0,0,0],
#              [0,0,0,0,0,0,1,0,0,0],
#              [0,0,0,0,0,0,0,1,0,0],
#              [0,0,0,0,0,0,0,0,1,0],
#              [0,0,0,0,0,0,0,0,0,1]])
R = 1*np.array(
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]])

J = np.zeros(n)
J_best = np.zeros(n)

A = data['A']
B = data['B']
C = data['C']
D = data['D']
system_discrete = cont2discrete((A, B, C, D), 0.002, method='zoh')
Ad, Bd, Cd, Dd, dt = system_discrete
print("A matrix shape:", Ad.shape)

At = np.eye(10)
I = np.eye(4)
# I = np.eye(10)



Aa = np.block([[Ad,np.zeros((10,10)),Bd],
      [np.zeros((10,10)),At,np.zeros((10,4))],
      [np.zeros((4,10)),np.zeros((4,10)),I]])
# Aa = np.block([[Ad,I-Ad],
#       [np.zeros((10,10)),I]])

Ba = np.block([[Bd],
      [np.zeros((10,4))],
      [I]])

# Ba = np.block([[Bd],
#       [np.zeros((10,4))]])

Ca = np.block([np.eye(10),-np.eye(10),np.zeros((10,4))])

# Ca = np.block([np.eye(10),-np.eye(10)])

print(A)
print(Ad)


Sa = Ca.T @ S @ Ca
Qa = Ca.T @ Q @ Ca
Ra = R

# P[n-1] = S
# J[n-1] = 1/2 * z[n-1].T @ P[n-1] @ z[n-1] 
# J_best[n-1]=1/2*(z[n-1].T @ P[n-1] @ z[n-1])

# J[n-2] = 1/2 * (A @ z[n-2] + B @ u[n-2]).T @ P[n-1] @ (A @ z[n-2] + B @ u[n-2]) + 1/2 * z[n-2].T @ Q @ z[n-2] + 1/2 * u[n-2].T @ R @ u[n-2]
# F[n-1] = np.linalg.inv(B.T @ P[n-1] @ B + R) @ B.T @ P[n-1] @ A
# u_best[n-1] = -F[n-1] @ z[n-1]
# P[n-2]=(A-B@F[n-1]).T @ P[n-1] @ (A-B@F[n-1])+F[n-1].T @ R @ F[n-1]+Q
# J_best[n-2]=1/2*(z[n-2].T @ P[n-2] @ z[n-2])


# def solve_riccati(A, B, Q, R, S, n=1000, tol=1e-6):
#     P[n-1] = S
#     F[n-1] = np.linalg.inv(B.T @ P[n-1] @ B + R) @ B.T @ P[n-1] @ A
#     for i in range(n-2, -1, -1):
#         P[i]=(A-B@F[i+1]).T @ P[i+1] @ (A-B@F[i+1])+F[i+1].T @ R @ F[i+1]+Q
#         F[i] = np.linalg.inv(B.T @ P[i] @ B + R) @ B.T @ P[i] @ A

#         # 计算差值矩阵
    
#         # 计算范数

#         diff_norm = np.linalg.norm(P[i] - P[i+1], 'fro')
#         ref_norm = np.linalg.norm(P[i+1], 'fro')
            
#         if ref_norm < 1e-15:
#             error = diff_norm  # 绝对误差
#         else:
#             error = diff_norm / ref_norm  # 相对误差
            
#             # 4. 检查收敛
#         if error < tol:
#             print(f"收敛于迭代 {n-i-1}, 误差 = {error:.2e}")
#             print(f"稳态P =\n{P[i]}")
#             print(f"稳态F =\n{F[i]}")
#             return P[i], F[i], True, i
        
#         if i==0:
#             print("错误: 未收敛")
#             return None, None, False, i
from scipy.linalg import solve_discrete_are




#[P_convergence,F_convergence,number]=solve_riccati(A, B, Q, R, S) 



# P_steady =Sa
P_steady =S
#F_steady
for i in range(0,n):
      # F_convergence= np.linalg.inv(Ba.T @ P_steady @ Ba + Ra) @ (Ba.T @ P_steady @ Aa)
      # P_steady = (Aa - Ba @ F_convergence).T @ P_steady @ (Aa - Ba @ F_convergence) + F_convergence.T @ Ra @ F_convergence + Qa
      F_convergence= np.linalg.inv(Bd.T @ P_steady @ Bd + R) @ (Bd.T @ P_steady @ Ad)
      P_steady = (Ad - Bd @ F_convergence).T @ P_steady @ (Ad - Bd @ F_convergence) + F_convergence.T @ R @ F_convergence + Q

print('F_convergence:')
print(F_convergence)


# 库函数计算结果
# 在 Python 中执行
# Ad_cl = Ad - Bd @ F_convergence
# e = np.linalg.eigvals(Ad_cl)
# print(f"闭环最大特征值: {np.max(np.abs(e))}")

