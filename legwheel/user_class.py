import math
from math import cos, sin, atan, atan2, sqrt, pi


import numpy as np

class Discreteness(object):

    def __init__(self, dt):
        self.dt = dt
        self.last_diff = 0.0
        self.last_sum = 0.0
        self.diff_num = 0.0
        self.sum_num = 0.0

    def Sum(self, s_num):
        self.sum_num = self.last_sum + self.dt * s_num
        self.last_sum = self.sum_num
        return self.sum_num

    def Diff(self, d_num):
        self.diff_num = (d_num - self.last_diff) / self.dt
        self.last_diff = d_num
        return self.diff_num

class RobotSensor:
    def __init__(self, model, data):
        """
        model: mujoco.MjModel
        data: mujoco.MjData
        """
        self.model = model
        self.data = data

    # ---------------------------
    # IMU
    # ---------------------------
    def get_orientation_quat(self):
        """Return quaternion [w, x, y, z]"""
        return self.data.sensor("perfect_quat").data.copy()

    def get_acc(self):
        """Return linear acceleration (m/s^2)"""
        return self.data.sensor("body_accel").data.copy()

    def get_gyro(self):
        """Return angular velocity (rad/s)"""
        return self.data.sensor("body_gyro").data.copy()

    # ---------------------------
    # Joint positions
    # ---------------------------
    def get_joint_positions(self):
        """Return all joint positions as dict"""
        return {
            "motor_l1_pos": self.data.sensor("motor_l1_pos").data[0],
            "motor_l2_pos": self.data.sensor("motor_l2_pos").data[0],
            "wheel_l_pos": self.data.sensor("wheel_l_pos").data[0],

            "motor_r1_pos": self.data.sensor("motor_r1_pos").data[0],
            "motor_r2_pos": self.data.sensor("motor_r2_pos").data[0],
            "wheel_r_pos": self.data.sensor("wheel_r_pos").data[0],

            "motor_l1_vel": self.data.sensor("motor_l1_vel").data[0],
            "motor_l2_vel": self.data.sensor("motor_l2_vel").data[0],
            "wheel_l_vel": self.data.sensor("wheel_l_vel").data[0],

            "motor_r1_vel": self.data.sensor("motor_r1_vel").data[0],
            "motor_r2_vel": self.data.sensor("motor_r2_vel").data[0],
            "wheel_r_vel": self.data.sensor("wheel_r_vel").data[0],
        }

    # ---------------------------
    # Combined state (recommended)
    # ---------------------------
    def get_state(self):
        """Return full robot state"""
        return {
            "quat": self.get_orientation_quat(),
            "acc": self.get_acc(),
            "gyro": self.get_gyro(),
            "joints": self.get_joint_positions(),
            "euler": self.quat_to_euler(self.get_orientation_quat()),
        }

    # ---------------------------
    # Optional: quaternion → euler
    # ---------------------------
    def quat_to_euler(self, quat):
        """Convert quaternion [w,x,y,z] → roll, pitch, yaw"""
        w, x, y, z = quat

        # roll (x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # yaw (z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
    def body_acc(self, imu):
        w=imu["quat"][0]
        x=imu["quat"][1]
        y=imu["quat"][2]
        z=imu["quat"][3]
        
        R=np.array([[1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
                   [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
                   [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]])    
        S=np.array([imu["acc"][0],imu["acc"][1],imu["acc"][2]])
        z_body=R@S-np.array([0,0,9.81])
        return z_body
######################

class Leg:
    def __init__(self, name, dt):
        self.name = name
        self.length = 0.0
        self.length_dot = Discreteness(dt)
        self.length_ddot = Discreteness(dt)
        self.theta = 0.0
        self.theta_dot = Discreteness(dt)
        self.theta_ddot = Discreteness(dt)

        self.phi_front = 0.0
        self.phi_back = 0.0
        self.phi_front_dot = Discreteness(dt)
        self.phi_back_dot = Discreteness(dt)

        self.t_front = 0.0
        self.t_back = 0.0

        self.f_l = 0.0
        self.t_l = 0.0

        self.x_vel = 0.0
        self.y_vel = 0.0

    def forward(self, motor, imu, length, Leg):

        theta_front = motor[self.name + "_front_pos"]
        theta_back = motor[self.name + "_back_pos"]
        lenth_base = 0.0
        length_front_big = length[self.name + "_front_big"]
        length_front_small = length[self.name + "_front_small"]
        length_back_big = length[self.name + "_back_big"]
        length_back_small = length[self.name + "_back_small"]

        x_a = -lenth_base# * cos(imu["euler"][1])
        y_a = 0#-lenth_base * sin(imu["euler"][1])
        x_e = lenth_base# * cos(imu["euler"][1])
        y_e = 0#lenth_base * sin(imu["euler"][1])

        x_b = x_a + length_back_big * cos(theta_back)
        y_b = y_a + length_back_big * sin(theta_back)

        x_d = x_e + length_front_big * cos(theta_front)
        y_d = y_e + length_front_big * sin(theta_front)

        l_bd = sqrt((x_b - x_d) ** 2 + (y_b - y_d) ** 2)
        A = 2*length_back_small*(x_d-x_b)
        B = 2*length_back_small*(y_d-y_b)
        C = length_back_small**2 + l_bd**2 - length_front_small**2   

        self.phi_back = 2*atan2(B+sqrt(A**2 + B**2 - C**2), A+C)

        x_c = x_b + length_back_small * cos(self.phi_back)
        y_c = y_b + length_back_small * sin(self.phi_back)

        self.phi_front = np.arctan2(y_c - y_d, x_c - x_d)

        self.length = sqrt(x_c ** 2 + y_c ** 2)
        self.theta = (-atan2(x_c, y_c)+imu["euler"][1])

        self.x_vel = x_c
        self.y_vel = y_c

    def vmc(self, f_l, t_l, motor, length):

        self.f_l=f_l
        self.t_l=t_l
        theta_front = motor[self.name + "_front_pos"]
        theta_back = motor[self.name + "_back_pos"]

        theta_front_vel = motor[self.name + "_front_vel"]
        theta_back_vel = motor[self.name + "_back_vel"]

        length_front_big = length[self.name + "_front_big"]
        length_back_big = length[self.name + "_back_big"]

        self.t_back = self.f_l * length_back_big * cos(self.theta - self.phi_front)*sin(theta_back - self.phi_back)/sin(self.phi_front - self.phi_back)+self.t_l * length_back_big * -sin(self.theta - self.phi_front)*sin(theta_back - self.phi_back)/(self.length*sin(self.phi_front - self.phi_back))
                        
        self.t_front = self.f_l * length_front_big * cos(self.theta - self.phi_back)*sin(self.phi_front - theta_front)/sin(self.phi_front - self.phi_back)+self.t_l * length_front_big * -sin(self.theta - self.phi_back)*sin(self.phi_front - theta_front )/(self.length*sin(self.phi_front - self.phi_back))    
    

        # self.t_back = self.f_l * length_back_big * sin(self.theta - self.phi_front)*sin(theta_back - self.phi_back)/sin(self.phi_front - self.phi_back)+self.t_l * length_back_big * cos(self.theta - self.phi_front)*sin(theta_back - self.phi_back)/(self.length*sin(self.phi_front - self.phi_back))
                        
        # self.t_front = self.f_l * length_front_big * sin(self.theta - self.phi_back)*sin(theta_front - self.phi_front)/sin(self.phi_front - self.phi_back)+self.t_l * length_front_big * cos(self.theta - self.phi_back)*sin(theta_front - self.phi_front)/(self.length*sin(self.phi_front - self.phi_back))    
    
        # self.x_vel = theta_back_vel*length_back_big*sin(theta_back - self.phi_back)*sin(self.phi_front)/sin(self.phi_back-self.phi_front) + ... 
        # theta_front_vel*length_front_big*sin(theta_front - self.phi_front)*sin(self.phi_back)/sin(self.phi_back-self.phi_front)
        
        # self.y_vel = -theta_back_vel*length_back_big*sin(theta_back - self.phi_back)*cos(self.phi_front)/sin(self.phi_back-self.phi_front) - ... 
        # theta_front_vel*length_front_big*sin(theta_front - self.phi_front)*cos(self.phi_back)/sin(self.phi_back-self.phi_front)

class State:
    def __init__(self):
        self.x = np.zeros((10, 1)) # x, w, theta, theta_l, theta_r, x_dot, w_dot, theta_dot, theta_l_dot, theta_r_dot

        self.x_target = np.zeros((10, 1))
        
        self.u = np.zeros((4, 1)) # 
    def update_state(self, Leg_left, Leg_right, length, motor,imu, road):
        
        Leg_left.theta_dot.Diff(Leg_left.theta)
        Leg_right.theta_dot.Diff(Leg_right.theta)

        self.x[0,0] = road.x[0,0]#(length["left_wheel"]*motor["left_wheel_pos"] + length["right_wheel"]*motor["right_wheel_pos"])*0.5#+Leg_left.theta*Leg_left.length+Leg_right.theta*Leg_right.length
        self.x[1,0] = imu["euler"][2]#(-length["left_wheel"]*motor["left_wheel_pos"] + length["right_wheel"]*motor["right_wheel_pos"])*0.5/length["R"]
        self.x[2,0] = imu["euler"][1]# self.x[7,0]*dt#
        self.x[3,0] = Leg_left.theta
        self.x[4,0] = Leg_right.theta

        self.x[5,0] = road.x[1,0]#(length["left_wheel"]*motor["left_wheel_vel"] + length["right_wheel"]*motor["right_wheel_vel"])*0.5#+Leg_left.theta_dot.diff_num*Leg_left.length+Leg_right.theta_dot.diff_num*Leg_right.length
        self.x[6,0] = imu["gyro"][2]#(-length["left_wheel"]*motor["left_wheel_vel"] + length["right_wheel"]*motor["right_wheel_vel"])*0.5/length["R"]
        self.x[7,0] = imu["gyro"][1]
        self.x[8,0] = Leg_left.theta_dot.diff_num
        self.x[9,0] = Leg_right.theta_dot.diff_num
        return self.x
    def update_target(self, x_target, w_target, x_dot_target, w_dot_target, theta_target, left_target, right_target):

        self.x_target[0,0] = x_target
        self.x_target[1,0] = w_target
        self.x_target[2,0] = theta_target
        self.x_target[3,0] = left_target
        self.x_target[4,0] = right_target
        self.x_target[5,0] = x_dot_target
        self.x_target[6,0] = w_dot_target
        self.x_target[7,0] = 0.0
        self.x_target[8,0] = 0.0
        self.x_target[9,0] = 0.0
        return self.x_target
    def update_control(self, u):

        self.u[0,0] = u[0]
        self.u[1,0] = u[1]
        self.u[2,0] = u[2]
        self.u[3,0] = u[3]
        return self.u

        

class LowPassFilter:
    def __init__(self, alpha=0.1):
        """
        初始化低通滤波器
        :param alpha: 滤波系数 (0.0 < alpha <= 1.0)
                      值越小，平滑度越高，延迟越大。
                      值越大，灵敏度越高，噪声越多。
        """
        self.alpha = alpha
        self.last_output = None  # 存储上一次的滤波结果

    def update(self, current_input):
        """
        计算新的滤波值
        :param current_input: 当前传感器读到的原始值
        :return: 滤波后的值
        """
        # 第一次运行时，直接使用当前值作为初始值
        if self.last_output is None:
            self.last_output = current_input
            return current_input

        # 核心算法：一阶低通滤波
        # 新值 = α * 当前输入 + (1 - α) * 上次输出
        filtered_value = (self.alpha * current_input) + \
                         ((1 - self.alpha) * self.last_output)
        
        # 更新状态
        self.last_output = filtered_value
        
        return filtered_value

    def reset(self):
        """重置滤波器"""
        self.last_output = None

class Change_length_fit:
    def __init__(self, K=np.zeros((4, 10)),K_coeff=np.zeros((4, 10, 5)),theta_coeff=np.zeros((4)),e1_coeff=np.zeros((5)),e2_coeff=np.zeros((5)),e3_coeff=np.zeros((5))):
        self.K = K
        self.K_coeff = K_coeff
        self.theta_coeff = theta_coeff
        self.e1 = e1_coeff
        self.e2 = e2_coeff
        self.e3 = e3_coeff
    def get_K(self, leg_length):
        # Example implementation - replace with actual logic
        for i in range(4):
            for j in range(10):
                self.K[i, j] = self.K_coeff[i, j, 0] * leg_length**4    + self.K_coeff[i, j, 1] * leg_length**3    + self.K_coeff[i, j, 2] * leg_length**2    + self.K_coeff[i, j, 3] * leg_length    + self.K_coeff[i, j, 4]
        return self.K
    def get_theta_target(self, leg_length):
        # Example implementation - replace with actual logic
        theta_target = self.theta_coeff[0] * leg_length**4+ self.theta_coeff[1] * leg_length**3+ self.theta_coeff[2] * leg_length**2+ self.theta_coeff[3] * leg_length+ self.theta_coeff[4]
        return theta_target
    def get_e1(self, leg_length):
        # Example implementation - replace with actual logic
        e1 = self.e1[0] * leg_length**4+ self.e1[1] * leg_length**3+ self.e1[2] * leg_length**2+ self.e1[3] * leg_length+ self.e1[4]
        return e1
    def get_e2(self, leg_length):
        # Example implementation - replace with actual logic
        e2 = self.e2[0] * leg_length**4+ self.e2[1] * leg_length**3+ self.e2[2] * leg_length**2+ self.e2[3] * leg_length+ self.e2[4]
        return e2
    def get_e3(self, leg_length):
        # Example implementation - replace with actual logic
        e3 = self.e3[0] * leg_length**4+ self.e3[1] * leg_length**3+ self.e3[2] * leg_length**2+ self.e3[3] * leg_length+ self.e3[4]
        return e3
    
class Mpc:
    def __init__(self,dt,K_coeff,P_coeff,Q_uu_coeff,A_coeff,B_coeff,AmBK_coeff,f_coeff,APf_coeff,BPf_coeff,Le_coeff,theta_coeff,theta_l_coeff,theta_r_coeff,N,rho,state,input):

        self.N = N
        self.state_num = state
        self.input_num = input
        self.rho = rho

        self.K_coeff = K_coeff
        self.P_coeff = P_coeff
        self.Q_uu_coeff = Q_uu_coeff
        self.A_coeff = A_coeff
        self.B_coeff = B_coeff
        self.AmBK_T_coeff = AmBK_coeff
        self.f_coeff = f_coeff
        self.APf_coeff = APf_coeff
        self.BPf_coeff = BPf_coeff

        self.Le_coeff = Le_coeff


        self.theta_coeff = theta_coeff
        self.theta_l_coeff = theta_l_coeff
        self.theta_r_coeff = theta_r_coeff

        # self.K_ = np.zeros((input,state))
        self.K = np.zeros((input,state))
        self.P = np.zeros((state,state))
        self.Q = np.array([100.0, 1.0, 5000.0, 4000.0, 4000.0,
                    1.0, 1.0, 1.0, 1.0, 1.0])
        self.R = np.array([1.0, 1.0, 1.0, 1.0])
        self.Q_uu = np.zeros((input,input))
        self.A = np.zeros((state,state))
        self.B = np.zeros((state,input))
        self.AmBK_T = np.zeros((state,state))
        self.f = np.zeros((state))
        self.APf = np.zeros((state))
        self.BPf = np.zeros((input))

        self.d = np.zeros((N-1,input))
        self.q = np.zeros((N-1,state))
        self.r = np.zeros((N-1,input))
        self.p = np.zeros((N,state))
        self.u = np.zeros((N-1,input))
        self.x = np.zeros((N,state))
        self.v = np.zeros((N,state))
        self.z = np.zeros((N-1,input))
        self.g= np.zeros((N,state))
        self.y= np.zeros((N-1,input))


        # self.H = np.zeros((state*(N+1)+input*N,state*(N+1)+input*N))
        self.u_min=np.array([-50.0,-50.0,-6.0,-6.0])
        self.u_max=np.array([50.0,50.0,6.0,6.0])

        self.x_min=np.vstack([-0.5,-1.5,-1.5])
        self.x_max=np.vstack([0.5,1.5,1.5])

        # self.X = np.zeros((state+input,1))
        # self.y = np.zeros((state,1))
        # self.u = np.zeros((input,1))
        # self.C = np.block([np.eye(state), np.zeros((state,input))])  # 10x14
        self.Le = np.zeros((state+input,state))
    # def leso(self, x, u):
    #     self.y[0:10,0] = x[0:10,0]
    #     self.u = u
    #     Ae=np.block([[self.A,self.B],
    #                  [np.zeros((4,10)),np.eye(4)]]) 
    #     Be=np.vstack([self.B,np.zeros((4,4))])

    #     self.X = Ae@self.X+Be@self.u+self.Le@(self.y-self.C@self.X)

    #     return self.X[10:14,0:1]
    #     pass

    def get_matrix(self,leg_lenth_l,leg_lenth_r):
        x = leg_lenth_l
        y = leg_lenth_r
        # 展平系数为 (40, 15) 矩阵
        coeffs_flat_k = self.K_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_p = self.P_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_a = self.A_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_b = self.B_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_q_uu = self.Q_uu_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_ambk_t = self.AmBK_T_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_f = self.f_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_apf = self.APf_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_bpf = self.BPf_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_theta = self.theta_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_theta_l = self.theta_l_coeff.reshape(-1, 10)  # 视图，无拷贝
        coeffs_flat_theta_r = self.theta_r_coeff.reshape(-1, 10)
        coeffs_flat_le = self.Le_coeff.reshape(-1, 10)  # 视图，无拷贝
        
        # 构造特征向量 [1, x, y, x², x*y, y²]
        x2 = x * x
        y2 = y * y
        xy = x * y
        x3 = x * x2
        x2y = x2 * y
        xy2 = x * y2
        y3 = y * y2
        x4 = x * x3
        x3y = x3 * y
        x2y2 = x2 * y2
        xy3 = x * y3
        y4 = y * y3
        features = np.array([1.0, x, y, x2, xy, y2, x3, x2y, xy2, y3], dtype=np.float32)#, x4, x3y, x2y2, xy3, y4

        # 矩阵乘法：40×6 与 6×1 得到 40×1，再 reshape 为 4×10
        # 使用 @ 运算符（或 np.dot）在 C 级完成，极快
        self.K[:, :] = (coeffs_flat_k @ features).reshape(4, 10)
        self.P[:, :] = (coeffs_flat_p @ features).reshape(10, 10)
        self.A[:, :] = (coeffs_flat_a @ features).reshape(10, 10)
        self.B[:, :] = (coeffs_flat_b @ features).reshape(10, 4)
        self.Q_uu[:, :] = (coeffs_flat_q_uu @ features).reshape(4, 4)
        self.AmBK_T[:, :] = (coeffs_flat_ambk_t @ features).reshape(10, 10)
        self.f = (coeffs_flat_f @ features).reshape(10)
        self.APf = (coeffs_flat_apf @ features).reshape(10)
        self.BPf = (coeffs_flat_bpf @ features).reshape(4)
        theta = (coeffs_flat_theta @ features).reshape(1)
        theta_l = (coeffs_flat_theta_l @ features).reshape(1)
        theta_r = (coeffs_flat_theta_r @ features).reshape(1)
        self.Le[:, :] = (coeffs_flat_le @ features).reshape(14, 10)
        

        return theta ,theta_l, theta_r, self.B,self.K
    def updata_ref(self):
        
        pass

    def ADMM(self, x_measure, x_ref, u_ref, eps_pri=1, eps_dua=1):
        self.x[0] = np.asarray(x_measure).flatten() 
        for j in range(1):
            self.p[self.N-1]=-self.P@x_ref[self.N-1]-self.rho*(self.v[self.N-1]-self.g[self.N-1])
            for i in range(self.N-2,-1,-1):
                self.q[i] = -self.Q * x_ref[i] - self.rho * (self.v[i] - self.g[i])
                self.r[i] = -self.R * u_ref[i] - self.rho * (self.z[i] - self.y[i])
                self.d[i]=self.Q_uu@(self.B.T@self.p[i+1]+self.r[i]+self.BPf)
                self.p[i] = self.AmBK_T@self.p[i+1]+self.q[i]+self.APf-self.K.T@self.r[i]
            for i in range(self.N-1):
                self.u[i] = -self.K@self.x[i]-self.d[i]
                self.x[i+1] = self.A@self.x[i]+self.B@self.u[i]+self.f
            for i in range(self.N):
                self.v[i] = self.x[i]+self.g[i]#np.clip(self.x[i]+self.g[i], self.x_min, self.x_max)
            for i in range(self.N-1):
                self.z[i] = np.clip(self.u[i]+self.y[i], self.u_min, self.u_max)

            for i in range(self.N):
                self.g[i] += self.x[i] - self.v[i]
            for i in range(self.N - 1):
                self.y[i] += self.u[i] - self.z[i]

            pri_x = np.max(np.abs(self.x - self.v))
            pri_u = np.max(np.abs(self.u - self.z))
            dua_x = self.rho * np.max(np.abs(self.v - self.v))
            dua_u = self.rho * np.max(np.abs(self.z - self.z))

            # if (pri_x < eps_pri and pri_u < eps_pri
            #         and dua_x < eps_dua and dua_u < eps_dua):
        return self.u#, True#self.x, 
