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