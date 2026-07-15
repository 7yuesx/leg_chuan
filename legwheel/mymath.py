import numpy as np


# 离散积分差分
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


# PID
class PID_control(object):
    def __init__(self, kp, ki, kd, targ_value):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.targ_value = targ_value

        self.computed_input = 0.0
        self.err = 0.0
        self.last_err = 0.0

        self.last_time = 0.0
        self.sum_err = 0.0
        self.theta_last = 0.0

        self.max_integral = 20000
        self.max_output = 300
        self.err_LpfRatio = 1

        self.integral = 0.0

        self.output = 0.0

    def position_pid(self, targ ,feedback_value):
        # single step duration
        self.last_err = self.err
        self.targ_value = targ # 增加的
        self.err = self.targ_value - feedback_value

        self.err = self.err * self.err_LpfRatio + self.last_err * (1 - self.err_LpfRatio)
        self.output = (self.err - self.last_err) * self.kd
        self.output += self.err * self.kp
        self.integral += self.err * self.ki

        return self.output 