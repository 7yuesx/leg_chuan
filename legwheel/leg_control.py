from LQR import F_convergence, Ad, Bd, I 
import math
import numpy as np
import mujoco as mj
from mujoco import viewer
import time
from math import cos ,sin, pi
import socket
import struct
from scipy.spatial.transform import Rotation as R

from user_class import RobotSensor, Leg, State, LowPassFilter
from mymath import PID_control
from pynput import keyboard
from kalman import Kalman

# 控制状态
cmd_v = 0.0
cmd_omega_w = 0.0
cmd_hight = 0
jump = 0
flag = 0
def on_press(key):
    global cmd_v, cmd_omega_w, cmd_hight,jump,flag
    try:
        if key == keyboard.Key.up: cmd_v = 2.5
        if key == keyboard.Key.down: cmd_v = -2.5
        if key == keyboard.Key.left: cmd_omega_w = 3.0
        if key == keyboard.Key.right: cmd_omega_w = -3.0
        if key == keyboard.Key.shift_l:cmd_hight = 0.001
        if key == keyboard.Key.shift_r:cmd_hight = -0.001   
        if key == keyboard.Key.space:jump = 1
    except AttributeError:
        pass

def on_release(key):
    global cmd_v, cmd_omega_w, cmd_hight,jump,flag
    # 松开按键时归零
    cmd_v = 0.0
    cmd_omega_w = 0.0
    cmd_hight = 0
    jump = 0
    flag = 1
# 开启后台监听
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


UDP_IP = "127.0.0.1"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
vofa_tail = b'\x00\x00\x80\x7f'


    
model = mj.MjModel.from_xml_path("legwheel.xml")
data = mj.MjData(model)

dt = model.opt.timestep

sensor = RobotSensor(model, data)
robot_state = State()
road = Kalman()
left=Leg("left", dt)
right=Leg("right", dt)

left_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
right_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
left_length_vel = PID_control(kp=100, ki=0, kd=0, targ_value=0)
right_length_vel = PID_control(kp=100, ki=0, kd=0, targ_value=0)

roll_pid = PID_control(kp=2000, ki=0, kd=1000, targ_value=0)

# z_wheel_l_filter = LowPassFilter(alpha=0.8)
# z_wheel_r_filter = LowPassFilter(alpha=0.8)

x_target=0
w_target=0
x_dot_target=0
w_dot_target=0

leg_length=0.12
j=0

jumped_l=0
jumped_r=0

state_mode = 0

b = a = U_last = U = np.zeros((4,1))
U_out = np.zeros((6,1))
with mj.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()

        # 仿真主循环

        state = sensor.get_state()

        length = {
            "left_front_big": 0.21,
            "left_back_big": 0.21,
            "left_front_small": 0.2462962962962,
            "left_back_small": 0.2462962962962,
            "left_wheel":0.05,

            "right_front_big": 0.21,
            "right_back_big": 0.21,
            "right_front_small": 0.2462962962962,
            "right_back_small": 0.2462962962962,
            "right_wheel":0.05,
            
            "R":0.165,
        }

        imu = {
            "acc": state["acc"],
            "gyro": state["gyro"],
            "euler": state["euler"],
            "quat": state["quat"]
        }

        motor = {
            "left_front_pos": state["joints"]["motor_l1_pos"],
            "left_back_pos": state["joints"]["motor_l2_pos"],
            "left_wheel_pos": state["joints"]["wheel_l_pos"],

            "right_front_pos": state["joints"]["motor_r1_pos"],
            "right_back_pos": state["joints"]["motor_r2_pos"],           
            "right_wheel_pos": state["joints"]["wheel_r_pos"],

            "left_front_vel": state["joints"]["motor_l1_vel"],
            "left_back_vel": state["joints"]["motor_l2_vel"],
            "left_wheel_vel": state["joints"]["wheel_l_vel"],

            "right_front_vel": state["joints"]["motor_r1_vel"], 
            "right_back_vel": state["joints"]["motor_r2_vel"],
            "right_wheel_vel": state["joints"]["wheel_r_vel"],
        }    

        left.forward(motor, imu, length, robot_state.x[2,0])
        right.forward(motor, imu, length, robot_state.x[2,0])

        F_c=F_convergence.copy()
        print(F_c)
        print(F_convergence)
        x_dot_target=cmd_v
        w_dot_target=cmd_omega_w

        x_target+=x_dot_target*dt
        w_target+=w_dot_target*dt

        leg_length+=cmd_hight

        if leg_length<0.12:
            leg_length=0.12
        if leg_length>0.3:
            leg_length=0.3

        max1=1500
        max2=1000
        max3=1000

        jumped_r=0
        jumped_l=0
        G_m=109

        left_target=0.0
        right_target=0.0
        
        left_length_pos.position_pid(leg_length,left.length)
        left_length_vel.position_pid(0,left.length_dot.Diff(left.length))
        right_length_pos.position_pid(leg_length,right.length)
        right_length_vel.position_pid(0,right.length_dot.Diff(right.length))
        roll_pid.position_pid(0,imu["euler"][0])


        if right.length<0.4 and jump and flag:
            jumped_r=5000
            max1=0
            max2=0
            
        elif left.length>=0.35 and jump:
            jumped_r=0
            flag=0
            max2=0
            max3=0
            G_m=0
            leg_length=0.1
        elif left.length<=0.11 and jump:
            jumped_r=0
            flag=0
            max3=0
            leg_length=0.12

        if left.length<0.4 and jump and flag:
            jumped_l=5000
            max1=0
            max2=0
            # leg_length=0.4
        elif left.length>=0.35 and jump:
            jumped_l=0
            flag=0
            G_m=0
            max2=0
            max3=0
            leg_length=0.1

        elif left.length<=0.11 and jump:
            jumped_r=0
            flag=0
            max3=0
            leg_length=0.12

        if left_length_pos.output>max1:
            left_length_pos.output=max1
        if left_length_pos.output<-max1:
            left_length_pos.output=-max1
        if right_length_pos.output>max1:
            right_length_pos.output=max1
        if right_length_pos.output<-max1:
            right_length_pos.output=-max1

        if left_length_vel.output>max2:
            left_length_vel.output=max2
        if left_length_vel.output<-max2:
            left_length_vel.output=-max2
        if right_length_vel.output>max2 :
            right_length_vel.output=max2
        if right_length_vel.output<-max2:
            right_length_vel.output=-max2
        
        if roll_pid.output>max3:
            roll_pid.output=max3
        if roll_pid.output<-max3:
            roll_pid.output=-max3

        z_body=sensor.body_acc(imu)
        road.filter(z_body,motor,length,dt)    

        m_l=m_r=0.353429173529
        g=9.81

        F_l=left_length_vel.output+roll_pid.output+left_length_pos.output+G_m*cos(robot_state.x[3,0])+jumped_l
        F_r=right_length_vel.output-roll_pid.output+right_length_pos.output+G_m*cos(robot_state.x[4,0])+jumped_r

        P_l=F_l*cos(robot_state.x[3,0])+U[0,0]*sin(robot_state.x[3,0])/left.length
        P_r=F_r*cos(robot_state.x[4,0])+U[1,0]*sin(robot_state.x[4,0])/right.length

        z_wheel_l=z_body[2]-left.length_ddot.Diff(left.length_dot.diff_num)*cos(left.theta)+2*left.length_dot.diff_num*left.theta_dot.diff_num*sin(left.theta)+left.length*left.theta_ddot.Diff(left.theta_dot.diff_num)*sin(left.theta)+left.length*left.theta_dot.diff_num**2*cos(left.theta)
        z_wheel_r=z_body[2]-right.length_ddot.Diff(right.length_dot.diff_num)*cos(right.theta)+2*right.length_dot.diff_num*right.theta_dot.diff_num*sin(right.theta)+right.length*right.theta_ddot.Diff(right.theta_dot.diff_num)*sin(right.theta)+right.length*right.theta_dot.diff_num**2*cos(right.theta)

        Fn_l=P_l+m_l*g+z_wheel_l*m_l    
        Fn_r=P_r+m_r*g+z_wheel_r*m_r


        if(Fn_l<20) :
            F_c[0,0]=0
            F_c[0,1]=0
            F_c[0,2]=0
            F_c[0,4]=0

            F_c[0,5]=0
            F_c[0,6]=0
            F_c[0,7]=0
            F_c[0,9]=0
            
            F_c[2,0]=0
            F_c[2,1]=0
            F_c[2,2]=0
            F_c[2,3]=0
            F_c[2,4]=0

            F_c[2,5]=0
            F_c[2,6]=0
            F_c[2,7]=0
            F_c[2,8]=0
            F_c[2,9]=0

        if(Fn_r<20) : 
            F_c[1,0]=0
            F_c[1,1]=0
            F_c[1,2]=0
            F_c[1,3]=0

            F_c[1,5]=0
            F_c[1,6]=0
            F_c[1,7]=0
            F_c[1,8]=0

            F_c[3,0]=0
            F_c[3,1]=0
            F_c[3,2]=0
            F_c[3,3]=0
            F_c[3,4]=0

            F_c[3,5]=0
            F_c[3,6]=0
            F_c[3,7]=0
            F_c[3,8]=0
            F_c[3,9]=0

        # U=U_last-F_convergence @ (np.block([[robot_state.update_state(left, right, length, motor,imu, road)],
        #                     [robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)],
        #                     [robot_state.update_control(U_last)]]))

        # U_d=Bd.T @ (I-Ad)@robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)
        # U=U_d-F_convergence @ (np.block([[robot_state.update_state(left, right, length, motor, imu, road)],
        #                     [robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)]]))

        U=-F_c @ (robot_state.update_state(left, right, length, motor,imu, road)-robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target, left_target, right_target))
    


        left.vmc(F_l+12, U[0,0], motor, length)
        right.vmc(F_r+12, U[1,0], motor, length)


        U_out = np.block([[left.t_front],[right.t_front],[left.t_back],[right.t_back],[U[2,0]],[U[3,0]]])

        if abs(imu["euler"][0])>1.0 or abs(imu["euler"][1])>1.0 :
            U_out[0,0]=0
            U_out[1,0]=0
            U_out[2,0]=0
            U_out[3,0]=0
            U_out[4,0]=0
            U_out[5,0]=0
            
        # if (abs(robot_state.x[3,0])>1.5 or abs(robot_state.x[4,0])>1.5) :
        #     U_out[0,0]=0
        #     U_out[1,0]=0
        #     U_out[2,0]=0
        #     U_out[3,0]=0
        #     U_out[4,0]=0
        #     U_out[5,0]=0  
      

        

        send_list = [
            robot_state.x[0,0],
            robot_state.x[1,0],
            robot_state.x[2,0],
            robot_state.x[3,0],
            robot_state.x[4,0],
            robot_state.x[5,0],
            robot_state.x[6,0],
            robot_state.x[7,0],
            robot_state.x[8,0],
            robot_state.x[9,0],
            robot_state.x_target[0,0],
            robot_state.x_target[1,0],
            robot_state.x_target[2,0],
            robot_state.x_target[3,0],
            robot_state.x_target[4,0],
            robot_state.x_target[5,0],
            robot_state.x_target[6,0],
            robot_state.x_target[7,0],
            robot_state.x_target[8,0],
            robot_state.x_target[9,0],
            left.length,
            right.length,
            Fn_l,
            Fn_r,  
        ]
        


        # 3. 数据打包为二进制 (f 代表 float, < 代表小端)
        # 这里的 'ffff' 代表有 4 个 float 数据
        binary_data = struct.pack('<' + 'f' * len(send_list), *send_list)
        sock.sendto(binary_data + vofa_tail, (UDP_IP, UDP_PORT))

        if U_out[0,0]>35:
            U_out[0,0]=35
        if U_out[0,0]<-35:
            U_out[0,0]=-35
        if U_out[1,0]>35:
            U_out[1,0]=35
        if U_out[1,0]<-35:  
            U_out[1,0]=-35
        if U_out[2,0]>35:
            U_out[2,0]=35
        if U_out[2,0]<-35:
            U_out[2,0]=-35
        if U_out[3,0]>35:
            U_out[3,0]=35
        if U_out[3,0]<-35:
            U_out[3,0]=-35
        if U_out[4,0]>5:
            U_out[4,0]=5
        if U_out[4,0]<-5:
            U_out[4,0]=-5
        if U_out[5,0]>5:
            U_out[5,0]=5
        if U_out[5,0]<-5:
            U_out[5,0]=-5
   

        for i in range(6):
            data.ctrl[i] =  U_out[i,0]   
        # 3. 物理引擎向前推进一步

        mj.mj_step(model, data)
        
        # 3. 刷新渲染器 (这里通常对应显示器的刷新频率)
        viewer.sync()
        U_last=U
        # 适当微休眠，防止 CPU 占用率 100%
        time.sleep(dt*1)