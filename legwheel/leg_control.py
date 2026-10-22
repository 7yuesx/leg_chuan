# from LQR import F_convergence, Ad, Bd, I 
import math
import numpy as np
import mujoco as mj
from mujoco import viewer
import time
from math import cos ,sin, pi
import socket
import struct
from scipy.spatial.transform import Rotation as R

from user_class import RobotSensor, Leg, State, Mpc
from mymath import PID_control
from pynput import keyboard
from kalman import Kalman
from coeff import A_coeff, B_coeff, K_coeff, P_coeff, Quu_inv_coeff, AmBK_T_coeff,f_coeff,APf_coeff,BPf_coeff ,Le_coeff,theta_coeff, theta_l_coeff, theta_r_coeff

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



model = mj.MjModel.from_xml_path("legwheel.xml")#legwheel  mjmodel
data = mj.MjData(model)

dt = model.opt.timestep

sensor = RobotSensor(model, data)
robot_state = State()
road = Kalman()
left=Leg("left", dt)
right=Leg("right", dt)

left_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
right_length_pos = PID_control(kp=2000, ki=0.0, kd=0, targ_value=0)
left_length_vel = PID_control(kp=125, ki=0, kd=0, targ_value=0)
right_length_vel = PID_control(kp=125, ki=0, kd=0, targ_value=0)

roll_pid = PID_control(kp=2000, ki=0, kd=1000, targ_value=0)

robot_mpc = Mpc(dt, K_coeff, P_coeff, Quu_inv_coeff, A_coeff, B_coeff, AmBK_T_coeff,f_coeff,APf_coeff,BPf_coeff,Le_coeff,theta_coeff, theta_l_coeff, theta_r_coeff, N=10, rho=2, state=10, input=4)
times = 1
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
# k_coeff=np.array([[[25.6845,-32.9517,17.6266,-5.2228,0.9282], [-1.7020,1.5833,-0.2756,-0.2794,-05068], [785.6561,-943.4036,442.6694,-100.0401,-57.7757], [-77.5215,32.2734,21.4093,-29.0478,26.8365], [191.0312,-291.7503,193.5358,-63.2923,-4.5981], [273.4761,-297.2553,129.2962,-29.1983,3.6442], [1.7838,-2.2798,1.3710,-0.6637,-0.5402], [47.8281,-56.0437,25.4980,-5.5514,-2.8019], [11.5037,-13.7858,6.6572,-0.6661,0.6363], [0.2029,0.5962,-0.7440,-0.8760,-0.0851]],
#                   [[25.6845,-32.9517,17.6266,-5.2228,0.9282], [1.7020,-1.5833,0.2756,0.2794,05068],  [785.6561,-943.4036,442.6694,-100.0401,-57.7757], [191.0312,-291.7503,193.5358,-63.2923,-4.5981], [-77.5215,32.2734,21.4093,-29.0478,26.8365], [273.4761,-297.2553,129.2962,-29.1983,3.6442], [-1.7838,2.2798,-1.3710,0.6637,0.5402], [47.8281,-56.0437,25.4980,-5.5514,-2.8019], [0.2029,0.5962,-0.7440,-0.8760,-0.0851], [11.5037,-13.7858,6.6572,-0.6661,0.6363]],
#                   [[16.7613,-26.1043,16.5385,-5.4239,-1.2086], [-1.1272,1.3905,-0.8866,0.5448,-0.5014], [-1576.4,1993.5,-1024.5,277.0579,-41.4832], [275.1361,-472.8729,338.1066,-132.4359,-34.0195], [505.5580,-622.1571,292.9324,-61.5938,-2.6412], [-129.8865,129.2661,-43.8260,4.2654,-4.9732], [-1.0609,1.3058,-0.8384,0.5435,-0.5532], [-118.4951,145.3071,-71.6176,18.3392,-2.6412],[-19.9594,22.6431,-9.3670,-3.1116,-0.7461],[-9.9819,9.1353,-2.4167,-2.4995,-0.0813]],
#                   [[16.7613,-26.1043,16.5385,-5.4239,-1.2086], [1.1272,-1.3905,0.8866,-0.5448,0.5014], [-1576.4,1993.5,-1024.5,277.0579,-41.4832], [505.5580,-622.1571,292.9324,-61.5938,-2.6412], [275.1361,-472.8729,338.1066,-132.4359,-34.0195], [-129.8865,129.2661,-43.8260,4.2654,-4.9732], [1.0609,-1.3058,0.8384,-0.5435,0.5532], [-118.4951,145.3071,-71.6176,18.3392,-2.6412],[-9.9819,9.1353,-2.4167,-2.4995,-0.0813],[-19.9594,22.6431,-9.3670,-3.1116,-0.7461]]])

# k_coeff=np.array([[[6.2174,-11.2883,8.3893,-3.4140,0.7898], [1.4340,-1.8905,1.1834,-0.5364,-0.4896], [923.8710,-1092.6,502.6034,-110.6934,-57.0662], [364.5066,-420.9464,197.7163,-61.6707,29.2380], [347.2482,-494.2305,293.6293,-84.8201,-2.8636], [172.8698,-205.0662,97.0404,-23.7418,3.2421], [-2.1572,2.0367,-0.4234,-0.3492,-0.5612], [39.0661,-46.6079,21.7275,-4.8873,-2.8456], [16.1460,-16.5183,6.9372,-0.5876,0.6249], [13.1891,-10.1115,2.2276,-1.2081,-0.0720]],
#                   [[6.2174,-11.2883,8.3893,-3.4140,0.7898], [-1.4340,1.8905,-1.1834,0.5364,0.4896], [923.8710,-1092.6,502.6034,-110.6934,-57.0662], [347.2482,-494.2305,293.6293,-84.8201,-2.8636], [364.5066,-420.9464,197.7163,-61.6707,29.2380], [172.8698,-205.0662,97.0404,-23.7418,3.2421], [2.1572,-2.0367,0.4234,0.3492,0.5612], [39.0661,-46.6079,21.7275,-4.8873,-2.8456], [13.1891,-10.1115,2.2276,-1.2081,-0.0720], [16.1460,-16.5183,6.9372,-0.5876,0.6249]],
#                   [[17.2035,-25.3140,16.7064,-5.8122,-1.1309], [2.1946,-2.3090,0.7162,0.2570,-0.4818], [-1240.9,1624.2,-872.0751,248.8905,-39.5076], [2912.8,-2879.2,1171.0,-271.5191,-23.8006], [685.0781,-969.6809,519.8358,-124.8295,4.6944], [-243.6033,258.0985,-101.7795,17.4144,-6.2708], [2.2857,-2.4649,0.8261,0.2417,-0.5325], [-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529]],
#                   [[17.2035,-25.3140,16.7064,-5.8122,-1.1309], [-2.1946,2.3090,-0.7162,-0.2570,0.4818], [-1240.9,1624.2,-872.0751,248.8905,-39.5076], [685.0781,-969.6809,519.8358,-124.8295,4.6944], [2912.8,-2879.2,1171.0,-271.5191,-23.8006], [-243.6033,258.0985,-101.7795,17.4144,-6.2708], [2.2857,-2.4649,0.8261,0.2417,-0.5325], [-104.5821,129.5426,-64.9653,17.0979,-2.5537],[-65.3237,58.0976,-17.3916,-0.6455,-0.1529],[-71.1397,67.6541,-23.1166,-1.4280,-0.8085]]])

# theta_coeff=np.array([-90.3137,94.5175,-35.6182,3.7520,0.6523])
# e1_coeff=np.array([7242.0, -7790.1, 3139.9, -571.7475, 41.9684])
# e2_coeff=np.array([5.2420, -5.6453, 2.2743, -0.4113, 0.029])
# e3_coeff=np.array([-233.8069, 251.4137, -101.2784, 18.4236, -1.3489])
# change_length=Change_length_fit(np.zeros((4, 10)),k_coeff,theta_coeff,e1_coeff,e2_coeff,e3_coeff)
leso_d_hat = np.zeros((4,1))
b = a = U_last = U = np.zeros((4,1))
U_out = np.zeros((6,1))
with mj.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        step_start = time.time()

        # 仿真主循环

        state = sensor.get_state()

        length = {
            "left_front_big": 0.17472,#0.21,
            "left_back_big": 0.17472,#0.21,
            "left_front_small": 0.208614,#0.2462962962962,
            "left_back_small": 0.208614,#0.2462962962962,
            "left_wheel":0.05,

            "right_front_big": 0.17472,#0.21,
            "right_back_big": 0.17472,#0.21,
            "right_front_small": 0.208614,#0.2462962962962,
            "right_back_small": 0.208614,#0.2462962962962,
            "right_wheel":0.05,
            
            "R":0.2,#0.165,
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

        # K=change_length.get_K((left.length+right.length)/2)
        # theta_target=left_target=right_target=0
        # e1=change_length.get_e1((left.length+right.length)/2)
        # e2=change_length.get_e2((left.length+right.length)/2)
        # e3_l=e3_r=change_length.get_e3((left.length+right.length)/2)
        # left_target=right_target=change_length.get_theta_target((left.length+right.length)/2)

        # F_c=F_convergence.copy()
        # F_c=K

        # print('K:')
        # print(K)
        # print('F_convergence:')
        # print(F_convergence)

        x_dot_target=cmd_v
        w_dot_target=cmd_omega_w

        x_target+=x_dot_target*dt
        w_target+=w_dot_target*dt

        # if((x_target-robot_state.x[0,0])>0.05 and x_dot_target!=0):
        #     x_target=robot_state.x[0,0]+0.05
        # if((x_target-robot_state.x[0,0])<-0.05 and x_dot_target!=0):
        #     x_target=robot_state.x[0,0]-0.05

        if((w_target-robot_state.x[1,0])>3.14):
            w_target-=2*3.14
        if((w_target-robot_state.x[1,0])<-3.14):
            w_target+=2*3.14

        
        leg_length+=cmd_hight

        if leg_length<0.15:
            leg_length=0.15
        if leg_length>0.32:
            leg_length=0.32

        max1=1500
        max2=1000
        max3=1000

        jumped_r=0
        jumped_l=0
        G_m=100

        # left_target=0.0
        # right_target=0.0
        
        left_length_pos.position_pid(leg_length,left.length)
        left_length_vel.position_pid(0,left.length_dot.Diff(left.length))
        right_length_pos.position_pid(leg_length,right.length)
        right_length_vel.position_pid(0,right.length_dot.Diff(right.length))
        roll_pid.position_pid(0,imu["euler"][0])


        if right.length<0.30 and jump and flag:
            jumped_r=5000
            max1=0
            max2=0
            
        elif left.length>=0.30 and jump:
            jumped_r=0
            flag=0
            max2=0
            max3=0
            G_m=0
            leg_length=0.12
        elif left.length<=0.12 and jump:
            jumped_r=0
            flag=0
            max3=0
            leg_length=0.12

        if left.length<0.30 and jump and flag:
            jumped_l=5000
            max1=0
            max2=0
            # leg_length=0.4
        elif left.length>=0.30 and jump:
            jumped_l=0
            flag=0
            G_m=0
            max2=0
            max3=0
            leg_length=0.12

        elif left.length<=0.12 and jump:
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

        P_l=F_l*cos(robot_state.x[3,0])+U[0]*sin(robot_state.x[3,0])/left.length
        P_r=F_r*cos(robot_state.x[4,0])+U[1]*sin(robot_state.x[4,0])/right.length

        z_wheel_l=z_body[2]-left.length_ddot.Diff(left.length_dot.diff_num)*cos(left.theta)+2*left.length_dot.diff_num*left.theta_dot.diff_num*sin(left.theta)+left.length*left.theta_ddot.Diff(left.theta_dot.diff_num)*sin(left.theta)+left.length*left.theta_dot.diff_num**2*cos(left.theta)
        z_wheel_r=z_body[2]-right.length_ddot.Diff(right.length_dot.diff_num)*cos(right.theta)+2*right.length_dot.diff_num*right.theta_dot.diff_num*sin(right.theta)+right.length*right.theta_ddot.Diff(right.theta_dot.diff_num)*sin(right.theta)+right.length*right.theta_dot.diff_num**2*cos(right.theta)

        Fn_l=P_l+m_l*g+z_wheel_l*m_l    
        Fn_r=P_r+m_r*g+z_wheel_r*m_r

        theta, theta_l, theta_r, B, K = robot_mpc.get_matrix(left.length,right.length)
        
        robot_state.update_state(left, right, length, motor, imu, road)
        robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target, 0, 0, 0)

        # leso_d_hat=robot_mpc.leso(robot_state.x-robot_state.x_target,U)
       
        
        x_target_ = np.asarray(robot_state.x_target).flatten()   # (10,)
        
        # 复制 N 步，得到参考轨迹 (N, 10)
        x_ref = np.tile(x_target_, (robot_mpc.N, 1))  
        

        # U_lqr=K@(robot_state.x-robot_state.x_target)

        # u_lqr = np.asarray(U_lqr).flatten()      
        u_lqr = np.zeros(4)                 # (4,)
        u_ref = np.tile(u_lqr, (robot_mpc.N, 1)) 



        if(Fn_l<20) :
            robot_mpc.K[0,0]=0
            robot_mpc.K[0,1]=0
            robot_mpc.K[0,2]=0
            robot_mpc.K[0,4]=0

            robot_mpc.K[0,5]=0
            robot_mpc.K[0,6]=0
            robot_mpc.K[0,7]=0
            robot_mpc.K[0,9]=0
            
            robot_mpc.K[2,0]=0
            robot_mpc.K[2,1]=0
            robot_mpc.K[2,2]=0
            robot_mpc.K[2,3]=0
            robot_mpc.K[2,4]=0

            robot_mpc.K[2,5]=0
            robot_mpc.K[2,6]=0
            robot_mpc.K[2,7]=0
            robot_mpc.K[2,8]=0
            robot_mpc.K[2,9]=0

            # robot_mpc.lambda_[0:robot_mpc.N,0,0]=0
            # robot_mpc.lambda_[0:robot_mpc.N,2,0]=0

            # leso_d_hat[0,0]=0
            # leso_d_hat[2,0]=0   
   

            for i in range(robot_mpc.N-1):
                robot_mpc.y[i,0]=0
                robot_mpc.y[i,2]=0
                robot_mpc.d[i,0]=0
                robot_mpc.d[i,2]=0
        
            


        if(Fn_r<20) : 
            robot_mpc.K[1,0]=0
            robot_mpc.K[1,1]=0
            robot_mpc.K[1,2]=0
            robot_mpc.K[1,3]=0

            robot_mpc.K[1,5]=0
            robot_mpc.K[1,6]=0
            robot_mpc.K[1,7]=0
            robot_mpc.K[1,8]=0

            robot_mpc.K[3,0]=0
            robot_mpc.K[3,1]=0
            robot_mpc.K[3,2]=0
            robot_mpc.K[3,3]=0
            robot_mpc.K[3,4]=0

            robot_mpc.K[3,5]=0
            robot_mpc.K[3,6]=0
            robot_mpc.K[3,7]=0
            robot_mpc.K[3,8]=0
            robot_mpc.K[3,9]=0

            # robot_mpc.lambda_[0:robot_mpc.N,1,0]=0
            # robot_mpc.lambda_[0:robot_mpc.N,3,0]=0

            # leso_d_hat[1,0]=0
            # leso_d_hat[3,0]=0
            
            

            for i in range(robot_mpc.N-1):
                robot_mpc.y[i,1]=0
                robot_mpc.y[i,3]=0
                robot_mpc.d[i,1]=0
                robot_mpc.d[i,3]=0           


        # if times :
            # robot_mpc.update_forward(robot_state.x, robot_state.x_target, leso_d_hat)
        
        times=0
        # if not times :
        #     U=U_n[1]
        #     times=1

        # U=U_last-F_convergence @ (np.block([[robot_state.update_state(left, right, length, motor,imu, road)],
        #                     [robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)],
        #                     [robot_state.update_control(U_last)]]))

        # U_d=Bd.T @ (I-Ad)@robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)
        # U=U_d-F_convergence @ (np.block([[robot_state.update_state(left, right, length, motor, imu, road)],
        #                     [robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target)]]))

        # U=-F_c @ (robot_state.update_state(left, right, length, motor,imu, road)-robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target, theta_target-e2, left_target-e3_l, right_target-e3_r))
        # U=-K @ (robot_state.update_state(left, right, length, motor,imu, road)-robot_state.update_target(x_target, w_target, x_dot_target, w_dot_target, theta_target, left_target, right_target))

        U_n=robot_mpc.ADMM(robot_state.x,x_ref,u_ref)
        U=U_n[0]

        left.vmc(F_l, U[0], motor, length)
        right.vmc(F_r, U[1], motor, length)
        # left.vmc(F_l+12, 0, motor, length)
        # right.vmc(F_r+12, 0, motor, length)

        U_out = np.block([[left.t_front],[right.t_front],[left.t_back],[right.t_back],[U[2]],[U[3]]])

        if abs(imu["euler"][0])>1.0 or abs(imu["euler"][1])>1.0 :
            U_out[0]=0
            U_out[1]=0
            U_out[2]=0
            U_out[3]=0
            U_out[4]=0
            U_out[5]=0
       
        # if (abs(robot_state.x[3,0])>1.5 or abs(robot_state.x[4,0])>1.5) :
        #     U_out[0,0]=0
        #     U_out[1,0]=0
        #     U_out[2,0]=0
        #     U_out[3,0]=0
        #     U_out[4,0]=0
        #     U_out[5,0]=0  


        

        send_list = [
            
            U_n[0][0],
            U_n[0][1],
            U_n[0][2],
            U_n[0][3],
            U_n[1][0],
            U_n[1][1],
            U_n[1][2],
            U_n[1][3],
            U_n[2][0],
            U_n[2][1],
            U_n[2][2],
            U_n[2][3],
            
        ]
        


        # 3. 数据打包为二进制 (f 代表 float, < 代表小端)
        # 这里的 'ffff' 代表有 4 个 float 数据
        binary_data = struct.pack('<' + 'f' * len(send_list), *send_list)
        sock.sendto(binary_data + vofa_tail, (UDP_IP, UDP_PORT))

        if U_out[0]>50:
            U_out[0]=50
        if U_out[0]<-50:
            U_out[0]=-50
        if U_out[1]>50:
            U_out[1]=50
        if U_out[1]<-50:  
            U_out[1]=-50
        if U_out[2]>50:
            U_out[2]=50
        if U_out[2]<-50:
            U_out[2]=-50
        if U_out[3]>50:
            U_out[3]=50
        if U_out[3]<-50:
            U_out[3]=-50
        if U_out[4]>7:
            U_out[4]=7
        if U_out[4]<-7:
            U_out[4]=-7
        if U_out[5]>7:
            U_out[5]=7
        if U_out[5]<-7:
            U_out[5]=-7
   

        for i in range(6):
            data.ctrl[i] =  U_out[i,0]   
        # 3. 物理引擎向前推进一步

        mj.mj_step(model, data)
        
        # 3. 刷新渲染器 (这里通常对应显示器的刷新频率)
        viewer.sync()
        U_last=U
        # 适当微休眠，防止 CPU 占用率 100%
        time.sleep(dt*1)
