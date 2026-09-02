import numpy as np

class Kalman:
    def __init__(self):
        self.x=np.array([[0.0],[0.0],[0.0]])

        self.z=np.array([[0.0],[0.0]])

        # self.u=np.array([[0.0]])

        self.P=np.array([[1.0,0.0,0.0],
                         [0.0,1.0,0.0],
                         [0.0,0.0,1.0]])

        self.H=np.array([[0.0,1.0,0.0],
                         [0.0,0.0,1.0]])

        self.I=np.eye(3)

        self.K=np.zeros((3,3))

        self.Q=np.array([[0.00001,0.000002,0.000000002],
                         [0.000002,0.00001,0.000002],
                         [0.000000002,0.000002,0.00001]])
        self.R=np.array([[1000.0,20.0],
                         [20.0,1000.0]])
        self.A=np.array([[1.0,0.002,0.000002],
                         [0.0,1.0,0.002],
                         [0.0,0.0,1.0]])
        # self.B=np.array([[0.000002],[0.002],[0.0]])

        

    def filter(self,z_body,motor,length,dt):
        self.z[0,0]=(length["left_wheel"]*motor["left_wheel_vel"] + length["right_wheel"]*motor["right_wheel_vel"])*0.5
        self.z[1,0]=z_body[0]

        # self.u[0,0] = z_body[0]

        self.x=self.A @ self.x #+ self.B @ self.u
        self.P=self.A @ self.P @ self.A.T + self.Q

        self.K=(self.P @ self.H.T)@ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x=self.x + self.K @ (self.z - self.H @ self.x)
        self.P=(self.I-self.K@self.H) @ self.P

