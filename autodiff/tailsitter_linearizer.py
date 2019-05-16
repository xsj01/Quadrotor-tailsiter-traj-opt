# -*- coding: utf-8 -*-
# Contains quadrotor dynamics implementation
import numpy as np
from Quadrotor.math.opt_math import RollPitchYaw, RotationMatrix

from Quadrotor.models.dynamics_drake import *

from pydrake.forwarddiff import jacobian, gradient, derivative


def default_moment_of_inertia():
    return np.array([[0.0023, 0, 0],
                     [0, 0.0023, 0],
                     [0, 0, 0.0040]])

def rot_mat(theta,axis,autodiff=False):
    #return R_0_to_1
    #R transform cooridinates from 0 frame to 1 frame
    #theta counterclockwise rotate 0 to 1
    if autodiff:
        cos=dd.cos
        sin=dd.sin
    else:
        cos=np.cos
        sin=np.sin
    if axis==0 or axis=='x':
        return np.array([[1.,0,0],
                        [0,cos(theta),-sin(theta)],
                        [0,sin(theta),cos(theta)]
                        ])
    if axis==1 or axis=='y':
        return np.array([[cos(theta),0,sin(theta)],
                        [0,1,0],
                        [-sin(theta),0,cos(theta)]])
    if axis==2 or axis=='z':
        return np.array([[cos(theta),-sin(theta),0],
                        [sin(theta),cos(theta),0],
                        [0,0,1.]])


class TailsitterDynamicsLinearizer:
    """ Linearizer for obtaining the linearized tailsitter system around a nominal point
    """
    def __init__(self, m_arg=0.5, L_arg=0.175, I_arg=default_moment_of_inertia(), kF_arg=1.0, kM_arg=0.0245):
        # Set model constants
        self.g_ = 9.81
        self.m_ = m_arg
        self.L_ = L_arg
        self.I_ = I_arg
        self.kF_ = kF_arg
        self.kM_ = kM_arg

        self.delta=0.1
        self.rho=1.29
        self.S = 0.1955
        self.chi=np.pi/4
        self.d=0.4
        self.kappa=1.
        self.c = 0.24

    def cal_F_M_1(self,state,u):

        roll=state[3]
        pitch=state[4]
        yaw=state[5]

        #calculate V in frame 1
        R_roll=np.array([[1,0,0,],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
        R_pitch=np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
        R_yaw=np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])

        R_1_to_0=(R_yaw.dot(R_pitch).dot(R_roll))
        #R_1_to_0=R_roll.dot(R_pitch).dot(R_yaw)
        R_0_to_1=R_1_to_0.T

        v_0=state[6:9]

        v_1=R_0_to_1.dot(v_0)

        V=np.linalg.norm(v_0)


        #calculate V in frame 2 and alpha Beta

        R_1_to_2=np.array([[0,0,1],[cos(self.chi),-sin(self.chi),0],[sin(self.chi),cos(self.chi),0]])
        R_2_to_1=R_1_to_2.T

        v_2=R_1_to_2.dot(v_1)

        

        #calculate Force Momentum in Frame 2

        FM_aero=np.array([[1,1,1,1],[-1,1,-1,1],[1,1,-1,-1.],[-1,1,1,-1.]]).dot(u)
        Fx_2=FM_aero[0]*cos(self.delta)
        Mx_2=FM_aero[1]*(sin(self.delta)*self.d+self.kappa*cos(self.delta))
        My_2=FM_aero[2]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*sin(self.chi)
        Mz_2=FM_aero[3]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*cos(self.chi)
        M_propeller_2=np.array([Mx_2,My_2,Mz_2])
        F_propeller_2=np.array([Fx_2,0,0.])


        if V==0:
            alpha=0.
            beta=0.
            F_net_2=F_propeller_2
            M_net_2=M_propeller_2
        else:
            alpha=atan2(v_2[2],v_2[0])
            #vxy=np.linalg.norm(v_2[0::2])
            beta=asin(v_2[1]/V)
            #beta=atan2(v_2[1],vxy)

            L_aero_2=0.5*self.rho*V**2*self.S*get_CL(alpha,beta)
            D_aero_2=0.5*self.rho*V**2*self.S*get_CD(alpha,beta)
            Y_aero_2=0.5*self.rho*V**2*self.S*get_CY(alpha,beta)

            ll_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cll(alpha,beta)
            m_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cm(alpha,beta)
            n_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cn(alpha,beta)

            DYL=np.array([D_aero_2,Y_aero_2,L_aero_2]).reshape(-1)

            F_aero_2=np.array([[-cos(alpha),0,sin(alpha)],[0,1,0],[-sin(alpha),0,-cos(alpha)]]).dot(DYL)
            M_aero_2=np.array([ll_aero_2,m_aero_2,n_aero_2]).reshape(-1)

            F_net_2=F_propeller_2+F_aero_2
            M_net_2=M_aero_2+M_propeller_2


        #Calculate Force Momentum in frame 0

        F_net_1=R_2_to_1.dot(F_net_2)

        M_net_1=R_2_to_1.dot(M_net_2)

        #print '1 to 0', R_1_to_0

        #F_gravity_0=np.array([0.,0.,-self.m_*self.g_])

        #F_net_0=F_net_0+F_gravity_0

        return F_net_1.reshape((-1,1)),M_net_1.reshape((-1,1))

    def dynamics(self, state, u):
        """ Calculate the dynamics, i.e: \dot(state) = f(state, u)
        :param state: the state of the quadrotor
        :param u: the inputs to four motors
        :return: the the time derivative of the quadrotor state
        """
        # Calculate force exerted by each motor, expressed in Body frame
        # if autograd:
        #     np=ag.np
        #     RollPitchYaw=ag.RollPitchYaw
        #     RotationMatrix=ag.RotationMatrix



        # # Compute gravity force, expressed in Newtonian frame (i.e. world frame)
        Fgravity_N = np.array([0, 0, -self.m_*self.g_]).reshape((-1, 1))

        # # Get current state
        #state = context.get_continuous_state_vector().CopyToVector()

        # Extract roll, pitch, yaw and their derivatives
        rpy = RollPitchYaw(state[3:6])
        rpyDt = state[9:12].reshape((-1, 1))

        #calculate Force Moment in body frame
        Faero_B,Tau_B=self.cal_F_M_1(state.reshape(-1),u.reshape(-1))

        

        # Convert roll-pitch-yaw to rotation matrix from inertial frame to body frame
        R_NB = RotationMatrix(rpy).matrix()

        #print 'R_NB',R_NB
        #raw_input()
        
        
        

        # # Calculate net force and acceleration, expressed in inertial frame
        Fnet_N = Fgravity_N + R_NB.dot(Faero_B)

        #print 'R_NB',R_NB
        #raw_input()
        xyzDDt = Fnet_N / self.m_

        # Calculate body's angular velocity in inertial frame, expressed in body frame
        w_BN_B = rpy.CalcAngularVelocityInChildFromRpyDt(rpyDt)

        # Compute body's angular acceleration, α, in inertia frame, due to the net moment 𝛕 on body,
        # rearrange Euler rigid body equation 𝛕 = I α + ω × (I ω)  and solve for α.
        wIw = np.cross(w_BN_B, self.I_.dot(w_BN_B), axis=0).reshape((-1, 1))
        # alpha_NB_B = np.linalg.solve(self.I_, Tau_B - wIw)
        alpha_NB_B = np.linalg.inv(self.I_).dot(Tau_B-wIw)
        alpha_NB_N = R_NB.dot(alpha_NB_B)

        # Calculate the 2nd time-derivative of rpy
        rpyDDt = rpy.CalcRpyDDtFromRpyDtAndAngularAccelInParent(rpyDt, alpha_NB_N)

        # Set derivative of pos by current velocity,
        # and derivative of vel by input, which is acceleration
        deriv = np.zeros_like(state).tolist()

        deriv[:6] = state[6:]+0*u[0]+0*u[1]+0*u[2]+0*u[3]

        deriv[8]=xyzDDt[2,0]
        deriv[7]=xyzDDt[1,0]
        deriv[6]=xyzDDt[0,0]
        for i in range(9, 12):
            v = rpyDDt.ravel()[i-9]
            if type(v) is np.ndarray:   
                v=v[0]
            deriv[i]=v   
        return np.array(deriv)

    def get_AB(self, state, u):
        """
        :param state: numpy array of floats, shape (12,), representing x0
        :param u: numpy array of floats, shape (4,), representing u0
        :return: A and B of linearized system around (x0, u0) stable point
        """
        def xDi_uj(uj,i,j):
            u_input=u.astype(dtype=object)
            u_input[j]=uj
            return self.dynamics(state, u_input).reshape(-1)[i]

        pf_px=jacobian(lambda state_:self.dynamics(state_, u), state)
        pf_pu=jacobian(lambda u_:self.dynamics(state, u_), u)

        return pf_px,pf_pu

if __name__ == '__main__':
    linearizer= TailsitterDynamicsLinearizer()
    state = np.zeros(12, dtype=np.float64)
    u = np.ones(4,dtype=np.float64)
    print linearizer.get_AB(state,u)