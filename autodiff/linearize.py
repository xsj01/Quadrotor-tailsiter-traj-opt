# -*- coding: utf-8 -*-
# Contains quadrotor dynamics implementation
import numpy as np
from Quadrotor.math.opt_math import RollPitchYaw, RotationMatrix

from pydrake.forwarddiff import jacobian, gradient, derivative


def default_moment_of_inertia():
    return np.array([[0.0023, 0, 0],
                     [0, 0.0023, 0],
                     [0, 0, 0.0040]])


class QuadrotorDynamicsLinearizer:
    """ Linearizer for obtaining the linearized quadrotor system around a nominal point
    """
    def __init__(self, m_arg=0.5, L_arg=0.175, I_arg=default_moment_of_inertia(), kF_arg=1.0, kM_arg=0.0245):
        # Set model constants
        self.g_ = 9.81
        self.m_ = m_arg
        self.L_ = L_arg
        self.I_ = I_arg
        self.kF_ = kF_arg
        self.kM_ = kM_arg

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

        uF_Bz = self.kF_ * u

        # Compute net force, expressed in body frame
        Faero_B = np.array([0, 0, np.sum(uF_Bz)]).reshape((-1, 1))

        # Compute x and y moment caused by motor forces, expressed in body frame
        Mx = self.L_ * (uF_Bz[1] - uF_Bz[3])
        My = self.L_ * (uF_Bz[2] - uF_Bz[0])

        # Compute moment in z, caused by air reaction force on rotating blades, expressed in body frame
        uTau_Bz = self.kM_ * u
        Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        # Net moment on body about its center of mass, expressed in body frame
        Tau_B = np.array([Mx, My, Mz]).reshape((-1, 1))

        # Compute gravity force, expressed in Newtonian frame (i.e. world frame)
        Fgravity_N = np.array([0, 0, -self.m_ * self.g_]).reshape((-1, 1))

        # Extract roll, pitch, yaw and their derivatives
        rpy = RollPitchYaw(state[3:6])
        rpyDt = state[9:12].reshape((-1, 1))

        # Convert roll-pitch-yaw to rotation matrix from inertial frame to body frame
        R_NB = RotationMatrix(rpy).matrix()

        # Calculate net force and acceleration, expressed in inertial frame
        Fnet_N = Fgravity_N + R_NB.dot(Faero_B)  # np.matmul(R_NB, Faero_B)
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
    linearizer= QuadrotorDynamicsLinearizer()
    state = np.ones(12, dtype=np.float64)
    u = np.ones(4,dtype=np.float64)
    print linearizer.get_AB(state,u)