# -*- coding: utf-8 -*-
# Trajectory optimization for quadrotor
import numpy as np
# from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.all import MathematicalProgram, Solve, cos, sin


class RollPitchYaw:
    def __init__(self, rpy):
        self.roll_pitch_yaw_ = rpy

    def roll_angle(self):
        return self.roll_pitch_yaw_[0]

    def pitch_angle(self):
        return self.roll_pitch_yaw_[1]

    def yaw_angle(self):
        return self.roll_pitch_yaw_[2]


class RotationMatrix:
    def __init__(self, rpy):
        r = rpy.roll_angle()
        p = rpy.pitch_angle()
        y = rpy.yaw_angle()

        c0, c1, c2 = cos(r), cos(p), cos(y)
        s0, s1, s2 = sin(r), sin(p), sin(y)

        c2_s1 = c2*s1
        s2_s1 = s2*s1

        Rxx = c2*c1
        Rxy = c2_s1 * s0 - s2 * c0
        Rxz = c2_s1 * c0 + s2 * s0
        Ryx = s2 * c1
        Ryy = s2_s1 * s0 + c2 * c0
        Ryz = s2_s1 * c0 - c2 * s0
        Rzx = -s1
        Rzy = c1 * s0
        Rzz = c1 * c0
        self.R_AB_ = np.array([[Rxx, Rxy, Rxz],
                               [Ryx, Ryy, Ryz],
                               [Rzx, Rzy, Rzz]])

    def matrix(self):
        return self.R_AB_


def default_moment_of_inertia():
    return np.array([[0.0023, 0, 0],
                     [0, 0.0023, 0],
                     [0, 0, 0.0040]])


class QuadrotorTrajectoryOptimization:
    def __init__(self, m_arg=0.5, L_arg=0.175, I_arg=default_moment_of_inertia(), kF_arg=1.0, kM_arg=0.0245):
        # Set model constants
        self.g_ = 9.81
        self.m_ = m_arg
        self.L_ = L_arg
        self.I_ = I_arg
        self.kF_ = kF_arg
        self.kM_ = kM_arg

    def quadrotor_dynamics(self, state, u):
        """ Calculate the dynamics, i.e: \dot(state) = f(state, u)
        :param state: the state of the quadrotor
        :param u: the inputs to four motors
        :return: the the time derivative of the quadrotor state
        """
        # Calculate force exerted by each motor, expressed in Body frame
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
        # print(R_NB)
        # print(Faero_B)
        # print(R_NB.dot(Faero_B))
        Fnet_N = Fgravity_N + R_NB.dot(Faero_B)  # np.matmul(R_NB, Faero_B)
        xyzDDt = Fnet_N / self.m_

        # Calculate body's angular velocity in inertial frame, expressed in body frame
        w_BN_B = rpy.CalcAngularVelocityInChildFromRpyDt(rpyDt)

        # Compute body's angular acceleration, α, in inertia frame, due to the net moment 𝛕 on body,
        # rearrange Euler rigid body equation 𝛕 = I α + ω × (I ω)  and solve for α.
        wIw = np.cross(w_BN_B, np.matmul(self.I_, w_BN_B)).reshape((-1, 1))
        alpha_NB_B = np.linalg.solve(self.I_, Tau_B - wIw)
        alpha_NB_N = np.matmul(R_NB, alpha_NB_B)

        # Calculate the 2nd time-derivative of rpy
        rpyDDt = rpy.CalcRpyDDtFromRpyDtAndAngularAccelInParent(rpyDt, alpha_NB_N)

        # Set derivative of pos by current velocity,
        # and derivative of vel by input, which is acceleration
        deriv = np.zeros((12,))
        deriv[:6] = state[6:]
        deriv[6:9] = xyzDDt.ravel()
        deriv[9:] = rpyDDt.ravel()
        return deriv

    def two_norm(self, x):
        '''
        Euclidean norm but with a small slack variable to make it nonzero.
        This helps the nonlinear solver not end up in a position where
        in the dynamics it is dividing by zero.

        :param x: numpy array of any length (we only need it for length 2)
        :return: numpy.float64
        '''
        slack = .001
        return np.sqrt(((x)**2).sum() + slack)

    def compute_trajectory_to_pose(self, state_initial=np.zeros((12,)), target_pose=(0, 0, 1, 0, 0, 0), minimum_time=5, maximum_time=10):
        """ Use trajectory optimization to compute control to target pose
        :param state_initial: initial state
        :param target_pose: target_pose, consisting of (x, y, z, roll, pitch, yaw)
        :param minimum_time: minimum time of the trajectory
        :param maximum_time: max time of the trajectory
        :return: trajectory, input_trajectory, time_array
        """
        # Direct Transcription
        mp = MathematicalProgram()

        # Set up knot points
        num_time_steps = 50
        dt = mp.NewContinuousVariables(1, "dt")
        time_array = 0.0
        for i in range(1, num_time_steps+1):
            time_array = np.vstack((time_array, i*dt))

        # Set up control inputs
        k = 0
        u = mp.NewContinuousVariables(4, "u_%d" % k)
        u_over_time = u
        for k in range(1, num_time_steps):
            u = mp.NewContinuousVariables(4, "u_%d" % k)
            u_over_time = np.vstack((u_over_time, u))

        # Set up state variables
        k = 0
        states_over_time = mp.NewContinuousVariables(12, "x_%d" % k)
        for k in range(1, num_time_steps+1):
            state = mp.NewContinuousVariables(12, "x_%d" % k)
            states_over_time = np.vstack((states_over_time, state))

        # Initial state constraints
        for j in range(12):
            mp.AddLinearConstraint(states_over_time[0][j] >= state_initial[j])
            mp.AddLinearConstraint(states_over_time[0][j] <= state_initial[j])

        # State transition constraints
        for i in range(0, num_time_steps):
            current_state = states_over_time[i]
            next_state = states_over_time[i+1]
            u = u_over_time[i]
            for j in range(12):
                mp.AddConstraint(next_state[j] - current_state[j] - self.quadrotor_dynamics(current_state, u)[j] * dt[0] >= 0)
                mp.AddConstraint(next_state[j] - current_state[j] - self.quadrotor_dynamics(current_state, u)[j] * dt[0] <= 0)

        final_state = states_over_time[-1]
        final_pose = final_state[:6]
        final_vel = final_state[6:]

        # Control effort cost
        mp.AddQuadraticCost(((u_over_time[:, 0]**2 + u_over_time[:, 1]**2 + u_over_time[:, 2]**2 + u_over_time[:, 3]**2).sum()))

        # Reach final pose
        for j in range(12):
            mp.AddConstraint(final_pose[j] <= target_pose[j])
            mp.AddConstraint(final_pose[j] >= target_pose[j])

        # Time constraints
        total_time = num_time_steps * dt
        mp.AddLinearConstraint(total_time[0] <= maximum_time)
        mp.AddLinearConstraint(total_time[0] >= minimum_time)

        result = Solve(mp)
        #######################################################################

        # Extract outputs
        trajectory = result.GetSolution(states_over_time)
        input_trajectory = result.GetSolution(u_over_time)
        dt = result.GetSolution(dt)
        time_array = 0.0
        for i in range(1, num_time_steps+1):
            time_array = np.vstack((time_array, i*dt))

        return trajectory, input_trajectory, time_array


if __name__ == "__main__":
    quadrotor = QuadrotorTrajectoryOptimization()
    traj, u_traj, time_array = quadrotor.compute_trajectory_to_pose()
    print(traj)