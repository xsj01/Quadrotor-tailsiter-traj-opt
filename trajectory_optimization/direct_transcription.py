# Implement direct transcription procedure
import os.path
import argparse
import numpy as np
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve

from dynamics import QuadrotorDynamics
from tailsitter_dynamics import TailsitterDynamics
from utilities import save_traj


class DirectTranscription:
    def __init__(self, dynamics_model):
        self.dynamics_model = dynamics_model

    def two_norm(self, x):
        '''
        Euclidean norm but with a small slack variable to make it nonzero.
        This helps the nonlinear solver not end up in a position where
        in the dynamics it is dividing by zero.

        :param x: numpy array of any length (we only need it for length 2)
        :return: numpy.float64
        '''
        slack = .001
        return np.sqrt((x**2).sum() + slack)

    def compute_trajectory(self, initial_state=np.zeros((12,)), target_state=(0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0), minimum_time=3, maximum_time=10):
        """ Use trajectory optimization to compute control to target state
        :param initial_state: initial state
        :param target_state: final state to be in, consisting of (x, y, z, roll, pitch, yaw) and their derivatives
        :param minimum_time: minimum time of the trajectory
        :param maximum_time: max time of the trajectory
        :return: trajectory, input_trajectory, time_array
        """
        # Direct Transcription
        mp = MathematicalProgram()

        # Set up knot points
        num_time_steps = 100
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
            mp.AddLinearConstraint(states_over_time[0][j] >= initial_state[j])
            mp.AddLinearConstraint(states_over_time[0][j] <= initial_state[j])

        # State transition constraints
        for i in range(0, num_time_steps):
            current_state = states_over_time[i]
            next_state = states_over_time[i+1]
            u = u_over_time[i]
            for j in range(12):
                mp.AddConstraint(next_state[j] - current_state[j] - self.dynamics_model.dynamics(current_state, u)[j] * dt[0] >= 0)
                mp.AddConstraint(next_state[j] - current_state[j] - self.dynamics_model.dynamics(current_state, u)[j] * dt[0] <= 0)

        final_state = states_over_time[-1]

        # Control effort cost
        mp.AddQuadraticCost(((u_over_time[:, 0]**2 + u_over_time[:, 1]**2 + u_over_time[:, 2]**2 + u_over_time[:, 3]**2).sum()))

        # Reach final state
        for j in range(12):
            mp.AddConstraint(final_state[j] <= target_state[j])
            mp.AddConstraint(final_state[j] >= target_state[j])

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


# Main function for calling direct transcription and saving trajectories
if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename",
                        type=str,
                        help="filename to save at",
                        required=True)
    parser.add_argument("-M", "--model",
                        type=str,
                        help="Select the model to run",
                        default="quadrotor")
    args = parser.parse_args()

    # Get filename and check for existence

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/traj/{}_{}.npz".format(args.filename, args.model)
    if os.path.exists(filename):
        raise Exception("{} already exists".format(filename))

    # Set up direct transcription problem
    if args.model == "quadrotor":
        dynamics_model = QuadrotorDynamics()
    else:
        dynamics_model = TailsitterDynamics()
    dir_trans = DirectTranscription(dynamics_model=dynamics_model)

    # Compute trajectory from initial state to final state
    initial_state = np.zeros((12,))
    initial_state[1] = 0
    target_state = np.zeros((12,))
    target_state[2] = 1
    traj, u_traj, time_array = dir_trans.compute_trajectory(initial_state=initial_state)

    # Save traj
    print("save to file: {}".format(filename))
    save_traj(filename, traj, u_traj, time_array)
