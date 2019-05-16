# Time-varying LQR around the nominal trajectory and u trajectory from trajectory optimization
import argparse
import os
import numpy as np
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.framework import VectorSystem, DiagramBuilder
from pydrake.systems.controllers import LinearQuadraticRegulator

from ..simulate.simulate import simulate

from ..models.quadrotor import Quadrotor
from ..autodiff.quadrotor_linearizer import QuadrotorDynamicsLinearizer
from utilities import load_traj


class TVLQR(VectorSystem):
    """ controller that uses time-varying LQR around nominal trajectory
    """
    def __init__(self, filename):
        """
        :param filename: the file to load trajectories from
        """
        VectorSystem.__init__(self,
                              12,  # number of inputs, the state of the quadrotor
                              4)  # number of outputs, the inputs to four motors

        self.traj, self.u_traj, self.time_array = load_traj(filename)
        assert(self.time_array.shape[0] == self.u_traj.shape[0]+1)
        self.final_time = self.time_array[-1][0]
        self.initial_state = self.traj[0]
        self.dynamics_linearizer = QuadrotorDynamicsLinearizer()

    def DoCalcVectorOutput(self, context, u, x, y):
        """
        :param context: the context of the quadrotor
        :param u: input from the quadrotor, which is its state
        :param x: the system's internal state
        :param y: the output to the quadrotor
        """
        # Get the control input corresponding to simulation time
        time = context.get_time()
        index = np.where(self.time_array <= time)[0][-1]

        # test if reached terminal time
        if index < self.u_traj.shape[0]:
            u0 = self.u_traj[index]
        else:
            u0 = self.u_traj[index-1]

        # Set as output, i.e control input to the quadrotor
        current_state = u
        target_state = self.traj[index]

        # linearize around nominal point
        A, B = self.dynamics_linearizer.get_AB(target_state, u0)

        # Get Q and R cost matrices
        Q = np.identity(12)
        Q[:6, :6] = 10 * np.identity(6)
        R = np.identity(4)

        # Obtain controller through LQR
        K, S = LinearQuadraticRegulator(A, B, Q, R)

        # x_bar = x - x0
        x_bar = np.array(current_state) - np.array(target_state)
        x_bar = x_bar.reshape((-1, 1))

        # u_bar = u - u0
        u_bar = -np.matmul(K, x_bar).reshape((-1,))
        out = u_bar + u0

        y[:] = out


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename",
                        type=str,
                        help="filename to load trajectory from",
                        required=True)
    parser.add_argument("-N", "--trials",
                        type=int,
                        help="Number of trials to run.",
                        default=5)
    parser.add_argument("-M", "--model",
                        type=str,
                        help="Select the model to run",
                        default="quadrotor")
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    # Load quadrotor u trajectory, and set duration of simulation & initial state
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/traj/{}.npz".format(args.filename)
    controller = TVLQR(filename)
    args.duration = controller.final_time + 5.0

    def initial_state_gen():
        return controller.initial_state

    quadrotor = Quadrotor()
    # Display in meshcat
    simulate(args, plant=quadrotor, controller=controller, initial_state_gen=initial_state_gen)
