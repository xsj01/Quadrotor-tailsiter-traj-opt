# Feed u trajectory generated in trajectory optimization directly to the quadrotor model
import argparse
import os
import numpy as np
from pydrake.systems.framework import VectorSystem, DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer

from ..simulate.simulate import simulate
from ..models.quadrotor import Quadrotor
from utilities import load_traj


class FeedUTrajController(VectorSystem):
    """ Quadrotor controller that feeds control inputs from u traj of trajectory optimization
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
            u = self.u_traj[index]
        else:
            u = np.zeros_like(self.u_traj[0])

        # Set as output, i.e control input to the quadrotor
        y[:] = u


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
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    # Load quadrotor u trajectory, and set duration of simulation & initial state
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/traj/{}.npz".format(args.filename)
    controller = FeedUTrajController(filename)
    args.duration = controller.final_time

    def initial_state_gen():
        return controller.initial_state

    quadrotor = Quadrotor()
    # Display in meshcat
    simulate(args, plant=quadrotor, controller=controller, initial_state_gen=initial_state_gen)
