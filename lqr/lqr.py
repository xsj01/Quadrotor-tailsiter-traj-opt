# Implement a LQR controller around stabilizable point
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (MeshcatVisualizer, DiagramBuilder, SceneGraph, MultibodyPositionToGeometryPose,
                         ConstantVectorSource, Simulator, VectorSystem, SignalLogger)
from pydrake.systems.controllers import LinearQuadraticRegulator

from ..models.quadrotor import Quadrotor
from ..models.tailsitter import Tailsitter
from ..autodiff.quadrotor_linearizer import QuadrotorDynamicsLinearizer
from ..autodiff.tailsitter_linearizer import TailsitterDynamicsLinearizer
from ..simulate.simulate import simulate


class LQRController(VectorSystem):
    """ Quadrotor controller that stabilizes around a steady state
    """
    def __init__(self, plant, dynamics_linearizer, target_pos=(0, 0, 1)):
        """
        :param height: the height to control to
        """
        VectorSystem.__init__(self,
                              12,  # number of inputs, the state of the quadrotor
                              4)  # number of outputs, the inputs to four motors
        self.target_pos = target_pos
        self.plant = plant
        self.dynamics_linearizer = dynamics_linearizer

    def DoCalcVectorOutput(self, context, u, x, y):
        """
        :param context: the context of the quadrotor
        :param u: input from the quadrotor, which is its state
        :param x: the system's internal state
        :param y: the output to the quadrotor
        """
        current_state = u
        target_state = np.zeros_like(current_state)
        target_state[:3] = self.target_pos

        # nominal input for stable point
        u0 = np.ones(shape=(4,))*self.plant.m_ * self.plant.g_ / 4

        # linearize around nominal point
        A, B = self.dynamics_linearizer.get_AB(target_state, u0)

        # Get Q and R cost matrices
        Q = np.identity(12)
        Q[:6, :6] = 10*np.identity(6)
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
    parser.add_argument("-N", "--trials",
                        type=int,
                        help="Number of trials to run.",
                        default=10)
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run each sim.",
                        default=5.0)
    parser.add_argument("-M", "--model",
                        type=str,
                        help="Select the model to run",
                        default="quadrotor")
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    if args.model == "quadrotor":
        plant = Quadrotor()
        linearizer = QuadrotorDynamicsLinearizer()
    else:
        plant = Tailsitter()
        linearizer = TailsitterDynamicsLinearizer()
    controller = LQRController(plant=plant, dynamics_linearizer=linearizer, target_pos=[0, 0, 1])

    def initial_state_gen():
        return np.zeros((12, ))
        return np.random.randn(12,)

    # Display in meshcat
    simulate(args, plant=plant, controller=controller, initial_state_gen=initial_state_gen)
