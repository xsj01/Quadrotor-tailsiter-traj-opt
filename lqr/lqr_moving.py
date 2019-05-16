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

class StableStateController(VectorSystem):
    def __init__(self, plant,dynamics_linearizer,target_state,ctrl_indx=range(12)):
        """
        :param height: the height to control to
        """
        VectorSystem.__init__(self,
                              12,  # number of inputs, the state of the quadrotor
                              4)
        self.ctrl_indx=ctrl_indx
        self.target_state=target_state
        self.plant = plant
        self.dynamics_linearizer = dynamics_linearizer

        v=target_state[6:9]
        _,_,target_u,rpy=plant.cal_alpha_beta_u_by_V(v)
        target_state[3:6]=rpy[:]
        self.target_u=target_u

    def get_output(self,current_state,Q=None,R=None):
        target_state=self.target_state
        # nominal input for stable point
        #u0 = np.ones(shape=(4,))*self.plant.m_ * self.plant.g_ / 4
        u0=self.target_u
        ctrl_indx=self.ctrl_indx
        # linearize around nominal point
        A, B = self.dynamics_linearizer.get_AB(target_state,u0)

        if not Q:
            Q = np.identity(12)
            Q[:6, :6] = 10*np.identity(6)

        Q=Q[ctrl_indx,:]
        Q=Q[:,ctrl_indx]

        A=A[ctrl_indx,:]
        A=A[:,ctrl_indx]

        B=B[ctrl_indx,:]

        if not R:
            R = np.identity(4)

        K, S = LinearQuadraticRegulator(A, B, Q, R)

        x_bar = np.array(current_state[ctrl_indx]) - np.array(target_state[ctrl_indx])
        x_bar = x_bar.reshape((-1, 1))

        # u_bar = u - u0
        u_bar = -np.matmul(K, x_bar).reshape((-1,))
        out = u_bar + u0

        return out

    pass

class LQRController(VectorSystem):
    """ Quadrotor controller that stabilizes around a steady state
    """
    def __init__(self, plant, dynamics_linearizer):
        """
        :param height: the height to control to
        """
        VectorSystem.__init__(self,
                              12,  # number of inputs, the state of the quadrotor
                              4)  # number of outputs, the inputs to four motors

        v=np.array([2.,0.,0.])
        target_state=np.zeros(12)
        target_state[2]=1
        target_state[6:9]=v[:]
        ctrl_indx=[1,2]+range(3,12)

        self.ctrl1=StableStateController(plant, dynamics_linearizer,target_state,ctrl_indx)

        v=np.array([-2.,0.,0.])
        target_state=np.zeros(12)
        target_state[2]=1
        target_state[6:9]=v[:]
        ctrl_indx=[1,2]+range(3,12)

        self.ctrl2=StableStateController(plant, dynamics_linearizer,target_state,ctrl_indx)

        v=np.array([0.,0.,0.])
        target_state=np.zeros(12)
        target_state[2]=1.
        target_state[6:9]=v[:]
        ctrl_indx=[1,2]+range(3,12)

        self.ctrl3=StableStateController(plant, dynamics_linearizer,target_state,ctrl_indx)

        #self.target_v = target_v
        self.plant = plant
        self.dynamics_linearizer = dynamics_linearizer
        # _,_,target_u,rpy=plant.cal_alpha_beta_u_by_V(v)
        # #target_v=v
        # target_state=np.zeros(12)
        # target_state[6:9]=v[:]
        # target_state[3:6]=rpy[:]

        self.count=0

        #self.target_state=target_state
        #self.target_u=target_u
        #self.dynamics_linearizer = dynamics_linearizer

    def DoCalcVectorOutput(self, context, u, x, y):
        """
        :param context: the context of the quadrotor
        :param u: input from the quadrotor, which is its state
        :param x: the system's internal state
        :param y: the output to the quadrotor
        """
        current_state = u

        self.count+=1

        if self.count<1000:
            out=self.ctrl1.get_output(current_state)
        else:
            out=self.ctrl3.get_output(current_state)
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
    controller = LQRController(plant=plant, dynamics_linearizer=linearizer)#, target_v=v)

    def initial_state_gen():
        return controller.ctrl1.target_state
        # return np.random.randn(12,)

    # Display in meshcat
    simulate(args, plant=plant, controller=controller, initial_state_gen=initial_state_gen)
