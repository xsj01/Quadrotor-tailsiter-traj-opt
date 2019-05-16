# -*- coding: utf-8 -*-
# Implement quadrotor model in python

import argparse
import os
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import SceneGraph
from pydrake.math import RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, VectorSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import AffineSystem
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake.all import LeafSystem, BasicVector, ConstantVectorSource, RigidTransform, FramePoseVector, Linearize, RigidBodyPlant


def default_moment_of_inertia():
    return np.array([[0.0023, 0, 0],
                     [0, 0.0023, 0],
                     [0, 0, 0.0040]])


class Quadrotor(LeafSystem):
    def __init__(self, m_arg=0.5, L_arg=0.175, I_arg=default_moment_of_inertia(), kF_arg=1.0, kM_arg=0.0245):
        LeafSystem.__init__(self)

        # Set model constants
        self.g_ = 9.81
        self.m_ = m_arg
        self.L_ = L_arg
        self.I_ = I_arg
        self.kF_ = kF_arg
        self.kM_ = kM_arg

        # Declare inputs to the model.
        # four propellor inputs
        self.DeclareVectorInputPort(name="propellor_force", model_vector=BasicVector(4))

        # Declare state vector of the model
        # x, y, z, roll, pitch, yaw + their velocities
        self.DeclareContinuousState(model_vector=BasicVector(np.zeros(12)), num_q=6, num_v=6, num_z=0)

        # Output state
        # To be used by controller as input, same as state
        self.DeclareVectorOutputPort(name="state", model_value=BasicVector(12), calc=self.CopyStateOut)

        # Output position
        # To be fed through MultibodyPositionToGeometryPose to meshcat
        # first four values is the quaternion, last three are x, y, z
        self.DeclareVectorOutputPort(name="position", model_value=BasicVector(7), calc=self.CopyPositionOut)

        self.source_id_ = None
        self.frame_id_ = None
        self.mbp = None

    def DoHasDirectFeedthrough(self, arg0, arg1):
        """
        :param arg0: input port
        :param arg1: output port
        :return: True if there is direct-feedthrough from given input port to the given output port
        """
        if arg0 == 0 and arg1 == 0:
            return False
        return False

    def CopyStateOut(self, context, output):
        """ Function to obtain entire state
        :param context:  context for performing calculations
        :param output: output to be set
        """
        out = context.get_continuous_state_vector().CopyToVector()
        output.SetFromVector(out)

    def CopyPositionOut(self, context, output):
        """ Function for obtaining output
        :param context: context for performing calculations
        :param output: output to be set
        """
        out = np.zeros((7,))

        # Obtain current state
        state = context.get_continuous_state_vector().CopyToVector()

        # Convert roll-pitch-yaw to quaternion
        quaternion_wxyz = RollPitchYaw(state[3:6]).ToQuaternion().wxyz()
        out[:3] = quaternion_wxyz[1:]
        out[3] = quaternion_wxyz[0]

        # set x and z pos
        out[4:] = state[:3]

        # Send output
        output.SetFromVector(out)

    def DoCalcTimeDerivatives(self, context, derivatives):
        """ Function that gets called to obtain derivatives, for simulation
        :param context: context for performing calculations
        :param derivatives: derivatives of the system to be set at current state
        """
        # Get input
        u = self.EvalVectorInput(context, 0).get_value()

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
        Fgravity_N = np.array([0, 0, -self.m_*self.g_]).reshape((-1, 1))

        # Get current state
        state = context.get_continuous_state_vector().CopyToVector()

        # Extract roll, pitch, yaw and their derivatives
        rpy = RollPitchYaw(state[3:6])
        rpyDt = state[9:12].reshape((-1, 1))

        # Convert roll-pitch-yaw to rotation matrix from inertial frame to body frame
        R_NB = RotationMatrix(rpy).matrix()

        # Calculate net force and acceleration, expressed in inertial frame
        Fnet_N = Fgravity_N + np.matmul(R_NB, Faero_B)
        xyzDDt = Fnet_N / self.m_

        # Calculate body's angular velocity in inertial frame, expressed in body frame
        w_BN_B = rpy.CalcAngularVelocityInChildFromRpyDt(rpyDt)

        # Compute body's angular acceleration, α, in inertia frame, due to the net moment 𝛕 on body,
        # rearrange Euler rigid body equation 𝛕 = I α + ω × (I ω)  and solve for α.
        wIw = np.cross(w_BN_B, np.matmul(self.I_, w_BN_B)).reshape((-1, 1))
        alpha_NB_B = np.linalg.solve(self.I_, Tau_B-wIw)
        alpha_NB_N = np.matmul(R_NB, alpha_NB_B)

        # Calculate the 2nd time-derivative of rpy
        rpyDDt = rpy.CalcRpyDDtFromRpyDtAndAngularAccelInParent(rpyDt, alpha_NB_N)

        # Set derivative of pos by current velocity,
        # and derivative of vel by input, which is acceleration
        deriv = np.zeros((12,))
        deriv[:6] = state[6:]
        deriv[6:9] = xyzDDt.ravel()
        deriv[9:] = rpyDDt.ravel()
        derivatives.get_mutable_vector().SetFromVector(deriv)

    def RegisterGeometry(self, scene_graph):
        """ Create the visual model of the system, and register in scene graph
        :param scene_graph: nexus for all geometry in a Diagram
        """
        # Import the visual model
        self.mbp = MultibodyPlant()
        parser = Parser(self.mbp, scene_graph)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_id = parser.AddModelFromFile(dir_path + "/quadrotor.urdf",
                                           "quadrotor")
        self.mbp.Finalize()

        # Get ids
        self.source_id_ = self.mbp.get_source_id()
        self.frame_id_ = self.mbp.GetBodyFrameIdIfExists(self.mbp.GetBodyIndices(model_id)[0])

    def source_id(self):
        return self.source_id_


# def LQRController(quadrotor, nominal_position=(0, 0, 1)):
#     """
#     :param quadrotor: the quadrotor system
#     :param nominal_position: the position to stabilize to
#     :return: a LQR controller that stabilizes to the nominal position
#     """
#     quad_context_goal = quadrotor.CreateDefaultContext()
#
#     # target stable position
#     x0 = np.zeros(shape=(12,))
#     x0[:3] = nominal_position
#
#     # nominal input for stable position
#     u0 = np.ones(shape=(4,))*quadrotor.m_ * quadrotor.g_ / 4
#
#     # Set the stabilizing target
#     quad_context_goal.FixInputPort(0, u0)
#     quad_context_goal.SetContinuousState(x0)
#
#     # Set up Q and R costs
#     Q = np.identity(12)
#     Q[:6, :6] = 10*np.identity(6)
#     R = np.identity(4)
#
#     # linearized = Linearize(quadrotor, quad_context_goal)
#     # print(quadrotor.mbp)
#
#     return LinearQuadraticRegulator(quadrotor, quad_context_goal, Q, R)


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
                        default=4.0)
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    # Build system diagram
    builder = DiagramBuilder()
    plant = builder.AddSystem(Quadrotor())

    # Connect geometry to scene graph
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterGeometry(scene_graph)
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant.mbp))
    builder.Connect(plant.get_output_port(1), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.source_id()))

    # Add controller
    # controller = builder.AddSystem(LQRController(plant, [0, 0, 1]))
    controller = builder.AddSystem(ConstantVectorSource([0, 0, 0, 0]))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    # builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Add meshcat visualization
    meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph, zmq_url=args.meshcat, open_browser=args.open_browser))
    builder.Connect(scene_graph.get_pose_bundle_output_port(), meshcat.get_input_port(0))

    # Build!
    diagram = builder.Build()

    # Simulate the system
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()

    for i in range(args.trials):
        context.set_time(0.)
        context.SetContinuousState(np.random.randn(12,))
        simulator.Initialize()
        simulator.StepTo(args.duration)
