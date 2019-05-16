# Simulate visualize the trajectory generated by trajectory optimization
import argparse
import os
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import SceneGraph
from pydrake.math import RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, VectorSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.all import LeafSystem, BasicVector

from utilities import load_traj


class PuppetPlant(LeafSystem):
    """ A system purely for visualizing trajectory
        Does not implement dynamics
    """
    def __init__(self, filename, model_name):
        """
        :param filename: trajectory file name
        :param model_name: model name, e.g. "quadrotor" or "tailsitter"
        """
        LeafSystem.__init__(self)

        # Output geometry state
        # To be fed through MultibodyPositionToGeometryPose to meshcat
        # first four values is the quaternion, last three are x, y, z
        self.DeclareVectorOutputPort(name="position", model_value=BasicVector(7), calc=self.CopyPositionOut)

        self.source_id_ = None
        self.frame_id_ = None
        self.mbp = None

        self.traj, self.u_traj, self.time_array = load_traj(filename)
        self.final_time = self.time_array[-1][0]
        self.model_name = model_name

    def CopyPositionOut(self, context, output):
        """ Function for obtaining current geometry state, for visualization
        :param context: context for performing calculations
        :param output: output to be set
        """
        # Get the state corresponding to simulation time
        time = context.get_time()
        index = np.where(self.time_array <= time)[0][-1]
        state = self.traj[index]

        out = np.zeros((7,))
        # Convert roll-pitch-yaw to quaternion
        quaternion_wxyz = RollPitchYaw(state[3:6]).ToQuaternion().wxyz()
        out[:3] = quaternion_wxyz[1:]
        out[3] = quaternion_wxyz[0]

        # set x and z pos
        out[4:] = state[:3]

        # Set output
        output.SetFromVector(out)

    def RegisterGeometry(self, scene_graph):
        """ Create the visual model of the system, and register in scene graph
        :param scene_graph: nexus for all geometry in a Diagram
        """
        # Import the visual model
        self.mbp = MultibodyPlant()
        parser = Parser(self.mbp, scene_graph)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.model_name == "quadrotor":
            urdf_path = dir_path + "/../models/quadrotor.urdf"
        else:
            urdf_path = dir_path + "/../models/tailsitter.urdf"
        model_id = parser.AddModelFromFile(urdf_path,
                                           self.model_name)
        self.mbp.Finalize()

        # Get ids
        self.source_id_ = self.mbp.get_source_id()
        self.frame_id_ = self.mbp.GetBodyFrameIdIfExists(self.mbp.GetBodyIndices(model_id)[0])

    def source_id(self):
        return self.source_id_


def visualize(args, plant):
    """ Visualize the trajectory in meshcat
    :param args: command line parsed arguments
    :param plant: system plant that contains the trajectory to be visualized
    """
    # Build system diagram
    builder = DiagramBuilder()
    plant = builder.AddSystem(plant)

    # Connect geometry to scene graph
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterGeometry(scene_graph)
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant.mbp))
    builder.Connect(plant.get_output_port(0), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.source_id()))

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
        simulator.Initialize()
        simulator.StepTo(args.duration)


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

    # Load quadrotor trajectory, and set duration of simulation
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + "/traj/{}_{}.npz".format(args.filename, args.model)
    puppet_plant = PuppetPlant(filename, model_name=args.model)
    args.duration = puppet_plant.final_time

    # Display in meshcat
    visualize(args, plant=puppet_plant)
