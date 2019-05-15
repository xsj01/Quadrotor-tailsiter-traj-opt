# Time-varying LQR around the nominal trajectory and u trajectory from trajectory optimization
import argparse
import numpy as np
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.framework import VectorSystem, DiagramBuilder
from pydrake.geometry import SceneGraph
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.analysis import Simulator

from quadrotor import Quadrotor
from utilities import load_traj


class TVLQR(VectorSystem):
    """ Quadrotor controller that uses time-varying LQR around nominal trajectory
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
        # if index < self.u_traj.shape[0]:
        #     u = self.u_traj[index]
        # else:
        #     u = np.zeros_like(self.u_traj[0])

        state = self.traj[index]
        current_state = u
        print(state)
        print(current_state)
        print("================")


        # Set as output, i.e control input to the quadrotor
        y[:] = np.zeros_like(self.u_traj[0])


def simulate(args, plant, controller, initial_state):
    """ Simulate the system with the provided controller
    :param args: command line arguments
    :param plant: system plant to be simulated
    :param controller: controller of the plant
    :param initial_state: initial state of the plant
    """
    # Build system diagram
    builder = DiagramBuilder()
    plant = builder.AddSystem(plant)

    # Connect geometry to scene graph
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterGeometry(scene_graph)
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant.mbp))
    builder.Connect(plant.get_output_port(1), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.source_id()))

    # Add controller
    controller = builder.AddSystem(controller)
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

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
        context.SetContinuousState(initial_state)
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
    MeshcatVisualizer.add_argparse_argument(parser)
    args = parser.parse_args()

    # Load quadrotor u trajectory, and set duration of simulation & initial state
    filename = "traj/{}.npz".format(args.filename)
    controller = TVLQR(filename)
    args.duration = controller.final_time
    initial_state = controller.initial_state

    quadrotor = Quadrotor()
    # Display in meshcat
    simulate(args, plant=quadrotor, controller=controller, initial_state=initial_state)
