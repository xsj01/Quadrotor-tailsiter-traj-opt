# Implement a PID controller for height regulation
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (MeshcatVisualizer, DiagramBuilder, SceneGraph, MultibodyPositionToGeometryPose,
                         ConstantVectorSource, Simulator, VectorSystem, SignalLogger)

from ..models.quadrotor import Quadrotor


class HeightController(VectorSystem):
    """ Quadrotor controller that controls height
    """
    def __init__(self, height=1.0):
        """
        :param height: the height to control to
        """
        VectorSystem.__init__(self,
                              12,  # number of inputs, the state of the quadrotor
                              4)  # number of outputs, the inputs to four motors
        self.target_height = height
        self.integral = 0.0
        self.current_time = 0.0

    def DoCalcVectorOutput(self, context, u, x, y):
        """
        :param context: the context of the quadrotor
        :param u: input from the quadrotor, which is its state
        :param x: the system's internal state
        :param y: the output to the quadrotor
        """
        # get z and z vel
        z = u[2]
        vz = u[8]

        # get z error, z error derivative, z error integral
        ez = self.target_height-z
        ezDot = -vz
        current_time = context.get_time()

        # reinitialize if current time is 0.0
        if current_time == 0:
            dt = 0
            self.integral = 0
        else:
            dt = current_time-self.current_time
        self.current_time = current_time

        self.integral += ez * dt

        # PID controller
        Kp = 2.0
        Ki = 1.0
        Kd = 1.0
        y[:] = [Kp*ez + Ki*self.integral + Kd*ezDot for _ in range(4)]


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
    controller = builder.AddSystem(HeightController(1.0))
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Add logger
    state_dim = 12
    state_log = builder.AddSystem(SignalLogger(state_dim))
    state_log.DeclarePeriodicPublish(0.0333, 0.0)
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))

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
        initial_state = np.zeros(shape=(12,))
        initial_state[5] = 0.7
        context.SetContinuousState(initial_state)
        simulator.Initialize()
        simulator.StepTo(args.duration)

        plt.close()
        plt.figure().set_size_inches(10, 5)
        plt.plot(state_log.sample_times(), state_log.data()[2, :])
        plt.plot(state_log.sample_times(), state_log.data()[8, :])
        plt.grid(True)
        plt.legend(["body_z", "body_z_d"])
        plt.show()