# Code for running simulation
import numpy as np
from pydrake.geometry import SceneGraph
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer


def simulate(args, plant, controller, initial_state_gen):
    """ Simulate the system with the provided controller
    :param args: command line arguments
    :param plant: system plant to be simulated
    :param controller: controller of the plant
    :param initial_state_gen: function for generating initial state
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
        initial_state = initial_state_gen()
        context.SetContinuousState(initial_state)
        simulator.Initialize()
        simulator.StepTo(args.duration)
