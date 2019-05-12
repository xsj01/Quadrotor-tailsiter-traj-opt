import argparse
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.examples.quadrotor import QuadrotorPlant, StabilizingLQRController
from pydrake.geometry import SceneGraph
from pydrake.math import RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, VectorSystem
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.rendering import MultibodyPositionToGeometryPose
from pydrake.systems.primitives import AffineSystem
from pydrake.all import LeafSystem, BasicVector, ConstantVectorSource, RigidTransform, FramePoseVector


class Block(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        # Declare inputs to the model.
        # In this case, it is acceleration command
        self.DeclareVectorInputPort(name="x", model_vector=BasicVector(1))

        # Declare state vector of the model
        # Here we have x pos, and x vel
        self.DeclareContinuousState(model_vector=BasicVector(np.zeros(2)), num_q=1, num_v=1, num_z=0)

        # Output geometry state
        # To be fed through MultibodyPositionToGeometryPose to meshcat
        # first four values is the quaternion, last three are x, y, z
        self.DeclareVectorOutputPort(name="state", model_value=BasicVector(7), calc=self.CopyStateOut)

        self.source_id_ = None
        self.frame_id_ = None
        self.geometry_pose_port = None
        self.mbp = None

    def CopyStateOut(self, context, output):
        """ Function for obtaining output
        :param context: context for performing calculations
        :param output: output to be set
        """
        # Obtain current state
        x = context.get_continuous_state_vector().CopyToVector()

        # Identity quaternion
        out = np.zeros((7,))
        out[3] = 1

        # set x and z pos
        out[4] = x[0]
        out[6] = 1

        # Send output
        output.SetFromVector(out)

    def DoCalcTimeDerivatives(self, context, derivatives):
        """ Function that gets called to obtain derivatives, for simulation
        :param context: context for performing calculations
        :param derivatives: derivatives of the system to be set at current state
        """
        # Get current state
        x = context.get_continuous_state_vector().CopyToVector()

        # Get input
        u = self.EvalVectorInput(context, 0).GetAtIndex(0)

        # Set derivative of pos by current velocity,
        # and derivative of vel by input, which is acceleration
        derivatives.get_mutable_vector().SetFromVector([x[1], u])

    def RegisterGeometry(self, scene_graph):
        self.mbp = MultibodyPlant()
        parser = Parser(self.mbp, scene_graph)

        model_id = parser.AddModelFromFile(FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
                                           "quadrotor")
        self.mbp.Finalize()

        self.source_id_ = self.mbp.get_source_id()
        self.frame_id_ = self.mbp.GetBodyFrameIdIfExists(self.mbp.GetBodyIndices(model_id)[0])

        # self.geometry_pose_port = self.AllocateGeometryPoseOutputPort()


    def source_id(self):
        return self.source_id_


parser = argparse.ArgumentParser()
parser.add_argument("-N", "--trials",
                    type=int,
                    help="Number of trials to run.",
                    default=5)
parser.add_argument("-T", "--duration",
                    type=float,
                    help="Duration to run each sim.",
                    default=4.0)
MeshcatVisualizer.add_argparse_argument(parser)
args = parser.parse_args()


builder = DiagramBuilder()
#plant = builder.AddSystem(QuadrotorPlant())
plant = builder.AddSystem(Block())

# controller = builder.AddSystem(StabilizingLQRController(plant, [0, 0, 1]))
# builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
command = builder.AddSystem(ConstantVectorSource([0]))
builder.Connect(command.get_output_port(0), plant.get_input_port(0))
# builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

scene_graph = builder.AddSystem(SceneGraph())
plant.RegisterGeometry(scene_graph)

to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant.mbp))
builder.Connect(plant.get_output_port(0), to_pose.get_input_port())
builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.source_id()))
# builder.Connect(plant.get_geometry_pose_output_port(), scene_graph.get_source_pose_port(plant.source_id()))


meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph, zmq_url=args.meshcat, open_browser=args.open_browser))
builder.Connect(scene_graph.get_pose_bundle_output_port(), meshcat.get_input_port(0))


diagram = builder.Build()

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
context = simulator.get_mutable_context()

for i in range(args.trials):
    context.set_time(0.)
    context.SetContinuousState(np.random.randn(2,))
    # context.SetContinuousState([1, 0])
    simulator.Initialize()
    simulator.StepTo(args.duration)
