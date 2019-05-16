# -*- coding: utf-8 -*-
# Implement quadrotor model in python

import argparse
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
from pydrake.all import LeafSystem, BasicVector, ConstantVectorSource, RigidTransform, FramePoseVector, Linearize, RigidBodyPlant, MathematicalProgram, Solve

from dynamics import *

import dynamics_drake as dd

import os

def default_moment_of_inertia():
    return np.array([[0.0023, 0, 0],
                     [0, 0.0023, 0],
                     [0, 0, 0.0040]])

def rot_mat(theta,axis,autodiff=False):
    #return R_0_to_1
    #R transform cooridinates from 0 frame to 1 frame
    #theta counterclockwise rotate 0 to 1
    if autodiff:
        cos=dd.cos
        sin=dd.sin
    else:
        cos=np.cos
        sin=np.sin
    if axis==0 or axis=='x':
        return np.array([[1.,0,0],
                        [0,cos(theta),-sin(theta)],
                        [0,sin(theta),cos(theta)]
                        ])
    if axis==1 or axis=='y':
        return np.array([[cos(theta),0,sin(theta)],
                        [0,1,0],
                        [-sin(theta),0,cos(theta)]])
    if axis==2 or axis=='z':
        return np.array([[cos(theta),-sin(theta),0],
                        [sin(theta),cos(theta),0],
                        [0,0,1.]])


class Tailsitter(LeafSystem):
    def __init__(self, m_arg=0.1, L_arg=0.175, I_arg=default_moment_of_inertia(), kF_arg=1.0, kM_arg=0.0245):
        LeafSystem.__init__(self)

        # Set model constants
        self.g_ = 9.81
        self.m_ = m_arg
        self.L_ = L_arg
        self.I_ = I_arg
        self.kF_ = kF_arg
        self.kM_ = kM_arg

        self.delta=0.1
        self.rho=1.29
        self.S = 0.1955
        self.chi=np.pi/4
        self.d=0.4
        self.kappa=1.
        self.c = 0.24
        #self.I_=np.diag([10,1,10])
        #self.m_=1.
        #self.g_=9.8

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
        out[:4] = quaternion_wxyz[::-1]
        #out[3] = quaternion_wxyz[0]

        # set x and z pos
        out[4:] = state[:3]

        # Send output
        output.SetFromVector(out)

    def cal_F_M_1(self,state,u):

        roll=state[3]
        pitch=state[4]
        yaw=state[5]

        #calculate V in frame 1
        R_roll=np.array([[1,0,0,],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
        R_pitch=np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
        R_yaw=np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])

        R_1_to_0=(R_yaw.dot(R_pitch).dot(R_roll))
        #R_1_to_0=R_roll.dot(R_pitch).dot(R_yaw)
        R_0_to_1=R_1_to_0.T

        v_0=state[6:9]

        v_1=R_0_to_1.dot(v_0)

        V=np.linalg.norm(v_0)


        #calculate V in frame 2 and alpha Beta

        R_1_to_2=np.array([[0,0,1],[cos(self.chi),-sin(self.chi),0],[sin(self.chi),cos(self.chi),0]])
        R_2_to_1=R_1_to_2.T

        v_2=R_1_to_2.dot(v_1)


        #calculate Force Momentum in Frame 2

        FM_aero=np.array([[1,1,1,1],[-1,1,-1,1],[1,1,-1,-1.],[-1,1,1,-1.]]).dot(u)
        Fx_2=FM_aero[0]*cos(self.delta)
        Mx_2=FM_aero[1]*(sin(self.delta)*self.d+self.kappa*cos(self.delta))
        My_2=FM_aero[2]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*sin(self.chi)
        Mz_2=FM_aero[3]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*cos(self.chi)
        M_propeller_2=np.array([Mx_2,My_2,Mz_2])
        F_propeller_2=np.array([Fx_2,0,0.])

        if V==0:
            alpha=0.
            beta=0.
            F_net_2=F_propeller_2
            M_net_2=M_propeller_2
        else:
            alpha=np.arctan2(v_2[2],v_2[0])
            #vxy=np.linalg.norm(v_2[0::2])
            beta=np.arcsin(v_2[1]/V)
            #beta=atan2(v_2[1],vxy)

            L_aero_2=0.5*self.rho*V**2*self.S*get_CL(alpha,beta)
            D_aero_2=0.5*self.rho*V**2*self.S*get_CD(alpha,beta)
            Y_aero_2=0.5*self.rho*V**2*self.S*get_CY(alpha,beta)

            ll_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cll(alpha,beta)
            m_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cm(alpha,beta)
            n_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cn(alpha,beta)

            DYL=np.array([D_aero_2,Y_aero_2,L_aero_2]).reshape(-1)

            F_aero_2=np.array([[-cos(alpha),0,sin(alpha)],[0,1,0],[-sin(alpha),0,-cos(alpha)]]).dot(DYL)
            M_aero_2=np.array([ll_aero_2,m_aero_2,n_aero_2]).reshape(-1)

            F_net_2=F_propeller_2+F_aero_2
            M_net_2=M_aero_2+M_propeller_2


        #Calculate Force Momentum in frame 0

        F_net_1=R_2_to_1.dot(F_net_2)

        M_net_1=R_2_to_1.dot(M_net_2)

        #print '1 to 0', R_1_to_0

        #F_gravity_0=np.array([0.,0.,-self.m_*self.g_])

        #F_net_0=F_net_0+F_gravity_0

        return F_net_1.reshape((-1,1)),M_net_1.reshape((-1,1))


    def DoCalcTimeDerivatives(self, context, derivatives, beta0=0.):
        """ Function that gets called to obtain derivatives, for simulation
        :param context: context for performing calculations
        :param derivatives: derivatives of the system to be set at current state
        """
        # Get input
        u = self.EvalVectorInput(context, 0).get_value()

        # Calculate force exerted by each motor, expressed in Body frame
        # uF_Bz = self.kF_ * u

        # # Compute net force, expressed in body frame
        # Faero_B = np.array([0, 0, np.sum(uF_Bz)]).reshape((-1, 1))

        # # Compute x and y moment caused by motor forces, expressed in body frame
        # Mx = self.L_ * (uF_Bz[1] - uF_Bz[3])
        # My = self.L_ * (uF_Bz[2] - uF_Bz[0])

        # # Compute moment in z, caused by air reaction force on rotating blades, expressed in body frame
        # uTau_Bz = self.kM_ * u
        # Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]

        # # Net moment on body about its center of mass, expressed in body frame
        # Tau_B = np.array([Mx, My, Mz]).reshape((-1, 1))

        # # Compute gravity force, expressed in Newtonian frame (i.e. world frame)
        Fgravity_N = np.array([0, 0, -self.m_*self.g_]).reshape((-1, 1))

        # # Get current state
        state = context.get_continuous_state_vector().CopyToVector()

        # Extract roll, pitch, yaw and their derivatives
        rpy = RollPitchYaw(state[3:6])
        rpyDt = state[9:12].reshape((-1, 1))

        #calculate Force Moment in body frame
        Faero_B,Tau_B=self.cal_F_M_1(state.reshape(-1),u.reshape(-1))

        

        # Convert roll-pitch-yaw to rotation matrix from inertial frame to body frame
        R_NB = RotationMatrix(rpy).matrix()

        #print 'R_NB',R_NB
        #raw_input()
        
        
        

        # # Calculate net force and acceleration, expressed in inertial frame
        Fnet_N = Fgravity_N + np.matmul(R_NB, Faero_B)

        #print 'R_NB',R_NB
        #raw_input()
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

        #raw_input()

    def RegisterGeometry(self, scene_graph):
        """ Create the visual model of the system, and register in scene graph
        :param scene_graph: nexus for all geometry in a Diagram
        """
        # Import the visual model
        self.mbp = MultibodyPlant()
        parser = Parser(self.mbp, scene_graph)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_id = parser.AddModelFromFile(dir_path + "/tailsitter.urdf",
                                           "tailsitter")
        self.mbp.Finalize()

        # Get ids
        self.source_id_ = self.mbp.get_source_id()
        self.frame_id_ = self.mbp.GetBodyFrameIdIfExists(self.mbp.GetBodyIndices(model_id)[0])

    def source_id(self):
        return self.source_id_

    def cal_alpha_beta_u_by_V(self,desired_V):
        V=np.linalg.norm(desired_V)

        if V==0:
            return 0,0,self.m_*self.g_/4*np.ones(4)/cos(self.delta),np.array([0.,0.,0.])


        mp = MathematicalProgram()
        alpha=mp.NewContinuousVariables(1, "alpha")[0]
        beta=mp.NewContinuousVariables(1, "beta")[0]
        theta2=mp.NewContinuousVariables(1, "theta")[0]

        theta1=np.arctan2(desired_V[1],desired_V[0])
        theta0=np.arctan2(desired_V[2],np.linalg.norm(desired_V[0:2]))
        #print theta

        R_v_to_b=np.array([[dd.cos(beta),-dd.sin(beta),0],
                            [dd.sin(beta),dd.cos(beta),0],
                            [0,0,1.]])
        R_b_to_2=np.array([[dd.cos(alpha),0,-dd.sin(alpha)],
                            [0,1,0],
                            [dd.sin(alpha),0,dd.cos(alpha)]])

        R_0_to_v=rot_mat(theta2,0,autodiff=True).dot(rot_mat(-np.pi,0)).dot(rot_mat(theta0,1)).dot(rot_mat(-theta1,2))

        R_0_to_2=R_b_to_2.dot(R_v_to_b).dot(R_0_to_v)
        #F_gravity_v=np.array([-sin(theta0),0.,cos(theta0)])*self.m_*self.g_
        F_gravity_0=np.array([0,0,-1.])*self.m_*self.g_
        #f1=self.m_*self.g_*dd.sin(alpha+theta)
        #F_gravity_2=R_b_to_2.dot(R_v_to_b).dot(F_gravity_v)
        F_gravity_2=R_0_to_2.dot(F_gravity_0)



        L_aero_2=0.5*self.rho*V**2*self.S*dd.get_CL(alpha,beta)[0,0]
        D_aero_2=0.5*self.rho*V**2*self.S*dd.get_CD(alpha,beta)[0,0]
        Y_aero_2=0.5*self.rho*V**2*self.S*dd.get_CY(alpha,beta)[0,0]

        F_aero_2=np.array([[-dd.cos(alpha),0,dd.sin(alpha)],
                            [0,1,0],
                            [-dd.sin(alpha),0,-dd.cos(alpha)]]).dot([D_aero_2,Y_aero_2,L_aero_2])

        # f2=L_aero_2*dd.cos(alpha)+D_aero_2*dd.sin(alpha)

        # Q1=(f1-f2)**2

        
        # mp.AddConstraint(Q1==0)
        # mp.AddConstraint(Y_aero_2**2==0)

        mp.AddConstraint(F_gravity_2[1]+F_aero_2[1]==0)
        mp.AddConstraint(F_gravity_2[2]+F_aero_2[2]==0)
        mp.AddConstraint(beta==0.)

        #mp.AddLinearConstraint(alpha>=-deg2rad(30))
        #mp.AddQuadraticCost(Q)
        result = Solve(mp)
        alpha=result.GetSolution(alpha)
        beta=result.GetSolution(beta)
        theta2=result.GetSolution(theta2)

        print alpha,beta,theta2
        #F_gravity_2x=result.GetSolution(F_gravity_2[0])
        #print [F_gravity_2x]
        print result.is_success()
        # print get_CL(alpha,0.)
        # print get_CD(alpha,0.)
        #R_0_to_v=rot_mat(theta2,0).dot(rot_mat(theta0,1)).dot(rot_mat(-theta1,2))
        R_v_to_b=np.array([[dd.cos(beta),-dd.sin(beta),0],
                            [dd.sin(beta),dd.cos(beta),0],
                            [0,0,1.]])
        R_b_to_2=np.array([[dd.cos(alpha),0,-dd.sin(alpha)],
                            [0,1,0],
                            [dd.sin(alpha),0,dd.cos(alpha)]])

        R_0_to_v=rot_mat(theta2,0).dot(rot_mat(-np.pi,0)).dot(rot_mat(theta0,1)).dot(rot_mat(-theta1,2))

        R_0_to_2=R_b_to_2.dot(R_v_to_b).dot(R_0_to_v)
        #F_gravity_v=np.array([-sin(theta0),0.,cos(theta0)])*self.m_*self.g_
        F_gravity_0=np.array([0,0,-1.])*self.m_*self.g_
        #f1=self.m_*self.g_*dd.sin(alpha+theta)
        #F_gravity_2=R_b_to_2.dot(R_v_to_b).dot(F_gravity_v)
        F_gravity_2=R_0_to_2.dot(F_gravity_0)


        L_aero_2=0.5*self.rho*V**2*self.S*get_CL(alpha,beta)
        D_aero_2=0.5*self.rho*V**2*self.S*get_CD(alpha,beta)
        Y_aero_2=0.5*self.rho*V**2*self.S*get_CY(alpha,beta)

        ll_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cll(alpha,beta)
        m_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cm(alpha,beta)
        n_aero_2=0.5*self.rho*V**2*self.c*self.S*get_Cn(alpha,beta)

        DYL=np.array([D_aero_2,Y_aero_2,L_aero_2]).reshape(-1)

        F_aero_2=np.array([[-cos(alpha),0,sin(alpha)],[0,1,0],[-sin(alpha),0,-cos(alpha)]]).dot(DYL)

        #F_gravity_2

        FM0=-(F_aero_2[0]+F_gravity_2[0])/cos(self.delta)
        FM1=-ll_aero_2/(sin(self.delta)*self.d+self.kappa*cos(self.delta))
        FM2=-m_aero_2/(cos(self.delta)*self.d-self.kappa*sin(self.delta))/sin(self.chi)
        FM3=-n_aero_2/(cos(self.delta)*self.d-self.kappa*sin(self.delta))/cos(self.chi)
        # Fx_2=FM_aero[0]*cos(self.delta)
        # Mx_2=FM_aero[1]*(sin(self.delta)*self.d+self.kappa*cos(self.delta))
        # My_2=FM_aero[2]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*sin(self.chi)
        # Mz_2=FM_aero[3]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*cos(self.chi)
        FM=np.array([FM0,FM1,FM2,FM3])

        u=np.linalg.inv(np.array([[1,1,1,1],[-1,1,-1,1],[1,1,-1,-1.],[-1,1,1,-1.]])).dot(FM)

        #rpy 
        
        R_1_to_2=np.array([[0,0,1],[cos(self.chi),-sin(self.chi),0],[sin(self.chi),cos(self.chi),0]])
        R_2_to_1=R_1_to_2.T

        R_1_to_0=R_0_to_2.T.dot(R_1_to_2)

        print R_1_to_0

        rpy=RollPitchYaw(R_1_to_0)#.SetFromRotationMatrix(R_1_to_0)
    
        #print
        return alpha,beta,u,[rpy.roll_angle(),rpy.pitch_angle(),rpy.yaw_angle()]    

    #def cal_rpy(alpha,beta,V)
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
    plant = builder.AddSystem(Tailsitter())

    # Connect geometry to scene graph
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterGeometry(scene_graph)
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant.mbp))
    builder.Connect(plant.get_output_port(1), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.source_id()))

    # Add controller
    # controller = builder.AddSystem(LQRController(plant, [0, 0, 1]))
    v=np.array([2.,0.,0.])
    ts=Tailsitter()
    _,_,u,rpy=ts.cal_alpha_beta_u_by_V(v)
    print u,rpy
    initial_state=np.zeros(12)
    initial_state[2]=1.
    initial_state[3:6]=rpy[:]
    #initial_state[3]=0.7
    initial_state[6:9]=v[:]

    #u=ts.m_*ts.g_/4*np.ones(4)/cos(ts.delta)

    #print u

    controller = builder.AddSystem(ConstantVectorSource(u))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    # builder.Connect(plant.get_output_port(0), controller.get_input_port(0))

    # Add meshcat visualization
    meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph, zmq_url=args.meshcat, open_browser=args.open_browser))
    builder.Connect(scene_graph.get_pose_bundle_output_port(), meshcat.get_input_port(0))

    # Build!
    diagram = builder.Build()

    

    # Simulate the system
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.)
    context = simulator.get_mutable_context()

    for i in range(args.trials):
        context.set_time(0.)
        context.SetContinuousState(initial_state)
        simulator.Initialize()
        simulator.StepTo(args.duration)
