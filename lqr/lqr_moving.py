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
        self.rpy=rpy[:]
        self.target_u=target_u

    def get_output(self,current_state,Q=None,R=None, return_KS=False):
        target_state=self.target_state
        # nominal input for stable point
        #u0 = np.ones(shape=(4,))*self.plant.m_ * self.plant.g_ / 4



        u0=self.target_u
        ctrl_indx=self.ctrl_indx
        # linearize around nominal point
        A, B = self.dynamics_linearizer.get_AB(target_state,u0)

        if Q is None:
            Q = np.identity(12)
            Q[:6, :6] = 50*np.identity(6)

        Q=Q[ctrl_indx,:]
        Q=Q[:,ctrl_indx]

        A=A[ctrl_indx,:]
        A=A[:,ctrl_indx]

        B=B[ctrl_indx,:]

        if R is None:
            R = np.identity(4)*10

        K, S = LinearQuadraticRegulator(A, B, Q, R)

        x_bar = np.array(current_state[ctrl_indx]) - np.array(target_state[ctrl_indx])
        x_bar = x_bar.reshape((-1, 1))

        # u_bar = u - u0
        u_bar = -np.matmul(K, x_bar).reshape((-1,))
        out = u_bar + u0

        if return_KS:
            new_K=np.zeros((4,12))
            new_S=np.zeros((12,12))

            # print S.shape
            # print 
            new_K[:,ctrl_indx]=K[:,:]
            j=0
            for i in ctrl_indx:
                new_S[i,ctrl_indx]=S[j,:]
                j+=1
            #print S
            #print new_S[ctrl_indx,:][:,ctrl_indx]-S
            #raw_input()
            #new_S[:,ctrl_indx][ctrl_indx]=S[:,:]
            return out,(new_K,new_S)
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

        v=np.array([1.9,1.9,0.])
        target_state=np.zeros(12)
        target_state[2]=1
        target_state[1]=-1
        target_state[6:9]=v[:]
        ctrl_indx=[2]+range(3,12)

        self.ctrl1=StableStateController(plant, dynamics_linearizer,target_state,ctrl_indx)

        v=-v
        #v=np.array([0.,-2.5,0.])
        target_state=np.zeros(12)
        target_state[2]=1.
        target_state[6:9]=v[:]
        ctrl_indx=[2]+range(3,12)

        self.ctrl2=StableStateController(plant, dynamics_linearizer,target_state,ctrl_indx)

        v=np.array([0.,-2.,0.])
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
        self.last_state=np.zeros(3)
        self.count=0

        self.end=False

        self.record=[]
        for i in range(17):
            self.record.append([])

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

        #print current_state[0]

        if self.count!=0 and np.linalg.norm(self.last_state-current_state[0:3])>=1 and (not self.end):
            write_data(self.record,'./dqxr.pkl')
            self.count=0
            self.end=True

        self.count+=1

        if self.count<600:
            out=self.ctrl1.get_output(current_state)
        else:
            out=self.ctrl2.get_output(current_state,R = 3*np.identity(4))
        y[:] = out

        for i in range(12):
            #print i
            self.record[i].append(current_state[i])
            #print self.record[i],current_state[i]
            #raw_input()
        for i in range(4):
            self.record[12+i].append(out[i])
        #print self.record[0],current_state[i]
        #raw_input
        #print len(self.record[0])
        tt=context.get_time()
        self.record[16].append(tt)
        self.last_state=current_state[0:3]

def write_data(record,file):

    import pickle as pkl
    record=np.array(record)
    #print record[0,0:20]
    with open(file, 'wb') as f:
        pkl.dump(record,f)
    print record[0,0:20],'done'
    pass



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
Sqr
    if args.model == "quadrotor":
        plant = Quadrotor()
        linearizer = QuadrotorDynamicsLinearizer()
    else:
        plant = Tailsitter()
        
        linearizer = TailsitterDynamicsLinearizer()
    controller = LQRController(plant=plant, dynamics_linearizer=linearizer)#, target_v=v)

    def initial_state_gen():
        return controller.ctrl1.target_state
        #return np.zeros(12)
        # return np.random.randn(12,)

    # Display in meshcat
    simulate(args, plant=plant, controller=controller, initial_state_gen=initial_state_gen)
