#Region of Attraction analysis
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

from lqr_moving import StableStateController,LQRController
import pickle as pkl
import time

import math

plant = Tailsitter()
linearizer = TailsitterDynamicsLinearizer()

v=np.array([2.,2.,0.])
target_state=np.zeros(12)
target_state[2]=1
#target_state[1]=-1
target_state[6:9]=v[:]
ctrl_indx=[2]+range(3,12)

Q = np.identity(12)*3.
Q[:6, :6] = 5*np.identity(6)
Q[6:9,6:9]= 10*np.identity(3)
R = np.identity(4)*2


controller=StableStateController(plant,linearizer,target_state,ctrl_indx)

target_state[3:6]=controller.rpy



xf=target_state.copy()
#print target_state


def calcF(xn):
    # Feel free to bring in whatever
    # global variables you need, e.g.:
    #global pendulum_plant
    
    '''
    Code submission for 3.5: populate
    this function to calculate the closed-loop
    system dynamics f(x) at the input point.
    '''

    u= controller.get_output(xn, Q=Q,R=R)
    #print u,'u'
    #K,S=ks

    return linearizer.dynamics( xn,u)

    #return np.zeros(4)

# print controller.target_state
# print calcF(controller.target_state)
# raw_input()

# Calculates V(xn)
def calcV(xn):
    # Feel free to bring in whatever
    # global variables you need

    '''
    Code submission for 3.5: populate
    this function to calculate V(x)
    at the input point.
    '''
    #global S, xf
    u,ks = controller.get_output(xn,Q=Q,R=R,return_KS=True)
    K,S=ks
    return np.dot(xn - xf, np.dot(S, xn - xf))
    
    #return 0.

# Calculates \dot{V}(xn).
def calcVdot(xn):
    # Feel free to bring in whatever
    # global variables you need
    
    '''
    Code submission for 3.5: populate
    this function to calculate Vdot(x)
    at the input point.
    '''
    u,ks = controller.get_output(xn,Q=Q,R=R,return_KS=True)
    K,S=ks
    #print calcF(xn)
    #raw_input()
    sb=np.dot(np.dot(S, xn - xf), calcF(xn))
    if sb is np.nan or sb is -np.nan:
        print xn,'sb'
    return np.dot(np.dot(S, xn - xf), calcF(xn))


start_time = time.time()

# Sample f, V, and Vdot over
# a grid defined by these parameters.
# (Odd numbers are good because there'll be
# a bin at exactly the origin.
# These are slightly strange numbers as we've
# tried to default these to something as small
# as possible while still giving reasonable results.
# Feel free to increase if your computer and patience
# can handle it.)

ind_x=6
ind_y=7
ind_z=2

name_x=r'\dot{x}'
name_y=r'\dot{y}'

# name_x='Roll'
# name_y='Pitch'

n_bins_x = 41
n_bins_y = 41
n_bins_z = 1
# For theta and thetad, we only need to span
# a small region around the fixed point
x_width = 3
y_width = 3
# x_width = np.pi/6
# y_width = np.pi/6
# For \dot{theta_2}, though, the default
# parameters for our pendulum lead us to
# need to search larger absolute \dot{theta_2}
# values (because the inertial wheel is relatively
# light).
# z_width = 2
# z_width = np.pi/2
z_width=0.

x = np.linspace(xf[ind_x]-x_width, xf[ind_x]+x_width, n_bins_x)
y = np.linspace(xf[ind_y]-y_width, xf[ind_y]+y_width, n_bins_y)

# print x
# print y
# raw_input()
z = np.linspace(xf[ind_z]-z_width, xf[ind_z]+z_width, n_bins_z)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

def calc_over_array(f):
    def state(dx,dy,dz):
        #v=np.array([2.,2.,0.])
        state=xf.copy()
        #state[2]=1.
        #target_state[1]=-1
        #state[6:9]=v[:]
        state[ind_x]=dx
        #print ind_y,dy
        state[ind_y]=dy
        state[ind_z]=dz
        #print state
        #raw_input()
        return state
    # ss=state(x[20],y[20],z[0])
    # print xf
    # print ss
    # print calcF(xf)
    # print calcF(ss)
    # # print calcVdot(ss)
    # raw_input()
    #sss=state(x[0],y[-1],z[0])
    # print sss
    # print xf
    # print sss-xf
    # print f(sss)
    # raw_input()

    return np.array([[[f(state(dx,dy,dz)) for dz in z] for dy in y] for dx in x])

#print calcV(target_state)



#raw_input()
V_samples = calc_over_array(calcV)
#print V_samples
f_samples = calc_over_array(calcF)
Vdot_samples = calc_over_array(calcVdot)

# print Vdot_samples[:,0,0]

# print V_samples[20,20,0]
# print f_samples[20,20,0]

# print f_samples[22,18,0]

# print f_samples[24,16,0]
# print Vdot_samples[20,20,0]



elapsed = time.time() - start_time
print "Computed %d x %d x %d sampling in %f seconds" % (n_bins_x, n_bins_y, n_bins_z, elapsed)



# This cell plots the samples using color-coded plots.
# Color coding:
#   V: blue = low-value, red = high-value
#   Vdot: blue = low value, yellow = around 0, red = high value
# The plot of Vdot is overlayed with a quiver plot of the samples
# of f.

# Select with slice of \dot{theta_2} we'll
# plot... this slice should be close to 0,
# as it's the middle bin.
theta2d_plotting_slice = n_bins_z / 2

plt.figure().set_size_inches(6,12)
plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.4, hspace=0.5)
# Plot V
Xplot, Yplot = np.meshgrid(x, y)
plt.subplot(2, 1, 1)
plt.pcolormesh(Xplot, Yplot, V_samples[:, :, theta2d_plotting_slice])
plt.title(r"V(x) in $\dot{x}-\dot{y}$ plane")# % z[theta2d_plotting_slice])
plt.xlabel("$%s$"%name_x)
plt.ylabel("$%s$"%name_y)
#plt.colorbar()

# Plot Vdot
# Use a sigmoid to try to squash the huge range of Vdot
# into something more visually appealing.
def sigmoid(x):
    #return x
    return 1. / (1 + np.exp(-x/4.))

plt.subplot(2, 1, 2)
#plt.figure().set_size_inches(12,12)
#fig = plt.figure(figsize=(8,8))
#plt.figure().set_size_inches(8,8)
Vdot_viz = sigmoid(Vdot_samples[:, :, theta2d_plotting_slice])
plt.pcolormesh(Xplot, Yplot, Vdot_viz, vmin=0.,vmax=1.)
plt.title("$\\dot{V}(x)$  overlaid with phase diagram f(x) in $%s-%s$ plane"%(name_x,name_y))# % z[theta2d_plotting_slice])
plt.xlabel("$%s$"%name_x)
plt.ylabel("$%s$"%name_y)
#plt.colorbar()

# Don't plot a quiver arrow at *every* point, but instead
# every ds points
# (lower = more quiver arrows)
ds = 2

# for f_samples[::ds, ::ds, 
#            theta2d_plotting_slice, ind_x], f_samples[::ds, ::ds, theta2d_plotting_slice, ind_y]

#print f_samples
plt.quiver(Xplot[::ds, ::ds], 
           Yplot[::ds, ::ds], 
           f_samples[::ds, ::ds, 
           theta2d_plotting_slice, ind_x], f_samples[::ds, ::ds, theta2d_plotting_slice, ind_y]);
plt.plot(target_state[ind_x],target_state[ind_y],'ro')


def estimate_rho(V, Vdot):
    '''
    Code submission for 3.6
    Fill in this function to use the samples of V and Vdot
    (Each array has dimension [n_bins, n_bins, n_bins_theta2d])
    to compute a maximal rho indicating the region of attraction
    of the fixed point at the upright.
    '''
    ## User code
    smallest_counterexample_V = np.max(V)
    for dx in range(V.shape[0]):
        for dy in range(V.shape[1]):
            for dz in range(V.shape[2]):
                if Vdot[dx, dy, dz] > 0 and V[dx, dy, dz] < smallest_counterexample_V:
                    smallest_counterexample_V = V[dx, dy, dz]
    return smallest_counterexample_V - 0.01
    
    return 0.

def gen_states(state_ranges,ctrl_indx,target_state):
    state=np.zeros(12)
    for i in ctrl_indx:
        state[i]=np.random.uniform(target_state[i]-state_ranges[i],target_state[i]+state_ranges[i])
    return state

def estimate_rho_MC(state_ranges,ctrl_indx,target_state,maxnum=5000000,smallest_counterexample_V=100):
    #rho=rho_init
    rholist=[]

    for i in range(maxnum):
        xn=gen_states(state_ranges,ctrl_indx,target_state)
        V=calcV(xn)
        #print xn
        #print 'V',V,smallest_counterexample_V
        if V>=smallest_counterexample_V:continue

        Vdot=calcVdot(xn)
        if Vdot>0:
            #print 'Vdot',Vdot
            print V
            smallest_counterexample_V=V
        rholist.append(smallest_counterexample_V)
        #raw_input()

    return smallest_counterexample_V - 0.01,rholist

    pass

state_ranges=[]
for i in range(12):
    state_ranges.append(0)

state_ranges[2]=2
state_ranges[3:6]=[np.pi/2]*3
state_ranges[6:9]=[2]*3
state_ranges[9:]=[2]*3
# rho,rholist=estimate_rho_MC(state_ranges,ctrl_indx,target_state)
# with open('roa_mc.pkl','wb') as f:
#     pkl.dump(rholist,f)
# plt.figure(figsize=(8,8))
# plt.plot(rholist)
# plt.xlabel('Iteration Steps')
# plt.ylabel(r'$\rho$')
# plt.title(r'Monte Carlo Simulation Result of $\rho$')
#rho=1.73849017987
rho=1.6451324023
#plt.figure
#rho = estimate_rho(V_samples, Vdot_samples)
print "Region of attraction estimated at V(x) <= ", rho

# Plot Vdot again, but overlay the region of attraction -- which,
# for quadratic V, is an ellipse.
#fig = plt.figure(figsize=(8,8))
# plt.figure().set_size_inches(8,8)
# # ax = fig.add_subplot(1, 1, 1)
# # fig.set_size_inches(12,6)
# plt.pcolormesh(Xplot, Yplot, Vdot_viz,vmin=0.,vmax=1.)

# The part of S we care about is the 2x2 submatrix from the 1st and 3rd rows
# and columns.
u,ks = controller.get_output(xf,Q=Q,R=R,return_KS=True)
K,S=ks

S_sub = np.reshape(S[[ind_x, ind_y, ind_x, ind_y], [ind_x, ind_x, ind_y, ind_y]], (2, 2))
# Extract its eigenvalues and eigenvectors, which tell us
# the axes of the ellipse

print S_sub

ellipseInfo = np.linalg.eig(S_sub)
# Eigenvalues are 1/r^2, Eigenvectors are axis directions
axis_1 = ellipseInfo[1][0, :]
r1 = math.sqrt(rho)/math.sqrt(ellipseInfo[0][0])
axis_2 = ellipseInfo[1][1, :]
r2 = math.sqrt(rho)/math.sqrt(ellipseInfo[0][1])
angle = math.atan2(-axis_1[1], axis_1[0])
from matplotlib.patches import Ellipse
ax=plt.gca()
ax.add_patch(Ellipse((xf[ind_x], xf[ind_y]), 
                     2*r1, 2*r2, 
                     angle=angle*180./math.pi, 
                     linewidth=2, fill=False, zorder=2,color='r'));
plt.title("$\\dot{V}(x)$ overlaid with phase diagram f(x) and estimated ROA in $%s-%s$ plane"%(name_x,name_y))
plt.xlabel("$%s$"%name_x)
plt.ylabel("$%s$"%name_y)
#plt.colorbar()
plt.plot(target_state[ind_x],target_state[ind_y],'ro')

# Report an interesting number that is easy to compute
# from the ellipse info
print "Area of your region of attraction: ", math.pi * r1 * r2





plt.show()

