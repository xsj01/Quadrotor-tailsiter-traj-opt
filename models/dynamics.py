# -*- coding: utf8 -*-
import numpy as np
from numpy import exp,cos,sin,deg2rad,pi,sign
import math

import scipy.io as scio

# def sign(x):
#     k=1e6
#     return 2./(1 + exp(-k*x))-1.
#     pass

# def abs(x):
#     return sign(x)*x
 
dataFile = './aero_reg_coeff.mat'
data = scio.loadmat(dataFile)
aero_reg_coeff=data['aero_reg_coeff'][0]
#print aero_reg_coeff[1]

#aero_reg_coeff=[]

def logistic_func(alpha, alpha1, k1, alpha2, k2):

    if k1 < 0 or k2 < 0 or alpha1 >= alpha2:
        print 'Invalid logistic function'
        return 0
        #error('Invalid logistic function');

    P1 =  1./(1 + exp(k1*(alpha - alpha1)));
    P2 = 1./(1 + exp(-k2*(alpha - alpha2)));
    yval = 1 - P1 - P2;

    return yval

def reg_CD( alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;


    comp_B = np.array([cb2, alpha*cb2]).reshape((-1,1))

    comp_N = np.array([cb2*ca3, ca*sb2, ca*cb2*sa2, 
         sa*ca2*cb2, sa*sb2, sa3*cb2,
         abs(sb)*cb*ca2, sa*ca*cb*abs(sb), abs(sb)*cb*sa2,
         ]).reshape((-1,1))

    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);
    comp_B1 = np.array([cb2, (alpha + deg2rad(180))*cb2]).reshape((-1,1))
    reg_B1 = w_B1*comp_B1;

    w_N1 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-10), .5);
    reg_N1 = w_N1*comp_N;

    w_B2 = logistic_func(alpha, deg2rad(-11), 20, deg2rad(18), 10);
    reg_B2 = w_B2*comp_B;

    w_N2 = logistic_func(alpha, deg2rad(18), 10, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);
    comp_B3 = np.array([cb2, (alpha - deg2rad(180))*cb2]).reshape((-1,1))
    reg_B3 = w_B3*comp_B3;

    reg = np.vstack([reg_B1, reg_N1, reg_B2, reg_N2, reg_B3])
    return reg

def get_CD(alpha,beta):
    return reg_CD(alpha,beta).T.dot(aero_reg_coeff[1])
    pass

def reg_CL(alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;

    b1 = -15;
    b2 = 18;
    comp_B = np.array([cb2, alpha*cb2]).reshape((-1,1))

    comp_N = np.array([cb2*ca3, ca*sb2, ca*cb2*sa2, 
     sa*ca2*cb2, sa*sb2, sa3*cb2,
     abs(sb)*cb*ca2, sa*ca*cb*abs(sb), abs(sb)*cb*sa2 ]).reshape((-1,1))

    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);
    comp_B1 = np.array([cb2, (alpha + deg2rad(180))*cb2]).reshape((-1,1))
    reg_B1 = w_B1*comp_B1;

    w_N11 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-20), .5);
    reg_N11 = w_N11*comp_N;

    w_B2 = logistic_func(alpha, deg2rad(-11), 20, deg2rad(18), 10);
    reg_B2 = w_B2*comp_B;

    w_N2 = logistic_func(alpha, deg2rad(18), 10, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);
    comp_B3 = np.array([cb2, (alpha - deg2rad(180))*cb2]).reshape((-1,1))
    reg_B3 = w_B3*comp_B3;

    
    reg = np.vstack([reg_B1, reg_N11, reg_B2, reg_N2, reg_B3])

    return reg

def get_CL(alpha,beta):
    return reg_CL(alpha,beta).T.dot(aero_reg_coeff[0])
    pass

def reg_CY( alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;
    comp_B1 = np.array([sb2, (alpha + deg2rad(180))*sb2]).reshape((-1,1))
    comp_B3 = np.array([sb2, (alpha - deg2rad(180))*sb2]).reshape((-1,1))

    #% Newtonian components
    signb = sign(beta);
    comp_N = np.array([signb*sb2, sb*cb*ca, sb*cb*sa ]).reshape((-1,1))


    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);

    reg_B1 = w_B1*comp_B1;

    w_N11 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-90), 1);
    reg_N11 = w_N11*comp_N;

    w_N12 = logistic_func(alpha, deg2rad(-90), 1, deg2rad(-11), 1);
    reg_N12 = w_N12*comp_N;

    w_N2 = logistic_func(alpha, deg2rad(20), 1, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);

    reg_B3 = w_B3*comp_B3;

    reg = np.vstack([reg_B1, reg_N11, reg_N12, reg_N2, reg_B3])

    return reg

def get_CY(alpha,beta):
    return reg_CY(alpha,beta).T.dot(aero_reg_coeff[2])
    pass

def reg_Cll( alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;

    comp_B = np.array([cb2, alpha*cb2]).reshape((-1,1))

#% % Newtonian components
    signb = sign(beta);
    # comp_N = np.array([signb*cb2*ca2, signb*sb2, signb*cb2*sa2, 
    #  sb*cb*ca, signb*sa*ca*cb2, sb*cb*sa ]).reshape((-1,1))
    comp_N = np.array([signb*sb2, 
     sb*cb*ca, sb*cb*sa ]).reshape((-1,1))

    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);
    comp_B1 = np.array([cb2, (alpha + deg2rad(180))*cb2]).reshape((-1,1))
    reg_B1 = w_B1*comp_B1;

    w_N11 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-90), 5);
    reg_N11 = w_N11*comp_N;

    w_N12 = logistic_func(alpha, deg2rad(-90), 5, deg2rad(-20), .5);
    reg_N12 = w_N12*comp_N;

    w_B2 = logistic_func(alpha, deg2rad(-11), 20, deg2rad(18), 10);
    reg_B2 = w_B2*comp_B;

    w_N22 = logistic_func(alpha, deg2rad(18), 10, deg2rad(90), 5);
    reg_N22 = w_N22*comp_N;

    w_N2 = logistic_func(alpha, deg2rad(90), 5, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);
    comp_B3 = np.array([cb2, (alpha - deg2rad(180))*cb2]).reshape((-1,1))
    reg_B3 = w_B3*comp_B3;

    reg = np.vstack([reg_B1, reg_N11, reg_N12, reg_B2, reg_N2, reg_N22, reg_B3])
    return reg

def get_Cll(alpha,beta):
    return reg_Cll(alpha,beta).T.dot(aero_reg_coeff[3])
    pass

def reg_Cm( alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;
    comp_B = np.array([cb2, alpha*cb2]).reshape((-1,1))

    #% % Newtonian components
    comp_N = np.array([cb2*ca2, sb2, cb2*sa2, 
         abs(sb)*cb*ca, sa*ca*cb2, abs(sb)*cb*sa ]).reshape((-1,1))

    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);
    comp_B1 = np.array([cb2, (alpha + deg2rad(180))*cb2]).reshape((-1,1))
    reg_B1 = w_B1*comp_B1;

    w_N11 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-20), .5);
    reg_N11 = w_N11*comp_N;

    w_B2 = logistic_func(alpha, deg2rad(-11), 20, deg2rad(18), 10);
    reg_B2 = w_B2*comp_B;

    w_N2 = logistic_func(alpha, deg2rad(18), 10, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);
    comp_B3 = np.array([cb2, (alpha - deg2rad(180))*cb2]).reshape((-1,1))
    reg_B3 = w_B3*comp_B3;

    reg = np.vstack([reg_B1, reg_N11, reg_B2, reg_N2, reg_B3])

    return reg

def get_Cm(alpha,beta):
    return reg_Cm(alpha,beta).T.dot(aero_reg_coeff[4])
    pass


def reg_Cn( alpha, beta ):
    sa = sin(alpha);
    ca = cos(alpha);
    sb = sin(beta);
    cb = cos(beta);
    sa2 = sa**2;
    ca2 = ca**2;
    sb2 = sb**2;
    cb2 = cb**2;
    sa3 = sa**3;
    ca3 = ca**3;
    sb3 = sb**3;
    cb3 = cb**3;
    comp_B = np.array([cb2, alpha*cb2]).reshape((-1,1))

    #% Newtonian components
    signb = sign(beta);
    comp_N = np.array([signb*sb2, sb*cb*ca, sb*cb*sa ]).reshape((-1,1))

    #% the first Bernoulli component occurs around -180 degree
    w_B1 = logistic_func(alpha, deg2rad(-190), 1, deg2rad(-170), 1);
    comp_B1 = np.array([cb2, (alpha + deg2rad(180))*cb2]).reshape((-1,1))
    reg_B1 = w_B1*comp_B1;

    #% the first Nowtonian component occurs at the range of [-170, 0] degree
    w_N1 = logistic_func(alpha, deg2rad(-170), 1, deg2rad(-10), 10);
    reg_N1 = w_N1*comp_N;

    #% the second Bernoulli component occurs arond 0 degree
    w_B2 = logistic_func(alpha, deg2rad(-10), 10, deg2rad(18), 10);
    reg_B2 = w_B2*comp_B;

    #% the second Newtonian component occurs at the range of [0, 170] degree
    w_N2 = logistic_func(alpha, deg2rad(0), .5, deg2rad(170), 1);
    reg_N2 = w_N2*comp_N;

    #% the third Bernoulli component occurs around 180 degree
    w_B3 = logistic_func(alpha, deg2rad(170), 1, deg2rad(190), 1);
    comp_B3 = np.array([cb2, (alpha - deg2rad(180))*cb2]).reshape((-1,1))
    reg_B3 = w_B3*comp_B3;

    reg = np.vstack([reg_B1, reg_N1, reg_B2, reg_N2, reg_B3])
    return reg

def get_Cn(alpha,beta):
    return reg_Cn(alpha,beta).T.dot(aero_reg_coeff[5])
    pass


def motion_body_frame(x,u):
    FM=np.array([[1,1,1,1],[-1,1,-1,1],[1,1,-1,-1.],[-1,1,1,-1.]]).dot(u)
    Fx=FM[0]*cos(delta)
    Mx=FM[1]*(sin(delta)*d+kappa*cos(delta))
    My=FM[2]*(cos(delta)*d-kappa*sin(delta))*sin(chi)
    Mz=FM[3]*(cos(delta)*d-kappa*sin(delta))*cos(chi)

    L_aero=0.5*rho*V**2*S*get_CL(alpha,beta)
    D_aero=0.5*rho*V**2*S*get_CD(alpha,beta)
    Y_aero=0.5*rho*V**2*S*get_CY(alpha,beta)

    ll_aero=0.5*rho*V**2*c*S*get_Cll(alpha,beta)
    m_aero=0.5*rho*V**2*c*S*get_Cm(alpha,beta)
    n_aero=0.5*rho*V**2*c*S*get_Cn(alpha,beta)

    F_aero=np.array([[-cos(alpha),0,sin(alpha)],[0,1,0],[-sin(alpha),0,-cos(alpha)]]).dot([D_aero,Y_aero,L_aero])
    pass


class Quadrotor(object):
    """docstring for Quadrotor"""
    def __init__(self):
        super(Quadrotor, self).__init__()
        #self.arg = arg
        self.delta=np.pi/4
        self.rho=1.
        self.S = 0.1955
        self.chi=0.3
        self.d=0.4
        self.kappa=1.
        self.c = 0.24
        self.I_=np.diag([10,1,10])
        self.m_=1.
        self.g_=9.8
        
    def cal_F_M_1(self,state,u):

        roll=state[3]
        pitch=state[4]
        yaw=state[5]

        #calculate V in frame 1
        R_roll=np.array([[1,0,0,],[0,cos(roll),-sin(roll)],[0,sin(roll),cos(roll)]])
        R_pitch=np.array([[cos(pitch),0,sin(pitch)],[0,1,0],[-sin(pitch),0,cos(pitch)]])
        R_yaw=np.array([[cos(yaw),-sin(yaw),0],[sin(yaw),cos(yaw),0],[0,0,1]])

        R_1_to_0=R_yaw.dot(R_pitch).dot(R_roll)
        R_0_to_1=R_1_to_0.T

        v_0=state[6:9]

        v_1=R_0_to_1.dot(v_0)

        V=np.linalg.norm(v_0)


        #calculate V in frame 2 and alpha Beta

        R_1_to_2=np.array([[0,0,1],[cos(self.chi),-sin(self.chi),0],[sin(self.chi),cos(self.chi),0]])
        R_2_to_1=R_1_to_2.T

        v_2=R_1_to_2.dot(v_1)

        alpha=np.arctan2(v_2[2],v_2[0])
        beta=np.arcsin(v_2[0]/V)

        #calculate Force Momentum in Frame 2

        FM_aero=np.array([[1,1,1,1],[-1,1,-1,1],[1,1,-1,-1.],[-1,1,1,-1.]]).dot(u)
        Fx_2=FM_aero[0]*cos(self.delta)
        Mx_2=FM_aero[1]*(sin(self.delta)*self.d+self.kappa*cos(self.delta))
        My_2=FM_aero[2]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*sin(self.chi)
        Mz_2=FM_aero[3]*(cos(self.delta)*self.d-self.kappa*sin(self.delta))*cos(self.chi)
        M_propeller_2=np.array([Mx_2,My_2,Mz_2])
        F_propeller_2=np.array([Fx_2,0,0.])

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

        #F_gravity_0=np.array([0.,0.,-self.m_*self.g_])

        #F_net_0=F_net_0+F_gravity_0

        return F_net_1.reshape((-1,1)),M_net_1.reshape((-1,1))

        #R_2_to_1=



        pass

    def DoCalcTimeDerivatives(self, context, derivatives):
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
        
        
        

        # # Calculate net force and acceleration, expressed in inertial frame
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

#print get_CD(deg2rad(20.54),deg2rad(-21.25))

#print deg2rad(10)

if __name__ == '__main__':

    alpha=0
    beta=0
    delta=0.1
    V=1.
    rho=1.
    S = 0.1955
    chi=0.3
    d=0.4
    kappa=1.
    c = 0.24
    I=np.diag([10,1,10])
    print get_CD(-0.153583417333,0)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  
    #main()
    beta = np.arange(-pi/2,(pi/2 + 0.1),0.1)
    alpha=np.arange(-pi,(pi + 0.1),0.1)
    #print alpha
    yal=np.zeros((alpha.shape[0],beta.shape[0]))

    for i in range(alpha.shape[0]):
        for j in range(beta.shape[0]):
            yal[i,j]=get_CD(alpha[i],beta[j])

    X, Y = np.meshgrid(alpha, beta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print X.shape,Y.shape,yal.shape
    ax.plot_surface(X, Y, yal.T)
    plt.show()

    q=Quadrotor()
    state=np.ones(12)
    u=np.ones(4)

    print q.cal_F_M_1(state,u)

