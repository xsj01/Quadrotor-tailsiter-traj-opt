# -*- coding: utf8 -*-
import numpy as np
from numpy import exp,cos,sin,deg2rad,pi,sign


import scipy.io as scio


 
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
    return reg_CD(alpha,beta).T.dot(aero_reg_coeff[0])
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
    return reg_CL(alpha,beta).T.dot(aero_reg_coeff[1])
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

#print get_CD(deg2rad(20.54),deg2rad(-21.25))

#print deg2rad(10)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  
    #main()
    beta = np.arange(-pi/2,(pi/2 + 0.1),0.1)
    alpha=np.arange(-pi,(pi + 0.1),0.1)
    #print alpha
    yal=np.zeros((alpha.shape[0],beta.shape[0]))

    for i in range(alpha.shape[0]):
        for j in range(beta.shape[0]):
            yal[i,j]=get_Cn(alpha[i],beta[j])

    X, Y = np.meshgrid(alpha, beta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print X.shape,Y.shape,yal.shape
    ax.plot_surface(X, Y, yal.T)
    plt.show()



