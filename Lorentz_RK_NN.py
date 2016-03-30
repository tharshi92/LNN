# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model
# A Neural Network will be used to estimate dz/dt at every time step
# The Runge-Kutta Method will be compared to the output of the neural network assisted simulation

# import all needed libraries 
import numpy as np;
import pylab as pl;
from mpl_toolkits.mplot3d import Axes3D;
from NeuralNet.py import BackPropagationNetwork

# define simulation parameters
T = 4.0;    # simultion length
h = 1e-2;    # timestep
N = T/h;   # number of steps

x = np.zeros(N);
y = np.zeros(N);
z = np.zeros(N);
dX = np.zeros(N);
dY = np.zeros(N);
dZ = np.zeros(N);

# parameters for the lorentz model
sigma = 10.0;
beta = 8.0/3.0;
rho = 28;

# initial conditions
x[0] = 1.0;
y[0] = 1.0;
z[0] = 0.0;

# define dynamical equation

def dynamics(x, y, z, sigma, beta, rho):
    
    dxdt = -sigma*(x - y);
    
    dydt = x*(rho - z) - y;
    
    dzdt = x*y - beta*z;
    
    return dxdt, dydt, dzdt;


# define derivative functions

def x_prime(sigma, x, y, z):

    return -sigma*(x - y);

def y_prime(rho, x, y, z):

    return x*(rho - z) - y;

def z_prime(beta, x, y, z):

    return x*y - beta*z;

# integrate using Runge Kutta Method

for i in xrange(0, int(N - 1)):

    p1, q1, r1 = dynamics(x[i], y[i], z[i], sigma, beta, rho);
    dX[i] = p1;
    dY[i] = q1;
    dZ[i] = r1;

    p2, q2, r2 = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                          sigma, beta, rho);
    p3, q3, r3 = dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                          sigma, beta, rho);
    p4, q4, r4 = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                          sigma, beta, rho);

    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;

# plot results

pl.figure(1);
pl.plot(x, label='x');
pl.plot(y, label='y');
pl.plot(z, label='z');
pl.legend();
pl.xlabel("timestep");

fig = pl.figure();
ax = fig.add_subplot(111, projection='3d');
ax.scatter(x, y, z);

ax.set_xlabel('X');
ax.set_ylabel('Y');
ax.set_zlabel('Z');

pl.show();
