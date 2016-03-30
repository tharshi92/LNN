# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model
# A Neural Network will be used to estimate dz/dt at every time step
# The Runge-Kutta Method will be compared to the output of the neural network assisted simulation

# import all needed libraries 
import numpy as np;
import pylab as pl;
from mpl_toolkits.mplot3d import Axes3D;
from NeuralNet import BackPropagationNetwork

# helper functions for scaling between 0 and 1

def getScaleParams(x):
    
    shift = np.amin(x, axis=0);
    norm = np.amax(x, axis=0) - shift;
    scaleParams = [shift, norm];
    return scaleParams;

def scale(x, scaleParams):

    return (x - scaleParams[0])/scaleParams[1];

def unscale(x, scaleParams):
    
    return scaled*scaleParams[1] + scaleParams[0];

# import simulation data

inFile = np.loadtxt('data.txt', skiprows=1);
data = np.column_stack((inFile[:,0], inFile[:,1], inFile[:,2]));
signal = inFile[:,5];

sPX = getScaleParams(data);
sPY = getScaleParams(signal);

X = scale(data, sPX);
y = scale(signal, sPY);

# train network!

bpn = BackPropagationNetwork((3, 3, 1));
 
print 'Network Structure:\n{0}\n'.format(bpn.shape);
print 'Training Via Gradient Descent.';

maxIter = int(1e5) + 1;
minErr = 1e-4;
cost = [];

for i in range(maxIter):
    err = bpn.trainEpoch(X, y, 1e0, 1e-3);
    if i % 1e3 == 0:
        cost.append(err);
    if i % 1e4 == 0:
        print 'Iteration {0}\tError: {1:0.6f}'.format(i, err);
    if i == maxIter - 1:
        print 'Gradient Descent has reached the maximum number of iterations:'
        print 'Final Error: {0:0.6f}'.format(err)
    if err <= minErr:
        print 'Minimum error reached at iteration {0}'.format(i);
        break
        
cost = np.array(cost);
yHat = bpn.forward(X);
errTrain = 0.5*np.sum((y - yHat)**2)/len(X);

print 'Final Cost (training):', errTrain;

# rerun simulation with trained network

# define simulation parameters
T = 1.0;    # simultion length
h = 1e-2;    # timestep
N = T/h;   # number of steps

params = np.array([10.0, 8.0/3.0, 28.0]);
x0 = np.array([1.0, 1.0, -1.0]);

x = np.zeros(N);
y = np.zeros(N);
z = np.zeros(N);
dX = np.zeros(N);
dY = np.zeros(N);
dZ = np.zeros(N);

# parameters for the lorentz model
sigma = params[0];
beta = params[1];
rho = params[2];

# initial conditions
x[0] = x0[0];
y[0] = x0[1];
z[0] = x0[2];

# define dynamical equation

def dynamics(x, y, z, sigma, beta, rho):
    
    dxdt = -sigma*(x - y);
    
    dydt = x*(rho - z) - y;
    
    dzdt = x*y - beta*z;
    
    return dxdt, dydt;

# integrate using Runge Kutta Method

for i in xrange(0, int(N - 1)):

    p1, q1 = dynamics(x[i], y[i], z[i], sigma, beta, rho);
    temp1 = scale(np.column_stack((x[i], y[i], z[i])), sPX);
    r1 = bpn.forward(temp1);
                     
    dX[i] = p1;
    dY[i] = q1;
    dZ[i] = r1;

    p2, q2 = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                      sigma, beta, rho);
    temp2 = np.scale(np.column_stack((x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0)), sPX);
    r2 = bpn.forward(temp2);
                     
    p3, q3= dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                     sigma, beta, rho);
    r3 = bpn.forward(np.column_stack((x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0)));
                     
    p4, q4 = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                      sigma, beta, rho);
    r4 = bpn.forward(np.column_stack((x[i] + h*p3, y[i] + h*q3, z[i] + h*r3)));

    x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
    y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
    z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;

##fig0 = pl.figure();
##pl.plot(tempX*xNorm + xShift, tempY*yNorm + yShift, label = 'fit');
##pl.plot(input, target, label = 'data');
##pl.grid(1);
##pl.title('Training Results');

figCost = pl.figure();
pl.plot(np.log(cost**2 + 1));
pl.grid(1);
pl.title('Cost Functions');
pl.xlabel('Iteration');
pl.ylabel('Log(c^2 + 1)');

fig1 = pl.figure();
pl.plot(x, label='x');
pl.plot(y, label='y');
pl.plot(z, label='z');
pl.legend();
pl.xlabel("timestep");

fig2 = pl.figure();
ax = fig2.add_subplot(111, projection='3d');
ax.scatter(x, y, z);

ax.set_xlabel('X');
ax.set_ylabel('Y');
ax.set_zlabel('Z');

pl.show();
