# Coded by Tharshi Srikannathasan March 2016
# This is a 4th order Runge-Kutta Scheme to Solve the Lorentz Model
# A Neural Network will be used to estimate dz/dt at every time step
# The Runge-Kutta Method will be compared to the output of the neural network assisted simulation

# import all needed libraries 
import numpy as np;
import pylab as pl;
from mpl_toolkits.mplot3d import Axes3D;
from NeuralNet import BackPropagationNetwork

# import simulation data

# train network!

bpn = BackPropagationNetwork((3, 3, 1));
 
print 'Network Structure:\n{0}\n'.format(bpn.shape);
print 'Training Via Gradient Descent.';

data = np.column_stack((x, y, z));
signal = dZ;
xShift = np.amin(data, axis=0);
xNorm = np.amax(data, axis=0) - xShift;

yShift = np.amin(signal, axis=0);
yNorm = np.amax(signal, axis=0) - yShift;

X = (data - xShift)/xNorm;
Y = (signal - yShift)/yNorm;

maxIter = int(1e4) + 1;
minErr = 1e-4;
cost = [];

for i in range(maxIter):
    err = bpn.trainEpoch(X, Y, 1e-1, 0);
    if i % 1e3 == 0:
        cost.append(err);
    if i % 1e3 == 0:
        print 'Iteration {0}\tError: {1:0.6f}'.format(i, err);
    if i == maxIter - 1:
        print 'Gradient Descent has reached the maximum number of iterations:'
        print 'Final Error: {0:0.6f}'.format(err)
    if err <= minErr:
        print 'Minimum error reached at iteration {0}'.format(i);
        break
        
cost = np.array(cost);
yHat = bpn.forward(X);
errTrain = 0.5*np.sum((Y - yHat)**2)/len(X);

print 'Final Cost (training):', errTrain;

# rerun simulation with trained network

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
