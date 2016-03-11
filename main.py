# Coded by Tharshi Srikannathasan March 2016
# This is the main function for learning

from NN import Neural_Network, trainer, computeNumericalGradient, Lorentz_RK_NN;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import random;
import pylab as pl;

# main code

gradCheck = 0;
costPlot = 1;

# generate examples

x, y, z, dynamics = Lorentz_RK(0.1, 2.0, 0.3, 4.0, 0.01);
Xt = np.column_stack((x, y, z));

dxdt, dydt, dzdt = dynamics;

# normalize the data
mu = max(np.max(x), np.max(y), np.max(z));
x1 = x/mu
x2 = y/mu
x3 = z/mu

Y = dzdt/mu
Y = Y.reshape((len(Y), 1))

# add errors to the data

sig_err = 0.1;
for i in range(len(x1)):

    x1[i] += sig_err*random.uniform(-1.0, 1.0);
    x2[i] += sig_err*random.uniform(-1.0, 1.0);
    x3[i] += sig_err*random.uniform(-1.0, 1.0);

X = np.column_stack((x1, x2, x3));

# create and train network

NN = Neural_Network();

if gradCheck:
    grad = NN.computeGradients(X, Y);
    numGrad = computeNumericalGradient(NN, X, Y);
    print 'Gradient Checking!'
    print 'numGrad: ', numGrad
    print ''
    print 'grad: ', grad
    print ''
    print np.linalg.norm(grad - numGrad)/np.linalg.norm(grad + numGrad)
    print ''

t = trainer(NN);
t.gradientDescent(X, Y, 0.1, 4000);

if costPlot:
    pl.figure();
    pl.plot(t.J, '-.');

# results
xf, yf, zf = Lorentz_RK_NN(0.1, 2.0, 0.3, 4.0, 0.01, NN);
Xf = np.column_stack((xf, yf, zf));

pl.figure()
pl.plot(Xt)

pl.figure()
pl.plot(Xf)

pl.figure()
pl.plot(Y)
pl.plot(NN.forward(X))

pl.show()