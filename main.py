# Coded by Tharshi Srikannathasan March 2016
# This is the main function for learning

from ANN_Regression import Neural_Network, trainer, computeNumericalGradient;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import random;
import pylab as pl

# main code

# generate examples

x, y, z, dynamics = Lorentz_RK(0.1, 2.0, 0.3, 4.0, 0.1);

# normalize the data

x1 = (x - np.mean(x))/np.std(x);
x2 = (y - np.mean(y))/np.std(y);
x3 = (z - np.mean(z))/np.std(z);

# add errors to the data

sig_err = 0.1;
for i in range(len(x1)):

    x1[i] += sig_err*random.uniform(-1.0, 1.0);
    x2[i] += sig_err*random.uniform(-1.0, 1.0);
    x3[i] += sig_err*random.uniform(-1.0, 1.0);

X = np.column_stack(np.transpose((x1, x2, x3)));

# 1

NN = Neural_Network();

F = NN.forward(X);

# 2

cost1 = NN.costFunction(X, dynamics);
dJdW1, dJdW2 = NN.costFunctionPrime(X, dynamics);


print dJdW1;
print '-------';
print dJdW2;

pl.plot(X.T);
pl.plot(F.T,'-.')
pl.show();