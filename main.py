# Coded by Tharshi Srikannathasan March 2016
# This is the main function for learning

from ANN_Regression import Neural_Network, trainer, computeNumericalGradient;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import random;
import pylab as pl;

# main code

# generate examples

x, y, z, dynamics = Lorentz_RK(0.1, 2.0, 0.3, 4.0, 0.04);

# normalize the data

x1 = (x - np.mean(x))/np.std(x);
x2 = (y - np.mean(y))/np.std(y);
x3 = (z - np.mean(z))/np.std(z);

Y = (z - np.mean(z))/np.std(z);
Y.reshape((100, 1))
print np.shape(Y)


# add errors to the data

sig_err = 0.0;
for i in range(len(x1)):

    x1[i] += sig_err*random.uniform(-1.0, 1.0);
    x2[i] += sig_err*random.uniform(-1.0, 1.0);
    x3[i] += sig_err*random.uniform(-1.0, 1.0);

X = np.column_stack((x1, x2, x3));
print 'X is', np.shape(X);
print 'Y is', np.shape(Y);
# print X;
# print '-------------------------'
# print Y;
# print '-------------------------'

# 1

NN = Neural_Network();

# F = NN.forward(X);

# 2

cost1 = NN.costFunction(X, Y);
dJdW1, dJdW2 = NN.costFunctionPrime(X, Y);

print cost1

# print dJdW1;
# print '-------';
# print dJdW2;

# 3

t = trainer(NN);
t.train(X, Y);
Yhat = NN.forward(X);


pl.figure();
pl.plot(t.J, '-.');

pl.figure();
pl.plot(Y);
pl.plot(Yhat,'-.')
pl.show();