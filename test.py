from ANN_Regression import Neural_Network#, trainer, computeNumericalGradient;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import pylab as pl;

X = np.random.rand(2, 3);
Y = np.zeros((2, 1));

for i in range(2):
    for j in range(1):
        Y[i][j] = np.linalg.norm(X[i])

layerSizes = [3, 5, 5, 2, 7, 6, 7, 1];
NN = Neural_Network(layerSizes);

dJdW = NN.costFunctionPrime(X,Y)
print 'done!'
