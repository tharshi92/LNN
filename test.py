from ANN_Regression import Neural_Network#, trainer, computeNumericalGradient;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import pylab as pl;

dataX = np.random.rand(10, 3);
dataY = np.linalg.norm(dataX, axis=1);

layerSizes = [3, 4, 4, 1];
NN = Neural_Network(layerSizes);
