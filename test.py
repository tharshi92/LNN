from ANN_Regression import Neural_Network, trainer#, computeNumericalGradient;
import numpy as np;
from Lorentz_RK import Lorentz_RK;
import pylab as pl;
import random;

# Options #

plotOriginalTarget = 1;
plotFit = 1;
plotCost = 0;

# Create Data #

def targetFunction(x):
    t = np.cos(x)*np.exp(-0.5*x**2.0);
    
    return t;

temp0 = np.linspace(-2*np.pi, 2*np.pi, 400);
temp0 = np.reshape(temp0, (len(temp0), 1));
truth = targetFunction(temp0);
temp0 = (temp0 - np.mean(temp0))/np.std(temp0);
truth = (temp0 - np.mean(temp0))/np.std(temp0);
  
sampleRate = 4;
  
X = np.zeros((len(temp0)/sampleRate, 1));
Y = np.zeros((len(temp0)/sampleRate, 1));

sigmaObs = 0.05;
for i in range(len(X)):
    X[i] = temp0[sampleRate*i];
    signalError = sigmaObs*np.random.uniform(-1, 1);
    Y[i] = truth[sampleRate*i] + signalError;
    
X = (X - np.mean(X))/np.std(X);
Y = (Y - np.mean(Y))/np.std(Y);

dimX = len(X.T);
dimY = len(Y.T);

print 'Number of training examples:', len(X);
print 'Input data X has', dimX, 'dimension(s). ShapeX = ', X.shape;
print 'Output data Y has', dimY, 'dimension(s). ShapeY = ', Y.shape;
print ' '

# Create and Train Neural Network #

layerConfig = [dimX, 3, 3, 3, 3, dimY];
NN = Neural_Network(layerConfig);

T = trainer(NN);
T.gradientDescent(X, Y, 1, 2500);
fit = NN.forward(temp0);

if plotOriginalTarget:
    fig0 = pl.figure();
    pl.plot(temp0, truth, label='truth');
    pl.plot(X, Y, '-.', label='observations');
    pl.grid(1);
    if plotFit:
        pl.plot(temp0, fit, label='fit');
    pl.legend();
    pl.xlabel('x')
    pl.ylabel('y')


if plotCost:
    pl.figure();
    pl.plot(T.J);
    pl.grid(1);
    
print np.linalg.norm(Y - NN.forward(X))

pl.show()

