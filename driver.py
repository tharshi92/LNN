# 3 layer Neural network attempts to fit two sets of training data
# Fit is tested with testing data
# Coded by Tharshi Srikannathasan tsrikann@physics.utoronto.ca

import numpy as np
from scipy import optimize
import pylab as pl
from NN import Neural_Network, computeNumericalGradient, trainer

# Options

plot = 1;           # turn on plots of cost, fit, and contour plots
testFunction = 0;   # use the test function and np.linspace() to generate training and data (** PROBLEM **)
exFunction = 1;     # use hand typed data to create training and testing data
gradTest = 0;       # check if the gradient implementation from NN.py works by checking with a numerical                          gradient

# create data (problem spot)

if exFunction:

    X = np.array(([3,5], [5,1], [10,2]), dtype=float);
    y = np.array(([75], [82], [93]), dtype=float);
    
    testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float);
    testY = np.array(([70], [89], [85], [75]), dtype=float);

if testFunction:
    
    X = np.linspace(-10, 10, 6);
    X = np.reshape(X, (len(X), 1));
    y = X*X + np.random.randn(len(X), 1);
    
    testX = np.random.uniform(-10, 10, 10);
    testX = np.reshape(testX, (len(testX), 1));
    testY = testX*testX;

print X.shape, type(X)
print testX.shape, type(testX)
print y.shape, type(y)
print testY.shape, type(testY)
    
# Normalize Data
xScale = np.amax(X, axis=0);
yScale = np.amax(y, axis=0);

X = X/xScale;
y = y/yScale;
testX = testX/xScale;
testY = testY/yScale;

# Network Parameters
dimX = len(X.T);     # dimension of input data
dimY = len(y.T);     # dimenstion of output data
n = 10;              # number of neurons in hidden layer
regParam = 1e-4;     # regularization hyperparameter

# create network
NN = Neural_Network(dimX, dimY, n, regParam);
T = trainer(NN);

# train network
T.train(X, y, testX, testY);

# output statistics
chiSqTrain = sum((y - NN.forward(X))**2);
chiSqTest = sum((testY - NN.forward(testX))**2);

print 'Estimated Chi-Squared Goodness of fit (training):', chiSqTrain;
print 'Estimated Chi-Squared Goodness of fit (testing):', chiSqTest;

if gradTest:
    grad = NN.computeGradients(X, y);
    numGrad = computeNumericalGradient(NN, X, y);
    print ''
    print '-----------------------------------------------------'
    print 'Gradient Checking'
    print 'numGrad: ', numGrad
    print ''
    print 'grad: ', grad
    print ''
    print np.linalg.norm(grad - numGrad)/np.linalg.norm(grad + numGrad), '<- this should be less than 1e-6'
    print '-----------------------------------------------------'
    print ''
if plot:
    pl.figure()
    pl.plot(T.J, label='Training Cost')
    pl.grid(1)
    pl.plot(T.testJ, label='Testing Cost')
    pl.xlabel('iteration')
    pl.title('Cost Functions')
    pl.legend()
    
    if exFunction:
        #Test network for various combinations of sleep/study:
        hoursSleep = np.linspace(0, 10, 100)
        hoursStudy = np.linspace(0, 5, 100)

        #Normalize data (same way training data was normalized)
        in1Norm = hoursSleep/xScale[0];
        in2Norm = hoursStudy/xScale[1];

        #Create 2-d versions of input for plotting
        a, b  = np.meshgrid(in1Norm, in2Norm)

        #Join into a single input matrix:
        allInputs = np.zeros((a.size, 2))
        allInputs[:, 0] = a.ravel()
        allInputs[:, 1] = b.ravel()
    
        allOutputs = NN.forward(allInputs)

        #Contour Plot:
        yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
        xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T
    
        pl.figure();
        CS = pl.contour(xx,yy,yScale*allOutputs.reshape(100, 100))
        pl.clabel(CS, inline=1, fontsize=10)
        pl.xlabel('Hours Sleep')
        pl.ylabel('Hours Study')
        pl.title('Contour Plot of Fit')
    
        #3D plot:

        from mpl_toolkits.mplot3d import Axes3D
        fig = pl.figure()
        ax = fig.gca(projection='3d')

        #Scatter training examples:
        ax.scatter(xScale[0]*X[:,0], xScale[1]*X[:,1], yScale*y, c='b', alpha = 1, s=30)
        ax.scatter(xScale[0]*testX[:,0], xScale[1]*testX[:,1], yScale*testY, c='r', alpha = 1, s=30)

        surf = ax.plot_surface(xx, yy, yScale*allOutputs.reshape(100, 100), \
                               cmap='jet', alpha = 0.5)


        ax.set_xlabel('Hours Sleep')
        ax.set_ylabel('Hours Study')
        ax.set_zlabel('Test Score')
        ax.set_title('Data Fit')


    if testFunction:
        # test network for various combinations
        numPoints = 100;
        x0 = np.linspace(-xScale, xScale, numPoints);
        x0 = np.reshape(x0, (len(x0), 1));
        # normalize data (same way training data was normalized)
        x0 = x0/xScale;
        # forward prop through network with trained weights
        fit = NN.forward(x0);
        # plot
        pl.figure();
        pl.scatter(xScale[0]*X[:, 0], yScale*y, c='b', marker = 'o');
        pl.scatter(xScale[0]*testX[:,0], yScale*testY, c='r', marker = 'o');
        pl.plot(x0*xScale, yScale*fit);
        pl.xlabel('x0');
        pl.ylabel('y');
        pl.title('Fit of Data')
        print ''
    
    pl.show();