# 3 layer Neural network attempts to fit two sets of training data
# Fit is tested with testing data
# Coded by Tharshi Srikannathasan tsrikann@physics.utoronto.ca
#
# Version 1 (0216)
#    3 Layer Net with regularization and minimization via python.optimize 'BFGS' method
#        Problems: Fitting functions that have concavity similar to even polynomials
        

import numpy as np
from scipy import optimize
import pylab as pl

class Neural_Network(object):
    def __init__(self, dimX, dimY, m, Lambda=0):        
        #Define Hyperparameters
        self.inputLayerSize = dimX
        self.outputLayerSize = dimY
        self.hiddenLayerSize = m
        
        #Weights (parameters)
        self.W1 = np.array(np.random.randn(self.inputLayerSize,self.hiddenLayerSize))
        self.W2 = np.array(np.random.randn(self.hiddenLayerSize,self.outputLayerSize))
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
        print '';
        print '--------------------------------------------------------'
        print 'INITIALIZING 3 LAYER NEURAL NETWORK...'
        print ''
        print 'INPUT DATA IS OF DIMENSION', dimX
        print 'OUTPUT DATA IS OF DIMENSION', dimY
        print 'THE HIDDEN LAYER CONTAINS', m, 'NEURONS'
        print 'GAUSSIAN REGULARIZATION PARAMATER', Lambda
        print '--------------------------------------------------------'
        print ''
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1.0/(1.0 + np.exp(-z));
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.linalg.norm(self.W1)**2.0 + np.linalg.norm(self.W2)**2.0);
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:
        W1_start = 0
        W1_end = self.hiddenLayerSize*self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], \
                             (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], \
                             (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
        
def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
        self.testJ.append(self.N.costFunction(self.testX, self.testY))
        
    def costWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        return cost
        
    def costGradWrapper(self, params, X, y):
        self.N.setParams(params)
        grad = self.N.computeGradients(X,y)
        return grad
        
    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costWrapper, params0, jac=self.costGradWrapper, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
        print ''
        
    def gradientDescent(self, X, y, testX, testY, alpha, maxiter):
    
        print 'Training Network using the Method of Gradient Descent';
    
        self.X = X;
        self.y = y;
        self.testX = testX;
        self.testY = testY;
        self.alpha = alpha;
        self.maxiter = maxiter;
        self.iter = 0;
        self.J = [];
        self.testJ = [];
        self.params = self.N.getParams();
    
        for i in range(self.maxiter):
            self.callbackF(self.params);
            grads = self.N.computeGradients(self.X, self.y);
            self.params -= alpha*grads;
            self.iter += 1;
        
        self.N.setParams(self.params);
        
        print 'Training Complete:';
        print 'Value of cost = ', self.J[maxiter - 1];
        print 'cost gradient = ', grads;
        print 'Final weights:', self.N.W1, self.N.W2;
        
if __name__ == '__main__':
    
    # Options

    plot = 1;           # turn on plots of cost, fit, and contour plots
    testFunction = 1;   # use the test function and np.linspace() to generate training and data (** PROBLEM **)
    exFunction = 0;     # use hand typed data to create training and testing data
    gradTest = 0;       # check if the gradient implementation from NN.py works by checking with a numerical                          gradient

    # create data (problem spot)

    if exFunction:

        X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float);
        y = np.array(([75], [82], [93]), dtype=float);
    
        testX = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float);
        testY = np.array(([70], [89], [85], [75]), dtype=float);

    if testFunction:
    
        sigErr = 0.04;
    
        X = np.linspace(-1, 1, 20);
        X = np.reshape(X, (len(X), 1));
        y = X + sigErr*np.random.randn(len(X), 1);
    
        testX = np.random.uniform(-1, 1, 10);
        testX = np.reshape(testX, (len(testX), 1));
        testY = testX + sigErr*np.random.randn(len(testX), 1);

    # print X.shape, type(X)
    # print testX.shape, type(testX)
    # print y.shape, type(y)
    # print testY.shape, type(testY)
    
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
    n = 300;              # number of neurons in hidden layer
    regParam = 1e-5;     # regularization hyperparameter

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
            numPoints = 100;
            #Test network for various combinations of sleep/study:
            hoursSleep = np.linspace(-xScale[0], xScale[0], numPoints)
            hoursStudy = np.linspace(-xScale[1], xScale[1], numPoints)

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
            yy = np.dot(hoursStudy.reshape(numPoints, 1), np.ones((1, numPoints)))
            xx = np.dot(hoursSleep.reshape(numPoints, 1), np.ones((1, numPoints))).T
    
            pl.figure();
            CS = pl.contour(xx,yy,yScale*allOutputs.reshape(numPoints, numPoints))
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

            surf = ax.plot_surface(xx, yy, yScale*allOutputs.reshape(numPoints, numPoints), \
                                   cmap='jet', alpha = 0.5)

            ax.set_xlabel('Hours Sleep')
            ax.set_ylabel('Hours Study')
            ax.set_zlabel('Test Score')
            ax.set_title('Data Fit')


        if testFunction:
            # test network for various combinations
            numPoints = 100;
            x0 = np.linspace(-xScale[0], xScale[0], numPoints);
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