# 3 layer Neural network attempts to fit training data
# Coded by Tharshi Srikannathasan tsrikann@physics.utoronto.ca
#
## Version 1 (0216)
##     3 Layer Net with regularization and minimization via python.optimize 'BFGS' method
##     Problems: Fitting functions that have concavity similar to even polynomials
## Version 2 (0316)
##     Fixed scaling issue for fitting curves
## Version 3 (040116)
##     Added Lorentz Model (BFGS Convergence is much faster than gradient descent with momentum)
##     no test data as of yet
##     regularization exists but is not used


import numpy as np
from scipy import optimize
import pylab as pl

class Neural_Network(object):
    def __init__(self, dimX, dimY, m, Lambda=0):    
            
        # Define Hyperparameters
        self.inputLayerSize = dimX
        self.outputLayerSize = dimY
        self.hiddenLayerSize = m
        
        # Weights (parameters)
        self.W1 = 1 * np.array(np.random.randn(self.inputLayerSize,self.hiddenLayerSize))
        self.W2 = 1 * np.array(np.random.randn(self.hiddenLayerSize,self.outputLayerSize))
        
        # Regularization Parameter:
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
        # Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1.0/(1.0 + np.exp(-z));
    
    def sigmoidPrime(self,z):
        # Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.linalg.norm(self.W1)**2.0 + np.linalg.norm(self.W2)**2.0);
        return J
        
    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
        
        return dJdW1, dJdW2
    
    # Helper functions for interacting with other methods/classes
    def getParams(self):
        # Get W1 and W2 Rolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        # Set W1 and W2 using single parameter vector:
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
        e = 1e-5

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            # Return the value we changed to zero:
            perturb[p] = 0
            
        # Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
class trainer(object):
    
    def __init__(self, N):
        # Make Local reference to network:
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
        # Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        # Make empty list to store training/testing costs:
        self.J = []
        self.testJ = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 10000, 'disp' : True}
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
    plot = 1            # turn on plots of cost and fit
    gradTest = 0;       # check if the gradient implementation from NN.py works by checking with a numerical gradient
    regression = 1;     # 1 -> regression, 0 -> simple classification
    
    if regression:
    
        # create data 
    
        # helper functions for scaling between 0 and 1

        def getScaleParams(x):
            
            shift = np.amin(x, axis=0);
            norm = np.amax(x, axis=0) - shift;
            scaleParams = [shift, norm];
            return scaleParams;

        def scale(x, scaleParams):

            return (x - scaleParams[0])/scaleParams[1];

        def unscale(x, scaleParams):
            
            return x*scaleParams[1] + scaleParams[0];

        # import simulation data

        inFile = np.loadtxt('data.txt', skiprows=1);
        data = np.column_stack((inFile[:,0], inFile[:,1], inFile[:,2]));
        signal = inFile[:,5];

        sPX = getScaleParams(data);
        sPY = getScaleParams(signal);

        X = scale(data, sPX);
        y = scale(signal, sPY);
        y = np.reshape(y, (len(y), 1));

        testX = X;
        testY = y;

        # Network Parameters
        dimX = X.shape[1];     # dimension of input data
        dimY = y.shape[1];     # dimenstion of output data
        n = 5;              # number of neurons in hidden layer
        regParam = 0;     # regularization hyperparameter

        # create network
        NN = Neural_Network(dimX, dimY, n, regParam);
        T = trainer(NN);

        # train network
        T.train(X, y, testX, testY);

        # output statistics
        errTrain = sum((y - NN.forward(X))**2);
        errTest = sum((testY - NN.forward(testX))**2);

        print 'Estimated Error of fit (training):', errTrain;
        print 'Estimated Error of fit (testing):', errTest;

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

        pl.figure();
        pl.plot(T.J, label='Training Cost')
        pl.grid(1)
        pl.plot(T.testJ, label='Testing Cost')
        pl.xlabel('iteration')
        pl.title('Cost Functions')
        pl.legend()

            # rerun simulation with trained network

        # define simulation parameters
        T = 8.0;    # simultion length
        h = 1e-3;    # timestep
        N = T/h;   # number of steps

        params = np.array([10.0, 8.0/3.0, 28.0]);
        x0 = np.array([1.508870, -1.531271, 25.46091]);

        x = np.zeros(N);
        y = np.zeros(N);
        z = np.zeros(N);
        dX = np.zeros(N);
        dY = np.zeros(N);
        dZ = np.zeros(N);

        xt = np.zeros(N);
        yt = np.zeros(N);
        zt = np.zeros(N);
        dXt = np.zeros(N);
        dYt = np.zeros(N);
        dZt = np.zeros(N);

        # parameters for the lorentz model
        sigma = params[0];
        beta = params[1];
        rho = params[2];

        # initial conditions
        x[0] = x0[0];
        y[0] = x0[1];
        z[0] = x0[2];
        xt[0] = x0[0];
        yt[0] = x0[1];
        zt[0] = x0[2];

        # define dynamical equation

        def dynamics(x, y, z, sigma, beta, rho):
            
            dxdt = -sigma*(x - y);
            
            dydt = x*(rho - z) - y;
            
            dzdt = x*y - beta*z;
            
            return dxdt, dydt, dzdt;

        # integrate using Runge Kutta Method

        for i in xrange(0, int(N - 1)):

            # update nn simulation

            p1, q1, dummy = dynamics(x[i], y[i], z[i], sigma, beta, rho);
            temp1 = scale(np.column_stack((x[i], y[i], z[i])), sPX);
            r1 = NN.forward(temp1);
            r1 = unscale(r1, sPY);
                             
            dX[i] = p1;
            dY[i] = q1;
            dZ[i] = r1;

            p2, q2, dummy = dynamics(x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0,\
                              sigma, beta, rho);
            temp2 = scale(np.column_stack((x[i] + h*p1/2.0, y[i] + h*q1/2.0, z[i] + h*r1/2.0)), sPX);
            r2 = NN.forward(temp2);
            r2 = unscale(r2, sPY);
                             
            p3, q3, dummy = dynamics(x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0,\
                             sigma, beta, rho);
            temp3 = scale(np.column_stack((x[i] + h*p2/2.0, y[i] + h*q2/2.0, z[i] + h*r2/2.0)), sPX);
            r3 = NN.forward(temp3);
            r3 = unscale(r3, sPY);
                             
            p4, q4, dummy = dynamics(x[i] + h*p3, y[i] + h*q3, z[i] + h*r3,\
                              sigma, beta, rho);
            temp4 = scale(np.column_stack((x[i] + h*p3, y[i] + h*q3, z[i] + h*r3)), sPX);
            r4 = NN.forward(temp4);
            r4 = unscale(r4, sPY);

            x[i+1] = x[i] + h*(p1 + 2.0*p2 + 2.0*p3 + p4)/6.0;
            y[i+1] = y[i] + h*(q1 + 2.0*q2 + 2.0*q3 + q4)/6.0;
            z[i+1] = z[i] + h*(r1 + 2.0*r2 + 2.0*r3 + r4)/6.0;

            # update truth
            
            p1t, q1t, r1t = dynamics(xt[i], yt[i], zt[i], sigma, beta, rho);
            dXt[i] = p1t;
            dYt[i] = q1t;
            dZt[i] = r1t;

            p2t, q2t, r2t = dynamics(xt[i] + h*p1t/2.0, yt[i] + h*q1t/2.0, zt[i] + h*r1t/2.0,\
                                     sigma, beta, rho);
            p3t, q3t, r3t = dynamics(xt[i] + h*p2t/2.0, yt[i] + h*q2t/2.0, zt[i] + h*r2t/2.0,\
                                     sigma, beta, rho);
            p4t, q4t, r4t = dynamics(xt[i] + h*p3t, yt[i] + h*q3t, zt[i] + h*r3t,\
                                     sigma, beta, rho);

            xt[i+1] = xt[i] + h*(p1t + 2.0*p2t + 2.0*p3t + p4t)/6.0;
            yt[i+1] = yt[i] + h*(q1t + 2.0*q2t + 2.0*q3t + q4t)/6.0;
            zt[i+1] = zt[i] + h*(r1t + 2.0*r2t + 2.0*r3t + r4t)/6.0;

        fig1 = pl.figure();
        pl.plot(x, 'g', label='x');
        pl.plot(y, 'r', label='y');
        pl.plot(z, 'k', label='z');
        pl.plot(xt, 'c', label='x truth');
        pl.plot(yt, 'm', label='y truth');
        pl.plot(zt, '0.75', label='z truth');
        pl.legend();
        pl.xlabel("timestep");

        ##fig2 = pl.figure();
        ##ax = fig2.add_subplot(111, projection='3d');
        ##ax.scatter(x, y, z, marker = 'o', color = 'r');
        ##ax.scatter(xt, yt, zt, marker = 'o', color = 'b');
        ##ax.set_xlabel('X');
        ##ax.set_ylabel('Y');
        ##ax.set_zlabel('Z');


        fig3 = pl.figure();
        pl.plot((x - xt), label='x error');
        pl.plot((y - yt), label='y error');
        pl.plot((z - zt), label='z error');
        pl.title('errors');
        pl.legend();
        pl.xlabel("timestep");


        pl.show();
            
    else:
        
        #
        # create data
        #
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float');
        y = np.array([[1], [0], [0], [1]], dtype = 'float');
        
        testX = np.array([[0.1, 0.2], [0.3, 0.8], [0.9, 0.4], [0.7, 0.9]], dtype = 'float');
        testY = np.array([[1], [0], [0], [1]], dtype = 'float');
        
        #
        # Network Parameters
        #
        
        dimX = len(X.T);     # dimension of input data
        dimY = len(y.T);     # dimenstion of output data
        n = 5;              # number of neurons in hidden layer
        regParam = 1e-4;     # regularization hyperparameter
        
        #
        # create network
        #
        
        NN = Neural_Network(dimX, dimY, n, regParam);
        T = trainer(NN);

        # train network
        T.train(X, y, testX, testY);
        
        fit = np.round(NN.forward(X));
        fitTest = np.round(NN.forward(testX));
        
        print fit;
        print '';
        print fitTest;
        
        pl.figure();
        pl.plot(T.J, label='Training Cost')
        pl.grid(1)
        pl.plot(T.testJ, label='Testing Cost')
        pl.xlabel('iteration')
        pl.title('Cost Functions')
        pl.legend()

        pl.show();
        
        
        
