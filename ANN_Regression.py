# 2 Hidden Layer Neural Netwwork for Regression of Lorentz Systems
# Tharshi Srikannathasan - tsrikann@physics.utoronto.ca
#
# Inspired by the work of Stephen Welch from Welch Labs
#

import numpy as np
from scipy import optimize

class Neural_Network(object):
    
    def __init__(self, layerConfig):        
        # define hyperparameters
        self.layerSizes = layerConfig;
        self.numLayers = len(self.layerSizes);
        L = self.numLayers;
        
        # initialize weights (parameters)
        
        self.weights = [];
        
        for l in range(L - 1):
            self.weights.append(np.random.rand(self.layerSizes[l], self.layerSizes[l + 1]));
            
        print 'Initializing Neural Network...';
        print 'There are ', self.numLayers, ' layers in the network.';
        print "Initial weights: ", self.weights;
        print 'The shapes of the weights are:'
        
        for l in range(self.numLayers - 1):
            print self.weights[l].shape
            
        print 'Inital setup is complete.'
        print ''
            
    def sigmoid(self, z):
        # apply sigmoid activation function to scalar, vector, or matrix
        return 1.0/(1.0 + np.exp(-z));

    def sigmoidPrime(self, z):
        # gradient of sigmoid
        sig = self.sigmoid(z);
        return sig*(1.0 - sig);
    
    def forward(self, A):
        # propogate inputs though network
        L = self.numLayers;
        self.Cs = [];   # list of linear combinations of nodes
        self.As = [];   # lists of activations
        self.a = A;
        self.As.append(self.a);

        for l in range(L - 1):
            temp1 = np.dot(self.a, self.weights[l]);
            self.a = self.sigmoid(temp1);
            self.Cs.append(temp1);
            self.As.append(self.a);
            
        bHat = self.a;
        return bHat;   # return estimate after forward propogation 

    def costFunction(self, A, b):
        # compute cost for given inputs and outputs, using the weights already stored in class
        bHat = self.forward(A);
        J = (1.0/len(A))*0.5*sum(np.linalg.norm(b - bHat, axis=1)**2.0);
        return J;

    def costFunctionPrime(self, A, b):
        # compute derivative with respect to weights given the data
        L = self.numLayers;
        bHat = self.forward(A);
        self.dJdWs = [];
        
        delta = -(b - bHat)*self.sigmoidPrime(self.Cs[L - 2]);
        self.dJdWs.append(np.dot(self.As[L - 2].T, delta));
        for l in reversed(range(L - 2)):
            delta = np.dot(delta, self.weights[l + 1].T)*self.sigmoidPrime(self.Cs[l]);
            self.dJdWs.append(np.dot(self.As[l].T, delta));

        return list(reversed(self.dJdWs));
        
    # helper functions for interacting with other classes
    def getParams(self):
        # unroll all weight matrices into vectors and combine them
        params = self.weights[0].ravel();
        
        for k in range(1, len(self.weights)):
            params = np.concatenate((params, self.weights[k].ravel()));
            
        return params;

    def setParams(self, params):
        # set weights using the single vector of parameters from getParams function
        startIndex = 0;
        endIndex = 0;
        
        for k in range(self.numLayers - 1):
            endIndex += self.layerSizes[k] * self.layerSizes[k + 1];
            self.weights[k] = np.reshape(params[startIndex:endIndex], (self.layerSizes[k], self.layerSizes[k + 1]));
            startIndex = endIndex;
  
    def computeGradients(self, A, b):
        dJdWs = self.costFunctionPrime(A, b);
        temp2 = dJdWs[0].ravel();
        
        for k in range(1, len(dJdWs)):
            temp2 = np.concatenate((temp2, dJdWs[k].ravel()));
            
        return temp2;

class trainer(object):
    
    def __init__(self, N):
        # make Local reference to network
        self.N = N;
        
        # make empty list to store costs
        self.J = [];

    def callbackF(self, params):
        self.N.setParams(params);
        self.J.append(self.N.costFunction(self.A, self.b));

    def costWrapper(self, params, A, b):
        self.N.setParams(params);
        cost = self.N.costFunction(A, b);
        return cost
        
    def costGradWrapper(self, params, A, b):
        self.N.setParams(params);
        grad = self.N.computeGradients(A, b);
        return grad
        
    def train(self, A, b):
        # make internal variables for the callback function
        self.A = A;
        self.b = b;
        
        params0 = self.N.getParams();
        options = {'disp': True};
        
        #_res = optimize.minimize(self.costWrapper, params0, method='BFGS', jac=self.costGradWrapper, args=(A, b), callback=self.callbackF, options=options);
        x0, fval, grid, Jout = optimize.brute(self.costWrapper, bounds, args=(A, b), Ns=100, full_output=1, disp=True);
        self.N.setParams(x0);
        self.optimizationResults = x0;
        
    def gradientDescent(self, A, b, alpha, maxiter):
        
        print 'Training Network using the Method of Gradient Descent';
        
        self.A = A;
        self.b = b;
        self.alpha = alpha;
        self.maxiter = maxiter;
        self.iter = 0;
        self.params = self.N.getParams();
        
        for i in range(self.maxiter):
            self.callbackF(self.params);
            grads = self.N.computeGradients(self.A, self.b);
            self.params -= alpha*grads;
            self.iter += 1;
            
        self.N.setParams(self.params);
            
        print 'Training Complete:';
        print 'Value of cost = ', self.J[maxiter - 1];
        print 'Norm of cost gradient = ', np.linalg.norm(grads);
        print 'Final weights:', self.N.weights; 

def computeNumericalGradient(N, X, y):
    
        params1 = N.getParams();
        numGrad = np.zeros(params1.shape);
        perturb = np.zeros(params1.shape);
        e = 1e-2;

        for p in range(len(params1)):
            # set perturbation vector
            perturb[p] = e;
            N.setParams(params1 + perturb);
            loss2 = N.costFunction(X, y);
            
            N.setParams(params1 - perturb);
            loss1 = N.costFunction(X, y);

            # compute numerical gradient
            numGrad[p] = (loss2 - loss1) / (2*e);

            # return the value we changed to zero:
            perturb[p] = 0;
            
        # return params to original value:
        N.setParams(params1)

        return numGrad 
        
        
        
