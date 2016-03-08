# 2 Hidden Layer Neural Netwwork for Regression of Lorentz Systems
# Tharshi Srikannathasan - tsrikann@physics.utoronto.ca
#
# Inspired by the work of Stephen Welch from Welch Labs
#


import numpy as np
from scipy import optimize

class Neural_Network(object):
    
    def __init__(self, layerSizes):        
        # define hyperparameters
        self.layerSizes = layerSizes;
        self.numLayers = len(self.layerSizes);
        
        # initialize weights (parameters)
        
        self.weights = [];
        
        for l in range(self.numLayers - 1):
            self.weights.append(np.random.rand(self.layerSizes[l], self.layerSizes[l + 1]));
            
    def sigmoid(self, z):
        # apply sigmoid activation function to scalar, vector, or matrix
        return 1.0/(1.0 + np.exp(-z));

    def sigmoidPrime(self, z):
        # gradient of sigmoid
        return np.exp(-z)/((1.0 + np.exp(-z))**2.0);
    
    def forward(self, A):
        # propogate inputs though network
        L = self.numLayers;
        self.Cs = [];   # list of linear combinations of nodes
        self.As = [];   # lists of activations
        self.a = A;
        
        for l in range(1, L):
            self.temp1 = np.dot(self.a, self.weights[l - 1]);
            self.Cs.append(self.temp1);
            self.a = self.sigmoid(self.temp1);
            self.As.append(self.a);
            
        bHat = self.a;
        return bHat;   # return output after forward propogation 

    def costFunction(self, A, b):
        # compute cost for given inputs and outputs, using the weights already stored in class
        bHat = self.forward(A);
        J = (1.0/len(A))*0.5*sum(np.linalg.norm(b - bHat, axis=1)**2.0);
        return J;

    def costFunctionPrime(self, A, b):
        # compute derivative with respect to weights given the data
        L = self.numLayers;
        self.bHat = self.forward(A);
        self.dJdWs = [];
        
        delta = -(b - self.bHat)*self.sigmoidPrime(self.Cs[L - 2]);
        for i in reversed(range(1, L-1)):
            self.dJdWs.append(np.dot(self.As[i].T, delta));
            delta = np.dot(delta, self.weights[i].T)*self.sigmoidPrime(self.Cs[i - 1]);
        
        return self.dJdWs;
        

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
        print x0
        print Jout
        self.N.setParams(x0);
        self.optimizationResults = x0;
        
    def gradientDescent(self, A, b, alpha, maxiter):
        
        self.A = A;
        self.b = b;
        self.alpha = alpha;
        self.maxiter = maxiter;
        self.iter = 0;
        
        
