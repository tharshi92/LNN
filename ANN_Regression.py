# Neural Networks Demystified
# Part 6: Training
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch


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
        
        return reversed(self.dJdWs);
        
#
#     #Helper Functions for interacting with other classes:
#     def getParams(self):
#         #Get W1 and W2 unrolled into vector:
#         params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
#         return params
#
#     def setParams(self, params):
#         #Set W1 and W2 using single paramater vector.
#         W1_start = 0
#         W1_end = self.hiddenLayerSize * self.inputLayerSize
#         self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
#         W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
#         self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
#
#     def computeGradients(self, X, y):
#         dJdW1, dJdW2 = self.costFunctionPrime(X, y)
#         return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
#
# class trainer(object):
#     def __init__(self, N):
#         #Make Local reference to network:
#         self.N = N
#
#     def callbackF(self, params):
#         self.N.setParams(params)
#         self.J.append(self.N.costFunction(self.X, self.y))
#
#     def costFunctionWrapper(self, params, X, y):
#         self.N.setParams(params)
#         cost = self.N.costFunction(X, y)
#         grad = self.N.computeGradients(X,y)
#         return cost, grad
#
#     def train(self, X, y):
#         #Make an internal variable for the callback function:
#         self.X = X
#         self.y = y
#
#         #Make empty list to store costs:
#         self.J = []
#
#         params0 = self.N.getParams()
#
#         options = {'maxiter': 200, 'disp' : True}
#         _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
#                                  args=(X, y), options=options, callback=self.callbackF)
#
#         self.N.setParams(_res.x)
#         self.optimizationResults = _res
#
