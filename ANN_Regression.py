# We are attempting to use a neural network to learn the dynamical behaviour of one out of three lorentz
#   components. No regularization, no ensemble, for this run

# Coded by Tharshi Srikannathasan March 2016, Inspired by WelchLabs!

# import modules
import numpy as np
from scipy import optimize
from Lorentz_RK import Lorentz_RK
from Lorentz import Lorentz
import random

class Neural_Network(object):
    def __init__(self):        
        # define Hyperparameters
        self.inputLayerSize = 3;
        self.outputLayerSize = 3;
        self.hiddenLayerSize = 4;
        
        # weights (parameters)
        self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize);
        self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize);
        
    def forward(self, X):
        # propogate inputs though network
        self.z2 = np.dot(self.W1, X);
        self.a2 = self.sigmoid(self.z2);
        self.z3 = np.dot(self.W2, self.a2);
        yHat = self.sigmoid(self.z3);
        return yHat;
        
    def sigmoid(self, z):
        # sigmoid activation function
        return 1.0/(1.0 + np.exp(-z));
    
    def sigmoidPrime(self,z):
        # derivative of sigmoid function
        return np.exp(-z)/((1.0 + np.exp(-z))**2.0);
    
    def costFunction(self, X, y):
        # cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X);
        J = 0.5*sum((y - self.yHat)**2);
        return J;
        
    def costFunctionPrime(self, X, y):
        # derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X);
        
        delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3));
        dJdW2 = np.dot(delta3, self.a2.T);
        
        delta2 = np.dot(self.W2.T, delta3)*self.sigmoidPrime(self.z2);
        dJdW1 = np.dot(delta2, X.T);  
        
        return dJdW1, dJdW2;
    
    # helper functions:
    def getParams(self):
        # get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()));
        return params;
    
    def setParams(self, params):
        # set W1 and W2 using single paramater vector.
        W1_start = 0;
        W1_end = self.hiddenLayerSize * self.inputLayerSize;
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize));
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize;
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
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
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
