import numpy as np
from scipy import optimize

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