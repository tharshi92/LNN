# Feed-Forward Neural Network with Backpropagation
# Coded by Tharshi Sri tsrikann@physics.utoronto.ca
# Inspired by Welch Labs and YouTube's Ryan Harris
#
# V1 March 14th 2016:
#     basic feed forward network with gradient descent (rate defaults to 0.1) minimization
#
# V2 March 22nd 2016:
#     added momentum (defaults to 0.5) and fixed scaling issues
#
# V3 March 23rd 2016:
#     added classification example (XOR)

#
# imports
#

import numpy as np;
import pylab as pl;

class BackPropagationNetwork:
    '''A back propagation network'''
    
    #
    # Class methods
    #
    def __init__(self, layerSize):
        '''Initialize the network'''
        
        # Layer info
        self.weights = [];
        self.layerCount = len(layerSize) - 1;
        self.shape = layerSize;
        
        # Input/Output data from last run
        self._layerInput = [];
        self._layerOutput = [];
        self._previousWeightDelta = [];
        
        # Create the weights (using slicing)
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 1.0, size = (l2, l1 + 1)));
            self._previousWeightDelta.append(np.zeros((l2, l1 +1)));
    
    #
    # Forward propagation method (Run it!)
    #
    def forward(self, input):
        '''Forward propagation given an input'''
        
        numInputs = input.shape[0];
        
        # Clear out the previous arrays
        self._layerInput = [];
        self._layerOutput = [];
        
        # Run it! (forward propagation)
        for i in range(self.layerCount):
            if i == 0:
                layerInput = self.weights[0].dot\
                                (np.vstack([input.T, np.ones([1, numInputs])]));
            else:
                layerInput = self.weights[i].dot\
                                (np.vstack([self._layerOutput[-1], np.ones([1, numInputs])]));
                
            self._layerInput.append(layerInput);
            self._layerOutput.append(self.act(layerInput));
            
        return self._layerOutput[-1].T
        
    #
    # backpropagation method (TRAINING DAY!)
    #
    def trainEpoch(self, input, target, trainingRate = 0.1, momentum = 0.5):
        '''This method trains the network for one epoch'''
        
        delta = [];
        numInputs = input.shape[0];
        
        # First run the network forward
        self.forward(input);
        
        # Calculate deltas
        for i in reversed(range(self.layerCount)):
            if i == self.layerCount - 1:
                # Compare to the target values
                outputDelta = self._layerOutput[i] - target.T;
                error = np.sum(outputDelta**2);
                delta.append(outputDelta * self.act(self._layerInput[i], True));
            else:
                # Compare to the following delta
                deltaPullback = self.weights[i + 1].T.dot(delta[-1]);
                delta.append(deltaPullback[:-1, :] * self.act(self._layerInput[i], True)); #** check this line **#
        
        # Compute weight*deltas
        for i in range(self.layerCount):
            delta_index = self.layerCount - 1 - i;
            
            if i == 0:
                layerOutput = np.vstack([input.T, np.ones([1, numInputs])]);
            else:
                layerOutput = np.vstack([self._layerOutput[i - 1], \
                                np.ones([1, self._layerOutput[i - 1].shape[1]])]);
            
            curWeightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) * \
                            delta[delta_index][None, :, :].transpose(2, 1, 0), axis=0);
                            
            weightDelta = trainingRate*curWeightDelta + momentum*self._previousWeightDelta[i];
            
            self.weights[i] -= weightDelta;
            self._previousWeightDelta[i] = weightDelta;
            
        return error;
                
        
    # Activation function
    def act(self, z, derivative = False):
        if not derivative:
            return 1.0/(1.0 + np.exp(-z));
        else:
            output = self.act(z);
            return output * (1.0 - output);
            
#
# If run as a script, create a test object
#
if __name__ == '__main__':
    
    bpn = BackPropagationNetwork((1, 10, 1));
     
    print 'Network Structure:\n{0}\n'.format(bpn.shape);
    print 'Training Via Gradient Descent.'
    
    # print 'Initial Weights:\n{0}\n'.format(bpn.weights);
    
    start = 0;
    end = 2*np.pi;
    input = np.linspace(start, end, 200);
    target = np.cos(np.pi*input) - 1 + np.cos(4*input);
    
    xShift = np.min(input)
    xNorm = np.max(input) - xShift;
    
    yShift = np.min(target);
    yNorm = np.max(target) - yShift;
    
    X = (input - xShift)/xNorm;
    y = (target - yShift)/yNorm;
    
    xTest = np.random.uniform(start, end, 50);
    yTest = np.cos(np.pi*xTest) - 1 + np.cos(4*xTest);
    
    xTest = (xTest - xShift)/xNorm;
    yTest = (yTest - yShift)/yNorm;
    
    maxIter = int(1e6) + 1;
    minErr = 1e-4;
    cost = [];
    
    for i in range(maxIter):
        err = bpn.trainEpoch(X, y, 1e-1, 1e-2);
        if i % 1e3 == 0:
            cost.append(err);
        if i % 1e3 == 0:
            print 'Iteration {0}\tError: {1:0.6f}'.format(i, err);
        if i == maxIter - 1:
            print 'Gradient Descent has reached the maximum number of iterations:'
            print 'Final Error: {0:0.6f}'.format(err)
        if err <= minErr:
            print 'Minimum error reached at iteration {0}'.format(i);
            break
            
    cost = np.array(cost);
    
    # Display Results
    tempX = np.linspace(1.5*start, 1.5*end, 1e3);
    tempX = (tempX - xShift)/xNorm;
    
    tempY = bpn.forward(tempX);
    yHat = bpn.forward(X);
    yHatTest = bpn.forward(xTest);
    
    errTest = 0.5*np.sum((yTest - yHatTest)**2)/len(xTest);
    errTrain = 0.5*np.sum((y - yHat)**2)/len(X);
    
    print 'Final Cost (training):', errTrain;
    print 'Final Cost (testing):', errTest;
    
    fig0 = pl.figure();
    pl.plot(tempX*xNorm + xShift, tempY*yNorm + yShift, label = 'fit');
    pl.plot(input, target, label = 'data');
    pl.grid(1);
    pl.title('Training Results');
    
    figCost = pl.figure();
    pl.plot(np.log(cost**2 + 1));
    pl.grid(1);
    pl.title('Cost Functions');
    pl.xlabel('Iteration');
    pl.ylabel('Log(c^2 + 1)');
    
    # figTest = pl.figure();
    # pl.plot(xTest*xNorm + xShift, yTest*yNorm + yShift, 'o');
    # pl.plot(xTest*xNorm + xShift, yHatTest*yNorm + yShift, 'o');
    # pl.title('Testing Results');
    # pl.grid(1);
    
    pl.show();
