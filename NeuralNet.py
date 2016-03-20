# Feed-Forward Neural Network with Backpropagation
# Coded by Tharshi Sri tsrikann@physics.utoronto.ca
# Inspired by Welch Labs
#
# V1 March 14th 2016:
#           basic feed forward network with gradient descent minimization

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
        
        # Create the weights (using slicing)
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 1, size = (l2, l1 + 1)));
    
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
            self._layerOutput.append(self.sgm(layerInput));
            
        return self._layerOutput[-1].T
        
    #
    # backpropagation method (TRAINING DAY!)
    #
    def trainEpoch(self, input, target, trainingRate = 0.1):
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
                delta.append(outputDelta * self.sgm(self._layerInput[i], True));
            else:
                # Compare to the following delta
                deltaPullback = self.weights[i + 1].T.dot(delta[-1]);
                delta.append(deltaPullback[:-1, :] * self.sgm(self._layerInput[i], True)); #** check this line **#
        
        # Compute weight*deltas
        for i in range(self.layerCount):
            delta_index = self.layerCount - 1 - i;
            
            if i == 0:
                layerOutput = np.vstack([input.T, np.ones([1, numInputs])]);
            else:
                layerOutput = np.vstack([self._layerOutput[i - 1], \
                                np.ones([1, self._layerOutput[i - 1].shape[1]])]);
            
            weightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) * \
                            delta[delta_index][None, :, :].transpose(2, 1, 0), axis=0);
            
            self.weights[i] -= trainingRate * weightDelta;
            
        return error;
                
        
    # Activation function
    def sgm(self, z, derivative = False):
        if not derivative:
            return 1.0/(1.0 + np.exp(-z));
        else:
            output = self.sgm(z);
            return output * (1.0 - output);
            
#
# If run as a script, create a test object
#
if __name__ == '__main__':
    print 'Test Neural Network on y = x^2 for -1 < x < 1'
    bpn = BackPropagationNetwork((1, 3, 3, 1));
    print 'Network Structure:\n{0}\n'.format(bpn.shape);
    # print 'Initial Weights:\n{0}\n'.format(bpn.weights);
    
    lvInput = np.linspace(-1, 1, 50);
    lvTarget = lvInput**2;
    
    maxIter = int(10000) + 1;
    minErr = 1e-5;
    cost = np.zeros(maxIter);
    
    for i in range(maxIter):
        err = bpn.trainEpoch(lvInput, lvTarget, 0.01);
        cost[i] = err;
        if i % 100000 == 0:
            print 'Iteration {0}\tError: {1:0.6f}'.format(i, err);
        if i == maxIter - 1:
            print 'Gradient Descent has reached the maximum number of iterations:'
            print 'Final Error: {0:0.6f}'.format(err)
        if err <= minErr:
            print 'Minimum error reached at iteration {0}'.format(i);
            break
    
    # print 'Final Weights:\n{0}\n'.format(bpn.weights);
    
    # Display
    lvOutput = bpn.forward(lvInput);
    # print 'Input:\n{0}\nOutput:\n{1}\n'.format(lvInput, lvOutput);
    
    fig0 = pl.figure();
    pl.plot(lvInput, lvOutput);
    pl.plot(lvInput, lvTarget, '.-');
    pl.grid(1);
    
    figCost = pl.figure();
    pl.plot(np.log(cost + 1), '.-');
    pl.grid(1);
    pl.title('Cost Functions');
    pl.xlabel('Iteration');
    pl.show();