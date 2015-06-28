# author(s): jeroen
# based on an example from https://www.youtube.com/watch?v=S75EdAcXHKk
# sample noisy training points trX, trY around y = 2x, through (0,0)
# and train a gradient to fit a least squares line

import sys, numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

X = ([8,3], [9,4], [5,5])

class NN(object):
    def __init__(self):
        # define Hyperparameters
        self.inputLayerSize = 2
        self.outptLayerSize = 1
        self.hiddenLayerSize = 3

        # define Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outptLayerSize)

    def forward(self, X):
        # Propagate inputs through the network
        self.z2 = np.dot( X, self.W1)
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot( self.a2, self.W2 )
        yHat = sigmoid(self.z3)
        return yHat

    def costFunctionPrime(self, X, y):
        # predict yHat using forward propagation
        self.yHat = self.forward(X)
        # dJ/dW2 = d costfunction / dW2
        # costfunction = 1/2 (y - yHat)**2
        # dJ/dW2 = -(y - yHat) * sigmoid( z3 ) / dW2
        # dJ/dW2 = -(y - yHat) * sigmoidPrime( z3 ) * dz3 / dW2
        # z3 = a2 * W2, z3' = a2
        # dJ/dW2 = a2.T * -(y - yHat) sigmoidPrime(z3)
        delta3 = np.multiply(-(y - self.yHat), sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z) / ((1 + np.exp(-z))**2)


def showSigmoid():
    testValues = np.arange(-5, 5, 0.01)
    plt.plot(testValues, sigmoid(testValues), linewidth=2)
    plt.plot(testValues, sigmoidPrime(testValues), linewidth=2)
    plt.grid(1)
    plt.legend('sigmoid', 'sigmoidPrime')
    plt.draw()
    plt.show()


