import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, inputLayer, hiddenLayer, outputLayer, learningRate=0.01, hyperparam=0.01):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.learningRate = learningRate
        self.hyperparam = hyperparam
        self.weights = []
        self.biases = []
        self.netOutputs  = []
        self.layerOutputs = []

    def leakyRELU(self, arr):
        return np.array([i if i>0 else self.hyperparam*i for i in arr])
    
    def NN_init(self):
        
        w1 = np.random.rand(self.hiddenLayer, self.inputLayer)
        w2 = np.random.rand(self.outputLayer, self.hiddenLayer)

        self.weights.append(w1)
        self.weights.append(w2)

        z1 = np.zeros(shape=(self.hiddenLayer))
        z2 = np.zeros(shape=(self.outputLayer))
        self.biases.append(z1)
        self.biases.append(z2)

        


NN = NeuralNetwork(4, 10, 3)
NN.NN_init()