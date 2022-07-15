import numpy as np

np.random.seed(0)

X = np.array([[1.0,2.0,3.0,2.5],
     [2.0,5.0,-1.0,-2.0],
     [-1.5,2.7,3.3,-0.8]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self,  inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)  
        self.dinputs = np.dot(dvalues, self.weights.T)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)

layer2.forward(layer1.output)
print(layer2.output)