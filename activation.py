from DenseLayer import Layer_Dense
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)


if __name__ == '__main__':
    print(activation1.output)
