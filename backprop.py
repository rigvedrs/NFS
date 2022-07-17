from activation import Activation_ReLU
from DenseLayer import Layer_Dense
from softmax import Activation_Softmax
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Categorical_Cross_Entropy_Loss import Loss_CategoricalCrossEntropy
from Categorical_Cross_Entropy_Loss import Activation_Softmax_Loss_CategoricalCrossEntropy


nnfs.init()

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])
print('loss', loss)

predictions = np.argmax(loss_activation.output, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print('acc:', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
