import numpy as np
from mlplayground.core.network import NeuralNetwork
from mlplayground.core.losses import mse, mse_grad
from mlplayground.core.activations import Sigmoid, ReLU, Tanh
from mlplayground.core.optim import BatchGD, AdamW

optimizer = AdamW(weight_decay=0.01)
X_train = np.random.rand(5, 3)
y = np.random.randn(5).reshape(-1, 1)
model = NeuralNetwork(
    [len(X_train[0]), 3, 1], 
    mse, 
    mse_grad, 
    optimizer,
    ["relu", "sigmoid"]
)