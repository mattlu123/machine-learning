import numpy as np
from .layers import Linear
from .optim import BatchGD, AdamW
from .activations import Sigmoid, ReLU, Tanh, Identity

class NeuralNetwork():
    
    def __init__(self, layer_dims, loss, loss_grad, optimizer, activations):
        self.layer_dims = layer_dims
        self.optimizer = optimizer
        self.loss = loss
        self.loss_grad = loss_grad
        self.layers = []

        for i in range(len(layer_dims) - 1):
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1]))

        self.activations = self.make_activation(self, layer_dims=layer_dims, activations=activations)

    @staticmethod
    def make_activation(self, layer_dims, activations):
        n_layers = len(layer_dims) - 1
        if isinstance(activations, str):
            acts = [activations] * n_layers
        elif isinstance(activations, list):
            if len(activations) != n_layers:
                raise ValueError(f"Need {n_layers} activations, got {len(activations)}")
            acts = list(activations)
        else:
            acts = [activations] * n_layers

        act_map = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "identity": Identity}

        out = []
        for i, act in enumerate(acts):
            if isinstance(act, str):
                out.append(act_map.get(act.lower(), Sigmoid)())
            elif isinstance(act, type):
                out.append(act())
            elif hasattr(act, "forward_pass") and hasattr(act, "backward_pass"):
                out.append(act)
            else:
                raise TypeError(f"Invalid activation for layer {i}: {act!r}")
        
        return out

    def forward(self, X_train):
        output = X_train
        for j, layer in enumerate(self.layers):
            z = layer.forward_pass(output)
            a = self.activations[j]
            output = a.forward_pass(z)

        return output

    def backward(self, grad_curr):
        param_grads = []
        for layer, act in zip(reversed(self.layers), reversed(self.activations)):
            grad_curr = act.backward_pass(grad_curr)
            dw, db, grad_curr = layer.backward_pass(grad_curr)
            param_grads.append((layer.W, dw))
            param_grads.append((layer.b, db))
        
        self.optimizer.step(param_grads)
        return grad_curr

    def train(self, epochs, X_train, y, batch_size=None):
        
        # mini-batch stuff here
        n = X_train.shape[0]
        batch_size = batch_size if batch_size else n

        # main loop
        for i in range(epochs):
            for s in range(0, n, batch_size):
                X_b, y_b = X_train[s: s+batch_size], y[s: s+batch_size]
                
                # forward pass
                output = self.forward(X_b)

                # evaluate loss
                loss = self.loss(output, y_b)
                loss_grad = self.loss_grad(output, y_b)
                for act in self.activations:
                    print(act.get_activations)

                # backprop
                self.backward(loss_grad)

        print("Training Completed")

    def pred(self, X):
        return self.forward(X)


    