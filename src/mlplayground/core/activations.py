from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    
    @abstractmethod
    def forward_pass(self, z): pass

    @abstractmethod
    def backward_pass(self, grad_out): pass

    # @abstractmethod
    def get_activations(self): pass

class Identity(Activation):
    def __init__(self):
        self.a = None

    def forward_pass(self, z):
        self.a = z
        return z

    def backward_pass(self, grad_out):
        return grad_out

    def get_activations(self):
        return self.a

class ReLU(Activation):

    def __init__(self):
        self.mask = None

    def forward_pass(self, z):
        self.mask = (z > 0).astype(z.dtype)
        return z * self.mask

    def backward_pass(self, grad_out):
        return grad_out * self.mask

    def get_activations(self):
        return self.mask

class Tanh(Activation):

    def __init__(self):
        self.a = None

    def forward_pass(self, z):
        self.a = np.tanh(z)
        return self.a

    def backward_pass(self, grad_out):
        return grad_out * (1 - self.a**2)

    def get_activations(self):
        return self.a

class Sigmoid(Activation):

    def __init__(self):
        self.a = None

    def forward_pass(self, z):
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward_pass(self, grad_out):
        return grad_out * (self.a * (1 - self.a))

    def get_activations(self):
        return self.a