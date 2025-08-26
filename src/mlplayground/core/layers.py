import numpy as np

# Perceptron class
class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)
        self.b = np.zeros(out_dim)
        self.X = None

    # Forward pass
    def forward_pass(self, X):
        
        self.X = X
        # construct pre-activations
        pre_activation = self.X @ self.W + self.b

        return pre_activation

    # Backward pass
    def backward_pass(self, grad_out):
        dw = self.X.T @ grad_out
        db = grad_out.sum(axis=0)
        grad_curr = grad_out @ self.W.T
        return dw, db, grad_curr