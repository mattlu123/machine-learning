import numpy as np
from mlplayground.core.layers import Linear

def test_linear_forward_shape():
    layer = Linear(5,3)
    out = layer.forward_pass(np.zeros((2, 5)))
    assert out.shape == (2, 3)

def test_linear_backward_shape():
    layer = Linear(5, 3)
    x = np.random.randn(2, 5)
    out = layer.forward_pass(x)
    dW, db, grad_x = layer.backward_pass(np.ones_like(out))
    assert grad_x.shape == x.shape
    assert dW.shape == layer.W.shape
    assert db.shape == layer.b.shape