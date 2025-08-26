import numpy as np
from mlplayground.core.network import NeuralNetwork
from mlplayground.core.losses import mse, mse_grad
from mlplayground.core.activations import Sigmoid, ReLU, Tanh, Identity
from mlplayground.core.optim import BatchGD, AdamW
import pytest

####################
### Forward Pass ###
####################

def test_forward_pass_2_layers():
    optimizer = AdamW(weight_decay=0.01)
    X_train = np.random.rand(5, 3)
    model = NeuralNetwork(
        [len(X_train[0]), 3, 2], 
        mse, 
        mse_grad, 
        optimizer,
        ["relu", "sigmoid"]
    )

    output = model.forward(X_train)

    # shape check
    assert output.shape == (5, 2)

    # values should be finite
    assert np.isfinite(output).all()

def test_forward_broadcast_single_activation_string():
    np.random.seed(1)
    optimizer = AdamW(weight_decay=0.01)
    X = np.random.rand(2, 3)

    # single string should broadcast to all layers
    model = NeuralNetwork([3, 3, 2], mse, mse_grad, optimizer, "relu")
    assert len(model.activations) == 2  # two layers -> two activations

    out = model.forward(X)
    assert out.shape == (2, 2)

def test_forward_mismatched_activation_count_raises():
    optimizer = AdamW(weight_decay=0.01)
    X = np.random.rand(2, 3)

    # three layers -> need 2 activations; passing 1 should raise/assert
    with pytest.raises(AssertionError):
        NeuralNetwork([3, 5, 2], mse, mse_grad, optimizer, ["relu"])

def test_forward_deterministic_given_fixed_weights():
    # If we force weights/biases to known constants, output should be deterministic
    optimizer = AdamW(weight_decay=0.01)
    X = np.array([[1.0, 0.5, -2.0],
                  [0.0, 0.0,  0.0]])

    model = NeuralNetwork([3, 4, 1], mse, mse_grad, optimizer, ["tanh", "sigmoid"])

    # Force weights/biases (adjust to your Linear layer attribute names)
    for layer in model.layers:
        layer.W[:] = 0.5   # or layer.weights
        layer.b[:] = 0.1   # or layer.bias

    out = model.forward(X)

    # Hard checks: shape + exact values (compute once and lock in)
    assert out.shape == (2, 1)
    # If you want to pin exact numbers, record them on first run:
    np.testing.assert_allclose(out, np.array([[0.450718], [0.57428 ]]), rtol=1e-6, atol=1e-6)

def test_forward_rejects_input_dim_mismatch():
    optimizer = AdamW(weight_decay=0.01)
    X = np.random.rand(5, 4)  # 4 features
    model = NeuralNetwork([3, 3, 1], mse, mse_grad, optimizer, ["relu", "sigmoid"])
    with pytest.raises((ValueError, AssertionError)):
        model.forward(X)

#####################
### Backward Pass ###
#####################

class DummyOptimizer:
    def __init__(self):
        self.calls = 0
        self.last_param_grads = None
    def step(self, param_grads):
        self.calls += 1
        self.last_param_grads = param_grads

def test_backward_returns_grad_shape():

    np.random.seed(0)
    X = np.random.randn(5, 3)
    y = np.random.randn(5, 1)

    opt = DummyOptimizer()
    model = NeuralNetwork([3, 4, 1], mse, mse_grad, opt, ["relu", "sigmoid"])

    yhat = model.forward(X)
    g_out = mse_grad(yhat, y)
    g_in = model.backward(g_out)

    assert g_in.shape == (5, 3)

def test_backward_calls_optimizer_once_and_pairs_all_params():
    np.random.seed(1)
    X = np.random.randn(4, 2)
    y = np.random.randn(4, 1)

    opt = DummyOptimizer()
    model = NeuralNetwork([2, 3, 1], mse, mse_grad, opt, ["relu", "sigmoid"])

    yhat = model.forward(X)
    g_out = mse_grad(yhat, y)
    _ = model.backward(g_out)

    # Optimizer called exactly once
    assert opt.calls == 1

    # One (W, dW) and (b, db) pair per layer
    param_grads = opt.last_param_grads
    assert len(param_grads) == 2 * len(model.layers)

    # Shapes match parameters
    for (param, grad) in param_grads:
        assert param.shape == grad.shape
        assert np.isfinite(grad).all()

def test_backward_zero_upstream_yields_zero_param_grads():
    np.random.seed(2)
    X = np.random.randn(3, 4)

    opt = DummyOptimizer()
    model = NeuralNetwork([4, 5, 2], mse, mse_grad, opt, ["tanh", "sigmoid"])

    _ = model.forward(X)
    g_out = np.zeros((X.shape[0], 2))
    _ = model.backward(g_out)

    for (_, g) in opt.last_param_grads:
        assert np.allclose(g, 0.0)

def test_backward_reverse_order():
    """
    Marks each layer's backward_pass to record call order.
    We stub backward_pass to pass grad through unchanged.
    """
    np.random.seed(3)
    X = np.random.randn(2, 3)
    y = np.random.randn(2, 1)

    opt = DummyOptimizer()
    model = NeuralNetwork([3, 7, 1], mse, mse_grad, opt, ["relu", "sigmoid"])

    order = []
    # Wrap original backward_pass to record order and pass through
    orig_bp = []
    for idx, L in enumerate(model.layers):
        orig_bp.append(L.backward_pass)
        def make_wrapper(i, orig):
            def wrapper(grad_out):
                order.append(i)
                # expect tuple (dW, db, grad_in)
                dW, db, grad_in = orig(grad_out)
                return dW, db, grad_in
            return wrapper
        model.layers[idx].backward_pass = make_wrapper(idx, L.backward_pass)

    yhat = model.forward(X)
    g_out = mse_grad(yhat, y)
    _ = model.backward(g_out)

    # Last layer index should appear first
    assert order[0] == len(model.layers) - 1
    # And we should have exactly one call per layer
    assert order == list(reversed(range(len(model.layers))))

def test_backward_matches_finite_difference_small_net(tol=1e-4):
    """
    Numerical gradient check on a tiny net with Identity activations to avoid kinks.
    Uses DummyOptimizer so params are not mutated.
    """
    np.random.seed(4)
    X = np.array([[ 1.0, -2.0],
                  [ 0.5,  0.3]])
    Y = np.array([[0.2],
                  [0.0]])

    opt = DummyOptimizer()
    # 2 -> 2 with Identity activation
    model = NeuralNetwork([2, 2], mse, mse_grad, opt, [Identity()])

    # Initialize weights to fixed values for determinism (if your Linear exposes W,b)
    W, b = model.layers[0].W, model.layers[0].b
    W[:] = np.array([[0.3, -0.2],
                     [0.1,  0.4]])
    b[:] = np.array([0.05, -0.1])

    def loss_value():
        yhat = model.forward(X)
        return mse(yhat, Y)

    _ = model.forward(X)
    g_out = mse_grad(model.forward(X), Y)
    _ = model.backward(g_out)
    # Extract analytic grads for W and b from captured list
    # last_param_grads = [(W_L, dW_L), (b_L, db_L), ...]; here only one layer
    (W_ref, dW_analytic), (b_ref, db_analytic) = opt.last_param_grads

    # finite differences
    def finite_diff(param, f, eps=1e-5):
        numg = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = param[idx]
            param[idx] = old + eps
            Lp = f()
            param[idx] = old - eps
            Lm = f()
            param[idx] = old
            numg[idx] = (Lp - Lm) / (2 * eps)
            it.iternext()
        return numg

    dW_num = finite_diff(W, loss_value)
    db_num = finite_diff(b, loss_value)

    np.testing.assert_allclose(dW_analytic, dW_num, rtol=tol, atol=tol)
    np.testing.assert_allclose(db_analytic, db_num, rtol=tol, atol=tol)
