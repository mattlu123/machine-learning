import numpy as np
from mlplayground.core.optim import AdamW, BatchGD

def test_batchgd():
    W = np.ones((2,2))
    dW = np.full_like(W, 0.5)
    opt = BatchGD(lr=0.1)
    opt.step([(W, dW)])
    assert np.allclose(W, 1 - 0.05)

def test_adamw():
    W = np.ones((2,2))
    dW = np.zeros_like(W)
    opt = AdamW(lr=1e-2, weight_decay=0.1, delta=1e-8)
    opt.step([(W, dW)])
    assert np.all(W < 1.0)