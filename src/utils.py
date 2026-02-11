import numpy as np

def _unbroadcast(grad, target_shape):
    g = grad if isinstance(grad, np.ndarray) else np.asarray(grad)

    if target_shape == ():
        return np.asarray(g.sum())

    target_shape = tuple(target_shape)

    if g.ndim >= len(target_shape):
        padded = (1,) * (g.ndim - len(target_shape)) + target_shape
    else:
        padded = target_shape

    axes = [i for i, (gs, ts) in enumerate(zip(g.shape, padded)) if ts == 1 and gs != 1]
    if axes:
        g = g.sum(axis=tuple(axes), keepdims=True)

    return g.reshape(target_shape)

def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def numerical_grad(f, x, eps=1e-5):
    x = np.asarray(x, dtype="float64")
    grad = np.zeros_like(x, dtype="float64")
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        e = np.zeros_like(x, dtype="float64")
        e[idx] = eps
        grad[idx] = (f(x + e) - f(x - e)) / (2 * eps)
        it.iternext()
    return grad
