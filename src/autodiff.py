from .core import to_ba

def backprop_diff(f, x):
    ba_x = to_ba(x)
    fx = f(ba_x)
    fx.backward()
    return ba_x.grad
