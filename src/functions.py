import numpy as np
from .core import BackproppableArray, to_ba

class BA_Exp(BackproppableArray):
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        self.x.grad += self.grad * np.exp(self.x.data)

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    return np.exp(x)

class BA_Log(BackproppableArray):
    def __init__(self, x):
        super().__init__(np.log(x.data), [x])
        self.x = x

    def grad_fn(self):
        self.x.grad += self.grad / self.x.data

def log(x):
    if isinstance(x, BackproppableArray):
        return BA_Log(x)
    return np.log(x)
