import numpy as np
from .core import BackproppableArray
from .utils import _unbroadcast

class BA_Add(BackproppableArray):
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x, y])
        self.x, self.y = x, y

    def grad_fn(self):
        self.x.grad += _unbroadcast(self.grad, self.x.data.shape)
        self.y.grad += _unbroadcast(self.grad, self.y.data.shape)

class BA_Sub(BackproppableArray):
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x, y])
        self.x, self.y = x, y

    def grad_fn(self):
        self.x.grad += _unbroadcast(self.grad, self.x.data.shape)
        self.y.grad -= _unbroadcast(self.grad, self.y.data.shape)

class BA_Mul(BackproppableArray):
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x, y])
        self.x, self.y = x, y

    def grad_fn(self):
        self.x.grad += _unbroadcast(self.grad * self.y.data, self.x.data.shape)
        self.y.grad += _unbroadcast(self.grad * self.x.data, self.y.data.shape)

class BA_Div(BackproppableArray):
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x, y])
        self.x, self.y = x, y

    def grad_fn(self):
        self.x.grad += _unbroadcast(self.grad / self.y.data, self.x.data.shape)
        self.y.grad += _unbroadcast(
            -self.grad * self.x.data / (self.y.data * self.y.data),
            self.y.data.shape
        )

class BA_MatMul(BackproppableArray):
    def __init__(self, x, y):
        assert len(x.data.shape) == 2
        assert len(y.data.shape) == 2
        super().__init__(x.data @ y.data, [x, y])
        self.x, self.y = x, y

    def grad_fn(self):
        self.x.grad += self.grad @ self.y.data.T
        self.y.grad += self.x.data.T @ self.grad

class BA_Sum(BackproppableArray):
    def __init__(self, x, axis):
        super().__init__(x.data.sum(axis, keepdims=True), [x])
        self.x = x
        self.axis = axis

    def grad_fn(self):
        self.x.grad += np.broadcast_to(self.grad, self.x.data.shape)

class BA_Reshape(BackproppableArray):
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        self.x.grad += np.asarray(self.grad).reshape(self.x.data.shape)

class BA_Transpose(BackproppableArray):
    def __init__(self, x, axes):
        super().__init__(x.data.transpose(axes), [x])
        self.x = x
        self.axes = axes

    def grad_fn(self):
        inv = np.argsort(self.axes)
        self.x.grad += np.transpose(self.grad, inv)
