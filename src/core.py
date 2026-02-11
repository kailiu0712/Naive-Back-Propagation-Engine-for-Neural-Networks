import numpy as np

_global_order_counter = 0

def get_next_order() -> int:
    global _global_order_counter
    rv = _global_order_counter
    _global_order_counter += 1
    return rv

def to_ba(x):
    from .ops import BackproppableArray  # avoid circular import issues
    if isinstance(x, BackproppableArray):
        return x
    if isinstance(x, np.ndarray):
        return BackproppableArray(x)
    if isinstance(x, float):
        return BackproppableArray(np.array(x))
    if isinstance(x, int):
        return BackproppableArray(np.array(float(x)))
    raise TypeError(f"could not convert {type(x)} to BackproppableArray")

class BackproppableArray:
    def __init__(self, np_array, dependencies=None):
        self.data = np_array
        self.grad = None
        self.order = get_next_order()
        self.dependencies = dependencies or []

    def __repr__(self):
        return f"({self.data}, type={type(self).__name__})"

    def all_dependencies(self):
        # BFS for unique nodes
        all_deps = []
        stack = [self]
        visited = set()
        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in visited:
                continue
            visited.add(nid)
            all_deps.append(node)
            stack.extend(node.dependencies)
        return all_deps

    def backward(self):
        nodes = sorted(self.all_dependencies(), key=lambda n: n.order, reverse=True)

        for node in nodes:
            node.grad = np.zeros_like(node.data)

        self.grad = np.array(1.0)

        for node in nodes:
            node.grad_fn()

    def grad_fn(self):
        pass

    # operator overloading lives here, but classes in ops.py
    def __add__(self, other):
        from .ops import BA_Add
        return BA_Add(self, to_ba(other))

    def __sub__(self, other):
        from .ops import BA_Sub
        return BA_Sub(self, to_ba(other))

    def __mul__(self, other):
        from .ops import BA_Mul
        return BA_Mul(self, to_ba(other))

    def __truediv__(self, other):
        from .ops import BA_Div
        return BA_Div(self, to_ba(other))

    def __radd__(self, other):
        from .ops import BA_Add
        return BA_Add(to_ba(other), self)

    def __rsub__(self, other):
        from .ops import BA_Sub
        return BA_Sub(to_ba(other), self)

    def __rmul__(self, other):
        from .ops import BA_Mul
        return BA_Mul(to_ba(other), self)

    def __rtruediv__(self, other):
        from .ops import BA_Div
        return BA_Div(to_ba(other), self)

    def __matmul__(self, other):
        from .ops import BA_MatMul
        return BA_MatMul(self, to_ba(other))

    def __rmatmul__(self, other):
        from .ops import BA_MatMul
        return BA_MatMul(to_ba(other), self)

    def sum(self, axis=None, keepdims=True):
        from .ops import BA_Sum
        # keepdims hardcoded in your BA_Sum; keep param for API compatibility
        return BA_Sum(self, axis)

    def reshape(self, shape):
        from .ops import BA_Reshape
        return BA_Reshape(self, shape)

    def transpose(self, axes=None):
        from .ops import BA_Transpose
        if axes is None:
            axes = tuple(range(self.data.ndim))[::-1]
        return BA_Transpose(self, axes)
