import numpy as np
from src import exp, log

class TestFxs:
    @staticmethod
    def f1(x): return x * 2 + 3

    @staticmethod
    def df1dx(x): return 2

    @staticmethod
    def f2(x): return x * x

    @staticmethod
    def df2dx(x): return 2 * x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        return (1 - (x - 2) * (x - 2)) / (((x - 2) * (x - 2) + 1) ** 2)

    @staticmethod
    def f4(x):
        return log(exp(x*x / 8 - 3*x + 5) + x)

    @staticmethod
    def g1(x):
        a = np.ones(3, dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())

    @staticmethod
    def g2(x):
        a = np.ones((4,5), dtype="float64")
        b = np.arange(20, dtype="float64")
        ax = x - a
        bx = log((x + b)*(x + b)).reshape((4,5)).transpose()
        y = bx @ ax
        return y.sum().reshape(())

    @staticmethod
    def h1(x):
        b = np.arange(5, dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())

    @staticmethod
    def h2(x):
        xb = exp(x + 3) / (4.2 * x.sum())
        xb += np.arange(5000, dtype="float64")
        return (xb * xb).sum().reshape(())
