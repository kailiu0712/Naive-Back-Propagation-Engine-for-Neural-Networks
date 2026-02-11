import time
import numpy as np
from src import numerical_diff, numerical_grad, backprop_diff
from tests.test_functions import TestFxs

def main():
    np.random.seed(42)
    num = np.random.randn()
    arr = np.random.randn(5)
    arr_5000 = np.random.randn(5000)

    print("grads for f1:", TestFxs.df1dx(num), numerical_diff(TestFxs.f1, num), backprop_diff(TestFxs.f1, num))
    print("grads for f2:", TestFxs.df2dx(num), numerical_diff(TestFxs.f2, num), backprop_diff(TestFxs.f2, num))
    print("grads for f3:", TestFxs.df3dx(num), numerical_diff(TestFxs.f3, num), backprop_diff(TestFxs.f3, num))

    print("grads for f4:", numerical_diff(TestFxs.f4, num), backprop_diff(TestFxs.f4, num))
    print("grads for g1:", numerical_diff(TestFxs.g1, num), backprop_diff(TestFxs.g1, num))
    print("grads for g2:", numerical_diff(TestFxs.g2, num), backprop_diff(TestFxs.g2, num))

    h1d2 = numerical_grad(TestFxs.h1, arr)
    h1d3 = backprop_diff(TestFxs.h1, arr)
    print("numerical grads for h1:", h1d2)
    print("autodiff grads for h1:", h1d3)

    start = time.time()
    h2d2 = numerical_grad(TestFxs.h2, arr_5000)
    mid = time.time()
    h2d3 = backprop_diff(TestFxs.h2, arr_5000)
    end = time.time()

    print("time for numerical grads (s):", mid - start)
    print("time for autodiff grads (s):", end - mid)

if __name__ == "__main__":
    main()
