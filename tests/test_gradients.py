import numpy as np
from src import numerical_diff, numerical_grad, backprop_diff
from .test_functions import TestFxs

def test_scalar_tests():
    np.random.seed(42)
    x = np.random.randn()

    assert np.allclose(TestFxs.df1dx(x), numerical_diff(TestFxs.f1, x), atol=1e-5)
    assert np.allclose(TestFxs.df1dx(x), backprop_diff(TestFxs.f1, x), atol=1e-5)

    assert np.allclose(TestFxs.df2dx(x), numerical_diff(TestFxs.f2, x), atol=1e-5)
    assert np.allclose(TestFxs.df2dx(x), backprop_diff(TestFxs.f2, x), atol=1e-5)

    assert np.allclose(TestFxs.df3dx(x), numerical_diff(TestFxs.f3, x), atol=1e-5)
    assert np.allclose(TestFxs.df3dx(x), backprop_diff(TestFxs.f3, x), atol=1e-5)

def test_vector_tests():
    np.random.seed(42)
    x = np.random.randn(5)

    ng = numerical_grad(TestFxs.h1, x)
    bg = backprop_diff(TestFxs.h1, x)
    assert np.allclose(ng, bg, atol=1e-5)
