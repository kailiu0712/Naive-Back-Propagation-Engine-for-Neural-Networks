# BackproppableArray (NumPy Autodiff Engine)

A minimal, naive **reverse-mode automatic differentiation** engine built on top of just **NumPy**.  
It implements a small computational graph system where arrays track dependencies and gradients can be computed via `.backward()`.

This project is designed to be **simple, readable, and hackable**—ideal for understanding how backpropagation works on the foundational level.

---

## Features

- **BackproppableArray** tensor type with:
  - `.data` (NumPy array storage)
  - `.grad` (accumulated gradient)
  - dependency tracking and topological backprop

- **Supported operations**
  - Elementwise: `+`, `-`, `*`, `/` (with broadcasting support via `_unbroadcast`)
  - Matrix multiplication: `@` (2D matrices)
  - Reductions/shape ops: `.sum()`, `.reshape()`, `.transpose()`
  - Functions: `exp`, `log`

- **Gradient checking**
  - Finite difference: `numerical_diff`, `numerical_grad`
  - Autodiff: `backprop_diff`

---

## Repository Structure

backproppable-array/
├─ README.md
├─ pyproject.toml
├─ src/
│   ├─ __init__.py
│   ├─ core.py        # BackproppableArray + to_ba + order counter + backward traversal
│   ├─ ops.py         # BA_Add/Sub/Mul/Div/MatMul/Sum/Reshape/Transpose
│   ├─ functions.py   # exp/log wrappers (extend here for new ops)
│   ├─ utils.py       # _unbroadcast + numerical_diff/numerical_grad
│   └─ autodiff.py    # backprop_diff
├─ tests/
│  ├─ test_functions.py
│  └─ test_gradients.py
└─ scripts/
   └─ run_demo.py
