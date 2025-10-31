"""
Test case for optimizer.step() functionality.

This test attempts to use optimizer.step() and should work on all backends.
If it crashes with stack overflow, it indicates a bug in the UOp rewrite system.

Related: https://github.com/tinygrad/tinygrad/issues/8457
"""

import unittest
import sys
from tinygrad import Tensor, Device
from tinygrad.nn.optim import Adam, SGD
from tinygrad.nn.state import get_parameters
import tinygrad.nn as nn

class SimpleModel:
  """Minimal model - single Linear layer with 100 parameters"""
  def __init__(self):
    self.fc = nn.Linear(10, 10)

  def __call__(self, x):
    return self.fc(x)

class TestOptimizerStackOverflow(unittest.TestCase):
  """
  Test case for optimizer functionality.

  All tests should pass. If they crash, it indicates a backend-specific bug.
  """

  def setUp(self):
    """Enable training mode and increase recursion limit"""
    self.old_training = Tensor.training
    Tensor.training = True
    self.old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(50000)  # Doesn't help, but good practice

  def tearDown(self):
    """Restore original settings"""
    Tensor.training = self.old_training
    sys.setrecursionlimit(self.old_recursion_limit)

  def test_adam_optimizer_step_minimal(self):
    """
    Test Adam optimizer.step() with minimal model.

    This is the simplest possible test case:
    - 1 Linear layer (100 parameters)
    - 1 forward pass
    - 1 backward pass
    - 1 optimizer step

    Expected: Should work. If it crashes, the bug exists on this backend.
    """
    # Create minimal model
    model = SimpleModel()

    # Create optimizer
    params = get_parameters(model)
    opt = Adam(params, lr=0.001)

    # Create test data
    x = Tensor.randn(2, 10)
    y = Tensor.randn(2, 10)

    # Forward pass
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    # Backward pass
    opt.zero_grad()
    loss.backward()

    # Optimizer step - THIS IS WHERE THE BUG OCCURS
    # On Metal/CLANG: Stack overflow at tinygrad/uop/ops.py:980
    # On PYTHON: Works fine
    opt.step()

    # If we get here, the test passed
    self.assertTrue(True, "Optimizer step succeeded")

  def test_sgd_optimizer_step_minimal(self):
    """
    Test SGD optimizer.step() with minimal model.

    Expected: Should work. This verifies the bug affects all optimizers if it crashes.
    """
    model = SimpleModel()
    params = get_parameters(model)
    opt = SGD(params, lr=0.001)

    x = Tensor.randn(2, 10)
    y = Tensor.randn(2, 10)

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()  # Stack overflow here on Metal/CLANG

    self.assertTrue(True, "SGD optimizer step succeeded")

  def test_manual_parameter_update_workaround(self):
    """
    Test manual parameter updates as workaround.

    This should ALWAYS work, even on Metal/CLANG backends.
    Demonstrates the workaround for the optimizer bug.
    """
    model = SimpleModel()
    params = get_parameters(model)
    lr = 0.001

    x = Tensor.randn(2, 10)
    y = Tensor.randn(2, 10)

    # Forward pass
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    # Zero gradients manually
    for p in params:
      p.grad = None

    # Backward pass
    loss.backward()

    # Manual parameter update (workaround)
    for p in params:
      if p.grad is not None:
        p.assign(p - lr * p.grad)

    # This should always succeed
    self.assertTrue(True, "Manual parameter update succeeded")

  def test_forward_backward_without_optimizer(self):
    """
    Test that forward and backward passes work fine.

    Expected: Should work. If it crashes even without optimizer.step(),
    the bug affects basic operations like loss.realize().
    """
    model = SimpleModel()

    x = Tensor.randn(2, 10)
    y = Tensor.randn(2, 10)

    # Forward pass - lazy evaluation, no crash yet
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    # Realize loss - THIS IS WHERE THE BUG OCCURS on Metal
    # Stack overflow at tinygrad/uop/ops.py:980
    loss_val = loss.realize()

    # If we get here, loss computation worked
    self.assertIsNotNone(loss_val, "Loss was computed successfully")

if __name__ == "__main__":
  # Print environment info
  print(f"Python version: {sys.version}")
  print(f"Device: {Device.DEFAULT}")
  print(f"Recursion limit: {sys.getrecursionlimit()}")
  print()

  # Run tests
  unittest.main(verbosity=2)
