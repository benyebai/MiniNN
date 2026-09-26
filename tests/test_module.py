import unittest

import numpy as np

from mininn.module import Module
from mininn.parameter import Parameter


class ExampleModule(Module):
    def __init__(self):
        self.weight = Parameter([[1.0, 2.0], [3.0, 4.0]])
        self.bias = Parameter([0.5, -0.5])

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]


class TestModule(unittest.TestCase):
    def test_base_module_has_no_parameters(self):
        self.assertEqual(Module().parameters(), [])

    def test_child_module_returns_owned_parameters(self):
        module = ExampleModule()

        self.assertEqual(module.parameters(), [module.weight, module.bias])

    def test_zero_grad_clears_all_parameter_gradients_in_place(self):
        module = ExampleModule()
        module.weight.grad[...] = 3.0
        module.bias.grad[...] = -2.0
        weight_grad_id = id(module.weight.grad)
        bias_grad_id = id(module.bias.grad)

        module.zero_grad()

        np.testing.assert_array_equal(
            module.weight.grad,
            np.zeros_like(module.weight.grad),
        )
        np.testing.assert_array_equal(
            module.bias.grad,
            np.zeros_like(module.bias.grad),
        )
        self.assertEqual(id(module.weight.grad), weight_grad_id)
        self.assertEqual(id(module.bias.grad), bias_grad_id)


if __name__ == "__main__":
    unittest.main()
