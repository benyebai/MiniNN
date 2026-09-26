import unittest

import numpy as np
from tensorgrad import Tensor

from mininn.activations import SiLU
from mininn.linear import Linear
from mininn.module import Module
from mininn.sequential import Sequential


class AddOne(Module):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs + 1.0


class MultiplyByTwo(Module):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs * 2.0


class TestSequential(unittest.TestCase):
    def test_applies_modules_in_order(self):
        model = Sequential(AddOne(), MultiplyByTwo())

        output = model(Tensor([1.0, 2.0]))

        # (inputs + 1) * 2, not inputs * 2 + 1.
        np.testing.assert_array_equal(output.data, [4.0, 6.0])

    def test_empty_sequential_returns_input_unchanged(self):
        model = Sequential()
        inputs = Tensor([1.0, 2.0])

        output = model(inputs)

        self.assertIs(output, inputs)

    def test_parameters_combines_parameters_from_each_module(self):
        first = Linear(2, 3)
        activation = SiLU()
        second = Linear(3, 1)
        model = Sequential(first, activation, second)

        self.assertEqual(
            model.parameters(),
            [first.weight, first.bias, second.weight, second.bias],
        )

    def test_zero_grad_clears_parameters_in_every_module(self):
        first = Linear(2, 3)
        second = Linear(3, 1)
        model = Sequential(first, SiLU(), second)
        for parameter in model.parameters():
            parameter.grad[...] = 5.0

        model.zero_grad()

        for parameter in model.parameters():
            np.testing.assert_array_equal(
                parameter.grad,
                np.zeros_like(parameter.grad),
            )

    def test_linear_activation_linear_forward(self):
        first = Linear(2, 2)
        first.weight.data[...] = [[1.0, 0.0], [0.0, 1.0]]
        first.bias.data[...] = [0.0, 0.0]

        second = Linear(2, 1)
        second.weight.data[...] = [[2.0, -1.0]]
        second.bias.data[...] = [0.5]

        model = Sequential(first, SiLU(), second)
        inputs = Tensor([1.0, -1.0])

        output = model(inputs)

        sigmoid = 1.0 / (1.0 + np.exp(-inputs.data))
        hidden = inputs.data * sigmoid
        expected = hidden @ second.weight.data.T + second.bias.data
        np.testing.assert_allclose(output.data, expected)


if __name__ == "__main__":
    unittest.main()
