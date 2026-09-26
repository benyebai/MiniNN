import unittest

import numpy as np
from tensorgrad import Tensor

from mininn.linear import Linear


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.layer = Linear(input_features=3, output_features=2)
        self.layer.weight.data[...] = [
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
        ]
        self.layer.bias.data[...] = [0.5, -1.0]

    def test_parameter_shapes(self):
        self.assertEqual(self.layer.weight.data.shape, (2, 3))
        self.assertEqual(self.layer.bias.data.shape, (2,))

    def test_parameters_returns_weight_and_bias(self):
        self.assertEqual(
            self.layer.parameters(),
            [self.layer.weight, self.layer.bias],
        )

    def test_vector_forward_matches_hand_calculation(self):
        inputs = Tensor([2.0, -1.0, 0.5])

        output = self.layer(inputs)

        self.assertEqual(output.data.shape, (2,))
        np.testing.assert_allclose(output.data, [2.0, -2.5])

    def test_batch_forward_preserves_leading_dimension(self):
        inputs = Tensor([
            [2.0, -1.0, 0.5],
            [0.0, 1.0, 2.0],
        ])

        output = self.layer(inputs)

        self.assertEqual(output.data.shape, (2, 2))
        np.testing.assert_allclose(
            output.data,
            [
                [2.0, -2.5],
                [8.5, 3.5],
            ],
        )

    def test_backward_reaches_input_weight_and_bias(self):
        inputs = Tensor([[2.0, -1.0, 0.5], [0.0, 1.0, 2.0]])

        self.layer(inputs).sum().backward()

        np.testing.assert_allclose(
            inputs.grad,
            [[0.0, 2.5, 5.0], [0.0, 2.5, 5.0]],
        )
        np.testing.assert_allclose(
            self.layer.weight.grad,
            [[2.0, 0.0, 2.5], [2.0, 0.0, 2.5]],
        )
        np.testing.assert_allclose(self.layer.bias.grad, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
