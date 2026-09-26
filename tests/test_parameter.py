import unittest

import numpy as np
from tensorgrad import Tensor

from mininn.parameter import Parameter


class TestParameter(unittest.TestCase):
    def test_parameter_is_a_tensor(self):
        parameter = Parameter([1.0, 2.0, 3.0], name="weight")

        self.assertIsInstance(parameter, Tensor)
        self.assertEqual(parameter.name, "weight")

    def test_parameter_uses_float64_data_and_zero_gradient(self):
        parameter = Parameter([[1, 2], [3, 4]])

        self.assertEqual(parameter.data.dtype, np.dtype("float64"))
        self.assertEqual(parameter.grad.shape, parameter.data.shape)
        np.testing.assert_array_equal(
            parameter.grad,
            np.zeros_like(parameter.data),
        )

    def test_parameter_participates_in_backpropagation(self):
        parameter = Parameter([1.0, 2.0, 3.0])

        loss = (parameter * 2.0).sum()
        loss.backward()

        np.testing.assert_array_equal(parameter.grad, [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
