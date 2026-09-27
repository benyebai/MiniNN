import unittest

import numpy as np

from mininn.optimizers import SGD
from mininn.parameter import Parameter


class TestSGD(unittest.TestCase):
    def test_step_matches_hand_calculated_update(self):
        parameter = Parameter([1.0, -2.0])
        parameter.grad[...] = [0.5, -1.0]
        optimizer = SGD([parameter], learning_rate=0.1)

        optimizer.step()

        # [1.0, -2.0] - 0.1 * [0.5, -1.0]
        np.testing.assert_allclose(parameter.data, [0.95, -1.9])

    def test_step_updates_every_parameter(self):
        weight = Parameter([[1.0, 2.0], [3.0, 4.0]])
        bias = Parameter([0.5, -0.5])
        weight.grad[...] = [[1.0, -2.0], [0.5, 3.0]]
        bias.grad[...] = [2.0, -4.0]
        optimizer = SGD([weight, bias], learning_rate=0.2)

        optimizer.step()

        np.testing.assert_allclose(
            weight.data,
            [[0.8, 2.4], [2.9, 3.4]],
        )
        np.testing.assert_allclose(bias.data, [0.1, 0.3])

    def test_step_does_not_clear_or_modify_gradients(self):
        parameter = Parameter([1.0, 2.0])
        parameter.grad[...] = [3.0, -4.0]
        gradient_before = parameter.grad.copy()
        optimizer = SGD([parameter], learning_rate=0.1)

        optimizer.step()

        np.testing.assert_array_equal(parameter.grad, gradient_before)

    def test_optimizer_updates_the_original_parameter_object(self):
        parameter = Parameter([2.0])
        parameter.grad[...] = [1.5]
        optimizer = SGD([parameter], learning_rate=0.4)

        optimizer.step()

        self.assertIs(optimizer.parameters[0], parameter)
        np.testing.assert_allclose(parameter.data, [1.4])


if __name__ == "__main__":
    unittest.main()
