import unittest

import numpy as np

from mininn.optimizers import Adam, SGD, clip_grad_norm
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


class TestAdam(unittest.TestCase):
    def test_first_step_matches_hand_calculated_update(self):
        parameter = Parameter([1.0, -2.0])
        parameter.grad[...] = [0.5, -1.0]
        optimizer = Adam([parameter], learning_rate=0.1)

        optimizer.step()

        # m = 0.9 * 0 + 0.1 * gradient
        np.testing.assert_allclose(optimizer.m[0], [0.05, -0.1])
        # v = 0.999 * 0 + 0.001 * gradient**2
        np.testing.assert_allclose(optimizer.v[0], [0.00025, 0.001])
        self.assertEqual(optimizer.steps, 1)

        # After bias correction, m_hat=[0.5, -1.0] and
        # v_hat=[0.25, 1.0], so the movement is approximately [1, -1].
        np.testing.assert_allclose(
            parameter.data,
            [0.9, -1.9],
            rtol=1e-7,
            atol=1e-8,
        )


class TestClipGradNorm(unittest.TestCase):
    def test_clips_global_norm_across_all_parameters(self):
        first = Parameter([10.0])
        second = Parameter([20.0])
        first.grad[...] = [3.0]
        second.grad[...] = [4.0]
        first_grad_id = id(first.grad)
        second_grad_id = id(second.grad)

        original_norm = clip_grad_norm(
            (parameter for parameter in [first, second]),
            max_norm=2.5,
        )

        # sqrt(3**2 + 4**2) = 5, so every gradient is scaled by 2.5 / 5.
        self.assertAlmostEqual(original_norm, 5.0)
        np.testing.assert_allclose(first.grad, [1.5])
        np.testing.assert_allclose(second.grad, [2.0])
        self.assertEqual(id(first.grad), first_grad_id)
        self.assertEqual(id(second.grad), second_grad_id)
        np.testing.assert_array_equal(first.data, [10.0])
        np.testing.assert_array_equal(second.data, [20.0])

    def test_leaves_gradients_unchanged_when_norm_is_below_limit(self):
        parameter = Parameter([7.0, 8.0])
        parameter.grad[...] = [0.3, 0.4]
        gradient_before = parameter.grad.copy()

        original_norm = clip_grad_norm([parameter], max_norm=1.0)

        self.assertAlmostEqual(original_norm, 0.5)
        np.testing.assert_array_equal(parameter.grad, gradient_before)


if __name__ == "__main__":
    unittest.main()
