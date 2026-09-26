import unittest

import numpy as np
from tensorgrad import Tensor

from mininn.activations import Sigmoid, SiLU, Softmax, Tanh


class TestActivations(unittest.TestCase):
    def test_parameterless_activations_have_no_parameters(self):
        for activation in [Sigmoid(), Tanh(), SiLU(), Softmax(axis=-1)]:
            with self.subTest(activation=type(activation).__name__):
                self.assertEqual(activation.parameters(), [])

    def test_sigmoid_matches_numpy(self):
        data = np.array([-2.0, 0.0, 2.0])

        output = Sigmoid()(Tensor(data))

        np.testing.assert_allclose(output.data, 1.0 / (1.0 + np.exp(-data)))

    def test_tanh_matches_numpy(self):
        data = np.array([-2.0, 0.0, 2.0])

        output = Tanh()(Tensor(data))

        np.testing.assert_allclose(output.data, np.tanh(data))

    def test_silu_matches_numpy(self):
        data = np.array([-2.0, 0.0, 2.0])
        sigmoid = 1.0 / (1.0 + np.exp(-data))

        output = SiLU()(Tensor(data))

        np.testing.assert_allclose(output.data, data * sigmoid)

    def test_softmax_uses_configured_axis(self):
        data = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])

        output = Softmax(axis=-1)(Tensor(data))

        shifted = data - np.max(data, axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.sum(
            np.exp(shifted), axis=-1, keepdims=True
        )
        np.testing.assert_allclose(output.data, expected)
        np.testing.assert_allclose(output.data.sum(axis=-1), np.ones(2))


if __name__ == "__main__":
    unittest.main()
