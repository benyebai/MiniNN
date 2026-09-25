import unittest

from mininn import Tensor


class TestInstallation(unittest.TestCase):
    def test_tensorgrad_is_available(self):
        value = Tensor([1.0, 2.0, 3.0])
        result = (value * 2.0).sum()
        result.backward()

        self.assertEqual(result.data.item(), 12.0)
        self.assertEqual(value.grad.tolist(), [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
