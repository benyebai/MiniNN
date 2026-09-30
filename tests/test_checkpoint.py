import tempfile
import unittest
from pathlib import Path

import numpy as np

from mininn.checkpoint import load_checkpoint, save_checkpoint
from mininn.linear import Linear
from mininn.optimizers import Adam


class TestCheckpoint(unittest.TestCase):
    def test_restored_training_matches_uninterrupted_training(self):
        uninterrupted_model = Linear(2, 1)
        uninterrupted_model.weight.data[...] = [[1.0, -2.0]]
        uninterrupted_model.bias.data[...] = [0.5]
        uninterrupted_optimizer = Adam(
            uninterrupted_model.parameters(),
            learning_rate=0.01,
        )

        first_gradients = [np.array([[0.4, -0.2]]), np.array([0.1])]
        for parameter, gradient in zip(
            uninterrupted_model.parameters(), first_gradients
        ):
            parameter.grad[...] = gradient
        uninterrupted_optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.npz"
            save_checkpoint(
                checkpoint_path,
                uninterrupted_model,
                uninterrupted_optimizer,
            )

            restored_model = Linear(2, 1)
            restored_optimizer = Adam(
                restored_model.parameters(),
                learning_rate=123.0,
                m_keep_rate=0.5,
                v_keep_rate=0.5,
                epsilon=1e-4,
            )
            load_checkpoint(checkpoint_path, restored_model, restored_optimizer)

        for expected, actual in zip(
            uninterrupted_model.parameters(), restored_model.parameters()
        ):
            np.testing.assert_array_equal(actual.data, expected.data)
        for expected, actual in zip(
            uninterrupted_optimizer.m, restored_optimizer.m
        ):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(
            uninterrupted_optimizer.v, restored_optimizer.v
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(restored_optimizer.steps, uninterrupted_optimizer.steps)
        self.assertEqual(
            restored_optimizer.learning_rate,
            uninterrupted_optimizer.learning_rate,
        )
        self.assertEqual(
            restored_optimizer.m_keep_rate,
            uninterrupted_optimizer.m_keep_rate,
        )
        self.assertEqual(
            restored_optimizer.v_keep_rate,
            uninterrupted_optimizer.v_keep_rate,
        )
        self.assertEqual(restored_optimizer.epsilon, uninterrupted_optimizer.epsilon)

        second_gradients = [np.array([[-0.3, 0.6]]), np.array([-0.2])]
        for model in [uninterrupted_model, restored_model]:
            for parameter, gradient in zip(model.parameters(), second_gradients):
                parameter.grad[...] = gradient

        uninterrupted_optimizer.step()
        restored_optimizer.step()

        for expected, actual in zip(
            uninterrupted_model.parameters(), restored_model.parameters()
        ):
            np.testing.assert_array_equal(actual.data, expected.data)
        for expected, actual in zip(
            uninterrupted_optimizer.m, restored_optimizer.m
        ):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(
            uninterrupted_optimizer.v, restored_optimizer.v
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(restored_optimizer.steps, 2)

    def test_rejects_checkpoint_with_different_parameter_shapes(self):
        original_model = Linear(2, 1)
        original_optimizer = Adam(original_model.parameters(), learning_rate=0.01)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.npz"
            save_checkpoint(checkpoint_path, original_model, original_optimizer)

            incompatible_model = Linear(3, 1)
            incompatible_optimizer = Adam(
                incompatible_model.parameters(), learning_rate=0.01
            )

            with self.assertRaisesRegex(ValueError, "shape does not match"):
                load_checkpoint(
                    checkpoint_path,
                    incompatible_model,
                    incompatible_optimizer,
                )


if __name__ == "__main__":
    unittest.main()
