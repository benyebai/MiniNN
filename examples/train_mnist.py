from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mnist_data import load_mnist
from tensorgrad import Tensor

from mininn.activations import SiLU, Softmax
from mininn.linear import Linear
from mininn.losses import mean_squared_error
from mininn.optimizers import SGD
from mininn.sequential import Sequential

train_images, train_labels, test_images, test_labels = load_mnist(Path("data/mnist"))


# e.g 3 -> [0, 0, 0, 3, 0, 0, 0, 0, 0, 0]
def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    targets = np.zeros(
        (labels.shape[0], classes),
        dtype=np.float64,
    )

    targets[np.arange(labels.shape[0]), labels] = 1.0

    return targets


# Why average across the whole batch? the lossses?
# well lets say u dont then a specific weight.grad will acculmate across all images in the batch
# and u would have to average it out with divide by batch anyways, but y not just once at the beginning

model = Sequential(Linear(784, 128), SiLU(), Linear(128, 10), Softmax(axis=-1))

optimizer = SGD(model.parameters(), 0.5)

train_targets = one_hot(train_labels)
loss_history: list[float] = []


for epoch in range(30):
    indices = np.random.permutation(train_images.shape[0])
    total_loss = 0.0
    correct = 0

    # batching!
    for start in range(0, train_images.shape[0], 128):
        batch_indices = indices[start : start + 128]

        batch_images = Tensor(train_images[batch_indices])
        batch_targets = Tensor(train_targets[batch_indices])

        model.zero_grad()

        predictions = model(batch_images)
        loss = mean_squared_error(predictions, batch_targets)

        loss.backward()
        optimizer.step()

        current_batch_size = batch_indices.shape[0]
        total_loss += loss.data.item() * current_batch_size

        predicted_labels = np.argmax(predictions.data, axis=1)
        correct += np.sum(predicted_labels == train_labels[batch_indices])

    average_loss = total_loss / train_images.shape[0]
    accuracy = correct / train_images.shape[0]
    loss_history.append(average_loss)

    print(f"epoch {epoch + 1}: loss={average_loss:.6f}, accuracy={accuracy:.2%}")


accuracy = test_accuracy(model)
output_path = save_prediction_grid(model, accuracy)
animation_path = save_prediction_animation(model)
loss_graph_path = save_loss_graph(loss_history)
print(f"test accuracy: {accuracy:.2%}")
print(f"saved prediction graphic to {output_path}")
print(f"saved prediction animation to {animation_path}")
print(f"saved training loss graph to {loss_graph_path}")
