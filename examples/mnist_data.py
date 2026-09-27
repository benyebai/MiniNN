import gzip
import struct
from pathlib import Path

import numpy as np


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count, rows, columns = struct.unpack(
            ">IIII",
            file.read(16),
        )

        if magic != 2051:
            raise ValueError("Invalid image magic")

        pixels = np.frombuffer(file.read(), dtype=np.uint8)

    images = pixels.reshape(count, rows * columns)

    return images.astype(np.float64) / 255.0


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        magic, count = struct.unpack(
            ">II",
            file.read(8),
        )

        if magic != 2049:
            raise ValueError("Invalid label magic")

        labels = np.frombuffer(file.read(), dtype=np.uint8)

    if labels.shape != (count,):
        raise ValueError("label count no good")

    return labels


def load_mnist(data_directory: Path):
    train_images = load_images(data_directory / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(data_directory / "train-labels-idx1-ubyte.gz")
    test_images = load_images(data_directory / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(data_directory / "t10k-labels-idx1-ubyte.gz")

    return train_images, train_labels, test_images, test_labels
