from tensorgrad import Tensor

from mininn.module import Module


class Sigmoid(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class SiLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return x.silu()


class Softmax(Module):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(self.axis)
