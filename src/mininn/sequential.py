import numpy as np
from tensorgrad import Tensor

from mininn.module import Module, Parameter


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = modules

    def __call__(self, inputs: Tensor) -> Tensor:
        for module in self.modules:
            inputs = module(inputs)

        return inputs

    def parameters(self) -> list[Parameter]:
        parameters = []

        for module in self.modules:
            parameters += module.parameters()

        return parameters
