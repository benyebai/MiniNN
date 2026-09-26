import numpy as np
from tensorgrad import Tensor

from mininn.module import Module, Parameter


class Linear(Module):
    def __init__(self, input_features: int, output_features: int):
        # output_features amount of rows
        # input_features amount of columns

        # Create weight and bias Parameters here.
        # we use xavier as starting (idk why but supposedly it dosent matter too much)
        bound = np.sqrt(6.0 / (input_features + output_features))
        self.weight = Parameter(
            np.random.uniform(-bound, bound, size=(output_features, input_features)),
            name="weight",
        )
        self.bias = Parameter(np.zeros(output_features), name="bias")

    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight.transpose() + self.bias

    def parameters(self) -> list[Parameter]:
        return [self.weight, self.bias]
