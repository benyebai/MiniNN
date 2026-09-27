from collections.abc import Iterable

from mininn.parameter import Parameter


class SGD:
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float,
    ):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self) -> None:

        # remember when losses is big, imagine the equation graph
        # every grad is bigger, so we want to subtract and minimize.
        for params in self.parameters:
            params.data -= self.learning_rate * params.grad
