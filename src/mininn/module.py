from tensorgrad import Tensor

from mininn.parameter import Parameter


class Module:
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("Must implement __call__ function")

    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad[...] = 0.0
