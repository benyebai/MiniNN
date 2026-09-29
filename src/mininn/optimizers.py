from collections.abc import Iterable

import numpy as np

from mininn.parameter import Parameter


def clip_grad_norm(
    parameters: Iterable[Parameter],
    max_norm: float,
) -> float:
    parameters = list(parameters)

    global_norm = 0.0
    for params in parameters:
        global_norm += np.sum((params.grad**2))

    global_norm = float(np.sqrt(global_norm))

    if global_norm > max_norm:
        scaling = max_norm / global_norm
        for params in parameters:
            params.grad *= scaling

    return global_norm


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


class Adam:
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float,
        m_keep_rate: float = 0.9,
        v_keep_rate: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.m_keep_rate = m_keep_rate
        self.v_keep_rate = v_keep_rate
        self.m = [np.zeros_like(params.data) for params in self.parameters]
        self.v = [np.zeros_like(params.data) for params in self.parameters]
        self.steps = 0
        self.epsilon = epsilon

    def step(self) -> None:
        self.steps += 1

        # remember when losses is big, imagine the equation graph
        # every grad is bigger, so we want to subtract and minimize.
        for i, params in enumerate(self.parameters):
            # 1) we already have the gradients
            # 2) update m and update v
            new_m = self.m_keep_rate * self.m[i] + (1 - self.m_keep_rate) * params.grad
            self.m[i] = new_m
            new_v = (
                self.v_keep_rate * self.v[i] + (1 - self.v_keep_rate) * params.grad**2
            )
            self.v[i] = new_v
            # 3) correct m, v
            corrected_m = new_m / (1 - (self.m_keep_rate**self.steps))
            corrected_v = new_v / (1 - (self.v_keep_rate**self.steps))
            # 4) calculate the movement
            # uh the epislon is just in case divide by 0, i guess its fine
            movement = corrected_m / (corrected_v ** (1 / 2) + self.epsilon)
            # 5) finally update the weight
            params.data -= self.learning_rate * movement
