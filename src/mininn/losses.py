from tensorgrad import Tensor


def mean_squared_error(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    return ((prediction - target) ** 2).mean()
