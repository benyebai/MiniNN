from tensorgrad import Tensor


class Parameter(Tensor):
    def __init__(self, data, name: str = ""):
        super().__init__(data, name=name)
