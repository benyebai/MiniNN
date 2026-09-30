from pathlib import Path

import numpy as np

from mininn.module import Module
from mininn.optimizers import Adam


def save_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Adam,
) -> None:
    parameters = model.parameters()

    if len(parameters) != len(optimizer.parameters):
        raise ValueError("Model and optimizer must contain the same parameters")

    checkpoint = {
        "format_version": np.array(1),
        "parameter_count": np.array(len(parameters)),
        "adam_steps": np.array(optimizer.steps),
        "adam_learning_rate": np.array(optimizer.learning_rate),
        "adam_m_keep_rate": np.array(optimizer.m_keep_rate),
        "adam_v_keep_rate": np.array(optimizer.v_keep_rate),
        "adam_epsilon": np.array(optimizer.epsilon),
    }

    for index, parameter in enumerate(parameters):
        checkpoint[f"parameter_{index}"] = parameter.data
        checkpoint[f"adam_m_{index}"] = optimizer.m[index]
        checkpoint[f"adam_v_{index}"] = optimizer.v[index]

    path = Path(path)
    with path.open("wb") as checkpoint_file:
        np.savez(checkpoint_file, **checkpoint)


def load_checkpoint(
    path: str | Path,
    model: Module,
    optimizer: Adam,
) -> None:
    parameters = model.parameters()

    with np.load(Path(path), allow_pickle=False) as checkpoint:
        format_version = int(checkpoint["format_version"])
        if format_version != 1:
            raise ValueError(f"Unsupported checkpoint version: {format_version}")

        parameter_count = int(checkpoint["parameter_count"])
        if parameter_count != len(parameters):
            raise ValueError(
                "Checkpoint parameter count does not match the model: "
                f"{parameter_count} != {len(parameters)}"
            )
        if parameter_count != len(optimizer.parameters):
            raise ValueError(
                "Checkpoint parameter count does not match the optimizer: "
                f"{parameter_count} != {len(optimizer.parameters)}"
            )

        saved_parameters = []
        saved_m = []
        saved_v = []

        for index, parameter in enumerate(parameters):
            parameter_data = checkpoint[f"parameter_{index}"]
            adam_m = checkpoint[f"adam_m_{index}"]
            adam_v = checkpoint[f"adam_v_{index}"]

            if parameter_data.shape != parameter.data.shape:
                raise ValueError(
                    f"Parameter {index} shape does not match: "
                    f"{parameter_data.shape} != {parameter.data.shape}"
                )
            if adam_m.shape != optimizer.m[index].shape:
                raise ValueError(
                    f"Adam m state {index} shape does not match: "
                    f"{adam_m.shape} != {optimizer.m[index].shape}"
                )
            if adam_v.shape != optimizer.v[index].shape:
                raise ValueError(
                    f"Adam v state {index} shape does not match: "
                    f"{adam_v.shape} != {optimizer.v[index].shape}"
                )

            saved_parameters.append(parameter_data.copy())
            saved_m.append(adam_m.copy())
            saved_v.append(adam_v.copy())

        # Mutate only after the entire checkpoint has been validated.
        for parameter, saved_data in zip(parameters, saved_parameters):
            parameter.data[...] = saved_data
        for current_m, saved_data in zip(optimizer.m, saved_m):
            current_m[...] = saved_data
        for current_v, saved_data in zip(optimizer.v, saved_v):
            current_v[...] = saved_data

        optimizer.steps = int(checkpoint["adam_steps"])
        optimizer.learning_rate = float(checkpoint["adam_learning_rate"])
        optimizer.m_keep_rate = float(checkpoint["adam_m_keep_rate"])
        optimizer.v_keep_rate = float(checkpoint["adam_v_keep_rate"])
        optimizer.epsilon = float(checkpoint["adam_epsilon"])
