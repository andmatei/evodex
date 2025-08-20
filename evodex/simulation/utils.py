import numpy as np
import gymnasium as gym

from pydantic import BaseModel

from typing import Tuple, Any
from pydantic import BaseModel, Field


class Kinematics(BaseModel):
    """
    Base class for kinematic properties of the robot.
    This class can be extended to include specific kinematic data.
    """

    position: Tuple[float, float] = Field(
        ..., description="Position of the robot in the environment"
    )
    angle: float = Field(..., description="Orientation angle of the robot")
    velocity: Tuple[float, float] = Field(
        ..., description="Linear velocity of the robot"
    )
    angular_velocity: float = Field(..., description="Angular velocity of the robot")


# TODO: Add function to build the observation space from a sample observation (pydantic model)
# TODO: Add a reverse function for observation (problem with np array with one element to scalar)
def to_observation(value: Any) -> Any:
    """
    Convert a value to a gymnasium observation format.
    This function can be extended to handle different types of observations.
    """
    if isinstance(value, BaseModel):
        return to_observation(value.model_dump())

    if isinstance(value, dict):
        return {key: to_observation(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        try:
            return np.array(value, dtype=np.float32)
        except (TypeError, ValueError):
            return tuple(to_observation(item) for item in value)

    if isinstance(value, bool):
        return np.array([int(value)], dtype=np.int8)

    if isinstance(value, (int, float)):
        return np.float32(value)

    return value
