from typing import Tuple
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
