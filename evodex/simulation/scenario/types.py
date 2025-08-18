from pydantic import BaseModel, Field
from typing import Tuple
from evodex.simulation.robot.spaces import ExtrinsicObservation as RobotObservation


class ObjectObservation(BaseModel):
    velocity: Tuple[float, float] = Field(
        ..., description="Velocity of the object being observed"
    )
    position: Tuple[float, float] = Field(
        ..., description="Position of the object being observed"
    )
    angle: float = Field(..., description="Angle of the object being observed")
    angular_velocity: float = Field(
        ..., description="Angular velocity of the object being observed"
    )


class Observation(BaseModel):
    object: ObjectObservation
    robot: RobotObservation
