from matplotlib.mlab import angle_spectrum
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


class Goal(BaseModel):
    position: Tuple[float, float] = Field(
        ..., description="Target position for the robot to reach"
    )
    angle: float = Field(..., description="Target angle for the robot to achieve")
    angular_velocity: float = Field(
        ..., description="Target angular velocity for the robot to achieve"
    )
    velocity: Tuple[float, float] = Field(
        ..., description="Target velocity for the robot to achieve"
    )


class Observation(BaseModel):
    object: ObjectObservation
    robot: RobotObservation
