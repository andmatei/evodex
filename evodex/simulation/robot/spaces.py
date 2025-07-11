from typing import List, Tuple
from pydantic import BaseModel, Field


class BaseAction(BaseModel):
    velocity: Tuple[float, float] = Field(..., description="Base velocity in x and y directions")
    omega: float = Field(..., ge=-1.0, le=1.0)


class Action(BaseModel):
    base: BaseAction
    fingers: List[List[float]]

class ActionScale(BaseModel):
    velocity: Tuple[Tuple[float, float], Tuple[float, float]] = Field(
        ..., description="Scale for base velocity in x and y directions"
    )
    omega: Tuple[float, float] = Field(
        ..., description="Scale for base angular velocity"
    )


class BaseObservation(BaseModel):
    position: Tuple[float, float]
    angle: float
    velocity: Tuple[float, float]
    angular_velocity: float


class SegmentObservation(BaseModel):
    angle: float
    angular_velocity: float


class FingerObservation(BaseModel):
    segments: List[SegmentObservation]
    fingertip_position: Tuple[float, float]


class Observation(BaseModel):
    base: BaseObservation
    fingers: List[FingerObservation]