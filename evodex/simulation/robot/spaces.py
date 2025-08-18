import numpy as np

from typing import List, Tuple
from pydantic import BaseModel, Field


class BaseAction(BaseModel):
    velocity: Tuple[float, float] = Field(
        ..., description="Base velocity in x and y directions"
    )
    omega: float = Field(..., description="Base angular velocity")


FingerAction = List[float]


class Action(BaseModel):
    base: BaseAction
    fingers: List[FingerAction]


class BaseObservation(BaseModel):
    position: Tuple[float, float]
    angle: float
    velocity: Tuple[float, float]
    angular_velocity: float


class SegmentObservation(BaseModel):
    joint_angle: float
    joint_angular_velocity: float

    is_touching: bool


class FingertipObservation(BaseModel):
    position: Tuple[float, float]
    velocity: Tuple[float, float]


class ExtrinsicFingerObservation(BaseModel):
    tip: FingertipObservation


IntrinsicFingerObservation = List[SegmentObservation]


class IntrinsicObservation(BaseModel):
    fingers: List[IntrinsicFingerObservation]


class ExtrinsicObservation(BaseModel):
    base: BaseObservation
    fingertips: List[ExtrinsicFingerObservation]
