from typing import List, Tuple
from pydantic import BaseModel, Field

from .utils import Scale

class ActionScale(BaseModel):
    velocity: Tuple[Scale, Scale] = Field(
        ..., description="Scale for base velocity in x and y directions"
    )
    omega: Scale = Field(
        ..., description="Scale for base angular velocity"
    )
    motor_rate: Scale = Field(
        ..., description="Scale for finger motor rates"
    )


class BaseAction(BaseModel):
    velocity: Tuple[float, float] = Field(..., description="Base velocity in x and y directions")
    omega: float = Field(..., description="Base angular velocity")


class Action(BaseModel):
    base: BaseAction
    fingers: List[List[float]]

    def scale(self, scale: ActionScale) -> "Action":
        """
        Scales the action values according to the provided ActionScale.
        Args:
            scale (ActionScale): The scaling parameters for the action.
        Returns:
            Action: A new Action instance with scaled values.
        """
        scaled_base = BaseAction(
            velocity=(
                scale.velocity[0].scale(self.base.velocity[0]),
                scale.velocity[1].scale(self.base.velocity[1])
            ),
            omega=scale.omega.scale(self.base.omega)
        )

        scaled_motor_rates = []
        for motor_rates in self.fingers:
            scaled_motor_rates.append([
                scale.motor_rate.scale(rate) for rate in motor_rates
            ])
        
        return Action(base=scaled_base, fingers=scaled_motor_rates)



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