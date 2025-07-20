from sys import monitoring
import numpy as np

from typing import List, Tuple
from pydantic import BaseModel, Field

class BaseAction(BaseModel):
    velocity: Tuple[float, float] = Field(
        ..., description="Base velocity in x and y directions"
    )
    omega: float = Field(..., description="Base angular velocity")


class Action(BaseModel):
    base: BaseAction
    fingers: List[List[float]]
    
    def flatten(self) -> np.ndarray:
        """
        Flattens the Action into a list of floats.
        Returns:
            List[float]: A flattened list of action values.
        """
        base_velocity = list(self.base.velocity)
        omega = [self.base.omega]
        finger_motor_rates = [rate for finger in self.fingers for rate in finger]

        return np.array(base_velocity + omega + finger_motor_rates).astype(float).tolist()
    
    @staticmethod
    def unflatten(flat_action: List[float], segments: List[int]) -> 'Action':
        """
        Converts a flattened list of action values back into an Action object.
        Args:
            flat_action (List[float]): A flattened list of action values.
        Returns:
            Action: An Action object constructed from the flattened values.
        """
        base_velocity = (flat_action[0], flat_action[1])
        omega = flat_action[2]
        finger_motor_rates = flat_action[3:]

        num_fingers = len(segments)
        start, end = 0, 0
        fingers = []
        for finger in range(num_fingers):
            end = start + segments[finger]
            fingers.append(finger_motor_rates[start:end])
            start = end

        return Action(
            base=BaseAction(velocity=base_velocity, omega=omega),
            fingers=fingers,
        )
    

class BaseObservation(BaseModel):
    position: Tuple[float, float]
    angle: float
    velocity: Tuple[float, float]
    angular_velocity: float


class SegmentObservation(BaseModel):
    joint_angle: float
    join_velocity: float

    position: Tuple[float, float]
    velocity: Tuple[float, float]


class FingerObservation(BaseModel):
    segments: List[SegmentObservation]
    fingertip_position: Tuple[float, float]


class Observation(BaseModel):
    base: BaseObservation
    fingers: List[FingerObservation]
