import numpy as np

from typing import List, Tuple
from pydantic import BaseModel, Field

from .utils import Scale
from .config import BaseConfig, RobotConfig

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
    def unflatten(flat_action: List[float], config: RobotConfig) -> 'Action':
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

        num_fingers = len(config.fingers)
        start, end = 0, 0
        fingers = []
        for finger in range(num_fingers):
            end = start + len(config.fingers[finger].segments)
            fingers.append(finger_motor_rates[start:end])
            start = end

        return Action(
            base=BaseAction(velocity=base_velocity, omega=omega),
            fingers=fingers,
        )
    

class ActionScale(BaseModel):
    velocity: Tuple[Scale, Scale] = Field(
        ..., description="Scale for base velocity in x and y directions"
    )
    omega: Scale = Field(..., description="Scale for base angular velocity")
    motor_rate: Scale = Field(..., description="Scale for finger motor rates")

    def __init__(self, config: RobotConfig, **data):
        super().__init__(**data)
        
        self._gain, self._offset = self._build_scales(config)

    def _build_scales(self, config: RobotConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the scales for the action based on the robot configuration.
        Args:
            config (RobotConfig): The robot configuration.
        Returns:
            Tuple[Scale, Scale]: Scales for base velocity and omega.
        """
        total_actions = 2 + sum(len(finger.segments) for finger in config.fingers)

        gain = np.zeros(total_actions)
        offset = np.zeros(total_actions)

        # TODO: This can be moved to the Scale class
        def get_params(scale: Scale) -> Tuple[float, float]:
            """
            Extracts the gain and offset from the scale.
            Args:
                scale (Scale): The scale object.
            Returns:
                Tuple[float, float]: Gain and offset values.
            """
            domain_range = scale.domain[1] - scale.domain[0]
            target_range = scale.target[1] - scale.target[0]
            gain = target_range / domain_range if domain_range != 0 else 1.0
            offset = scale.target[0] - scale.domain[0] * gain

            return gain, offset
        
        gain[0], offset[0] = get_params(self.velocity[0])
        gain[1], offset[1] = get_params(self.velocity[1])
        gain[2], offset[2] = get_params(self.omega)

        motor_gain, motor_offset = get_params(self.motor_rate)
        gain[3:] = motor_gain
        offset[3:] = motor_offset

        return gain, offset

    # TODO: Make this more efficient with numpy
    def rescale(self, action: Action, normalise: bool = False) -> Action:
        """
        Rescales the action values according to the provided ActionScale.
        Args:
            action (Action): The action to be rescaled.
        Returns:
            Action: A new Action instance with rescaled values.
        """
        rescaled_base_velocity = (
            self.velocity[0].rescale(action.base.velocity[0], normalise),
            self.velocity[1].rescale(action.base.velocity[1], normalise),
        )
        rescaled_omega = self.omega.rescale(action.base.omega, normalise)
        rescaled_fingers = [
            [self.motor_rate.rescale(rate, normalise) for rate in finger]
            for finger in action.fingers
        ]

        return Action(
            base=BaseAction(velocity=rescaled_base_velocity, omega=rescaled_omega),
            fingers=rescaled_fingers,
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
