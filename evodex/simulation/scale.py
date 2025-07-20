from pydantic import BaseModel, Field
from typing import Tuple, List
import numpy as np
from evodex.simulation.robot import RobotConfig, Action
from .utils import NormalizedScale

class ActionScaleConfig(BaseModel):
    velocity: Tuple[NormalizedScale, NormalizedScale] = Field(
        ..., description="Scale for base velocity in x and y directions"
    )
    omega: NormalizedScale = Field(..., description="Scale for base angular velocity")
    motor_rate: NormalizedScale = Field(..., description="Scale for finger motor rates")
    
    
class ActionScaler:
    def __init__(self, scale_config: ActionScaleConfig, robot_config: RobotConfig):
        """
        Initializes the ActionScaler with the provided scale configuration and robot configuration.
        Args:
            scale_config (ActionScale): The action scale configuration.
            robot_config (RobotConfig): The robot configuration.
        """
        self._segment_counts = [len(finger.segments) for finger in robot_config.fingers]
        self._gain, self._offset = self._build_scales(scale_config)

    def _build_scales(self, scale_config: ActionScaleConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the gain and offset arrays for scaling actions.
        Args:
            scale_config (ActionScaleConfig): The action scale configuration.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The gain and offset arrays.
        """
        gain = np.zeros(sum(self._segment_counts) + 3)
        offset = np.zeros_like(gain)

        # Base action scales
        gain[0], offset[0] = scale_config.velocity[0].gain, scale_config.velocity[0].offset
        gain[1], offset[1] = scale_config.velocity[1].gain, scale_config.velocity[1].offset
        gain[2], offset[2] = scale_config.omega.gain, scale_config.omega.offset

        # Finger motor rate scales
        motor_gain, motor_offset = scale_config.motor_rate.gain, scale_config.motor_rate.offset
        gain[3:], offset[3:] = motor_gain, motor_offset

        return gain, offset
    
    def rescale(self, action: Action, normalise: bool = False) -> Action:
        """
        Rescales the action values according to the provided ActionScale.
        Args:
            action (Action): The action to be rescaled.
            normalise (bool): If True, scales from domain to target; otherwise scales from target to domain.
        Returns:
            Action: A new Action instance with rescaled values.
        """
        flat_action = np.array(action.flatten())
        if normalise:
            scaled_action = flat_action * self._gain + self._offset
        else:
            scaled_action = (flat_action - self._offset) / self._gain if self._gain.any() else flat_action

        return Action.unflatten(scaled_action.tolist(), self._segment_counts)


# TODO: Add scaling for Observation