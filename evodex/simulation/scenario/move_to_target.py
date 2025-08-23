import pymunk
import numpy as np

from typing import Optional, Tuple, Literal
from pydantic import Field

from .core import GroundScenario, ScenarioConfig, ScenarioRegistry
from .types import Goal, Observation, ObjectObservation

from evodex.simulation.robot import Robot, Action
from evodex.simulation.robot.utils import Reference


class MoveToTarget(ScenarioConfig):
    name: Literal["move_to_target"] = "move_to_target"
    target_position: Optional[Tuple[float, float]] = Field(default=None)


@ScenarioRegistry.register
class MoveToTargetScenario(GroundScenario[MoveToTarget]):
    def __init__(self, config: MoveToTarget):
        super().__init__(config)

    def setup(
        self, space: pymunk.Space, robot: Robot, seed: Optional[int] = None
    ) -> None:
        super().setup(space, robot, seed)

        if self.config.target_position is None:
            # Random target position within the screen bounds
            self.target_position = self._random.uniform(
                low=[0, 0],
                high=[self.config.screen.width, self.config.screen.height],
            ).tolist()
        else:
            self.target_position = self.config.target_position

    def get_reward(self, robot: Robot, action: Action) -> float:
        """
        Calculates a dense, shaped reward for the move to target task.
        This is the standard method for goal-conditioned environments, especially with HER.
        """

        # Calculate the distance to the target position
        distance_to_target = np.linalg.norm(
            np.array(robot.position) - np.array(self.target_position)
        )

        # Reward is inversely proportional to the distance
        reward = -distance_to_target

        # Optionally, you can add a small constant to avoid negative rewards
        return float(reward)

    def is_terminated(self, robot: Robot) -> bool:
        """
        The task is considered complete when the robot is within a certain distance of the target position.
        """
        distance_to_target = np.linalg.norm(
            np.array(robot.position) - np.array(self.target_position)
        )
        return float(distance_to_target) < 1.0

    def get_observation(self, robot: Robot) -> Observation:
        return Observation(
            object=ObjectObservation(
                position=self.target_position,
                velocity=(0, 0),  # No velocity for static target
                angle=0.0,  # No angle for static target
                angular_velocity=0.0,  # No angular velocity for static target
                size=(0, 0),
            ),
            robot=robot.get_extrinsic_observation(
                Reference(position=self.target_position)
            ),
        )

    def get_goal(self, robot: Robot) -> Goal:
        """Get the target position as the goal."""
        return Goal(
            position=self.target_position,
            velocity=(0, 0),  # No velocity goal for static target
            angle=0.0,  # No angle goal for static target
            angular_velocity=0.0,  # No angular velocity goal for static target
        )
