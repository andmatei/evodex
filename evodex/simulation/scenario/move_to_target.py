import pymunk
import pygame
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

        self.previous_distance: Optional[float] = None

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

        # --- Weights for reward components (tune these) ---
        PROGRESS_WEIGHT = 1.0
        SUCCESS_BONUS = 250.0
        DISTANCE_PENALTY = 0.1

        # --- 1. Calculate Progress Reward ---
        current_distance = np.linalg.norm(
            np.array(robot.position) - np.array(self.target_position)
        )

        reward = 0.0
        if self.previous_distance is not None:
            # Reward is the reduction in distance
            reward += PROGRESS_WEIGHT * float(self.previous_distance - current_distance)

        self.previous_distance = float(current_distance)

        # --- 2. Add Success Bonus ---
        if current_distance < 10.0:  # Your success radius
            print("🎉 Success!")
            reward += SUCCESS_BONUS

        # --- 3. Small penalty for being far from the target ---
        reward -= DISTANCE_PENALTY * float(current_distance)

        return float(reward)

    def is_terminated(self, robot: Robot) -> bool:
        """
        The task is considered complete when the robot is within a certain distance of the target position.
        """
        distance_to_target = np.linalg.norm(
            np.array(robot.position) - np.array(self.target_position)
        )
        return float(distance_to_target) < 10.0

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

    # TODO: Work only with observations from the robot
    def get_achieved_goal(self, robot: Robot):
        """Get the current position of the robot as the achieved goal."""
        return Goal(
            position=robot.position,
            velocity=robot.base.body.velocity,
            angle=robot.base.body.angle,
            angular_velocity=robot.base.body.angular_velocity,
        )

    def render(self, screen):
        pygame.draw.circle(
            screen,
            pygame.Color("lightgreen"),
            self.target_position,
            10,
            2,
        )
