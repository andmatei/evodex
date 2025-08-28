import numpy as np
import pymunk
import pygame

from pydantic import Field
from typing import Tuple, Optional, Literal

from .core import GroundScenario, ScenarioRegistry, ScenarioConfig
from .types import Goal, Observation, ObjectObservation
from .object import AnyObjectConfig, ObjectConfig, ObjectFactory

from evodex.simulation.robot import Robot
from evodex.simulation.robot.utils import Reference


# TODO: Add the goal in the scenario config
class MoveObjectToTargetScenarioConfig(ScenarioConfig):
    name: Literal["move_cube_to_target"] = "move_cube_to_target"
    target_pos: Optional[Tuple[float, float]] = Field(
        default=None, description="Target position"
    )
    success_radius: float = Field(..., description="Success radius")
    object: AnyObjectConfig = Field(..., description="The object to be moved") # type: ignore


@ScenarioRegistry.register
class MoveObjectToTargetScenario(GroundScenario[MoveObjectToTargetScenarioConfig]):
    """
    Scenario where a cube is moved to a target position.
    The cube is initialized at a random position above the ground.
    The target position is also initialized randomly within the screen bounds.
    The scenario is considered successful when the cube is within a certain radius of the target position.
    """

    def __init__(self, config: MoveObjectToTargetScenarioConfig):
        super().__init__(config)
        self.cube_body: Optional[pymunk.Body] = None
        self.cube_shape: Optional[pymunk.Shape] = None

    def setup(
        self, space: pymunk.Space, robot: Robot, seed: Optional[int] = None
    ) -> None:
        super().setup(space, robot, seed)

        if self.config.target_pos is None:
            # Random target position within the screen bounds
            self.target_pos = self._random.uniform(
                low=[0, 0],
                high=[self.config.screen.width, self.config.screen.height],
            ).tolist()
        else:
            self.target_pos = self.config.target_pos


        if not isinstance(self.config.object, ObjectConfig):
            raise ValueError("Invalid object configuration.")

        if self.config.object.position is not None:
            self.object_position = self.config.object.position
        else:
            # Random initial position for the cube
            self.object_position = np.array(
                [
                    np.random.uniform(
                        low=20,
                        high=self.config.screen.width - 20,
                    ),
                    20,  # Slightly above the ground
                ]
            ).tolist()


        self.object = ObjectFactory.create(self.config.object)
        self.object.add_to_space(space)

    def is_terminated(self, robot: Robot) -> bool:
        if self.object:
            object_pos = np.array([self.object.position.x, self.object.position.y])
            if np.linalg.norm(object_pos - self.target_pos) < self.config.success_radius:
                return True
        return False

    def get_observation(self, robot: Robot) -> Observation:
        if not self.object:
            raise ValueError("Scenario is not initialized.")

        return Observation(
            object=ObjectObservation(
                position=(self.object.position.x, self.object.position.y),
                velocity=(self.object.velocity.x, self.object.velocity.y),
                angle=self.object.angle,
                angular_velocity=self.object.angular_velocity,
                size=self.config.cube_size,
            ),
            robot=robot.get_extrinsic_observation(Reference.from_body(self.cube_body)),
        )

    def get_goal(self, robot: Robot) -> Goal:
        """Get the target position as the goal."""
        return Goal(
            position=self.target_pos,
            velocity=(0, 0),  # No velocity goal for static target
            angle=0.0,  # No angle goal for static target
            angular_velocity=0.0,  # No angular velocity goal for static target
        )

    def get_achieved_goal(self, robot: Robot) -> Goal:
        """Get the current position of the cube as the achieved goal."""
        if not self.cube_body:
            raise ValueError("Scenario is not initialized.")

        return Goal(
            position=(self.cube_body.position.x, self.cube_body.position.y),
            velocity=(self.cube_body.velocity.x, self.cube_body.velocity.y),
            angle=self.cube_body.angle,
            angular_velocity=self.cube_body.angular_velocity,
        )

    def render(self, screen):
        pygame.draw.circle(
            screen,
            pygame.Color("lightgreen"),
            self.target_pos,
            self.config.success_radius,
            2,
        )
