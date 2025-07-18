import numpy as np
import pymunk
import pygame

from pydantic import Field
from typing import Tuple, Optional

from .core import GroundScenario, ScenarioRegistry, ScenarioConfig
from .utils import COLLISION_TYPE_SCENARIO_OBJECT_START, pymunk_to_pygame_coord
from .types import Observation

from evodex.simulation.robot import Robot, Action


class MoveCubeToTargetScenarioConfig(ScenarioConfig):
    target_pos: Optional[Tuple[float, float]] = Field(
        None, description="Target position"
    )
    success_radius: float = Field(..., description="Success radius")
    cube_size: Tuple[float, float] = Field(..., description="Cube size")
    cube_initial_pos: Optional[Tuple[float, float]] = Field(
        None, description="Cube initial position"
    )


@ScenarioRegistry.register
class MoveCubeToTargetScenario(GroundScenario[MoveCubeToTargetScenarioConfig]):
    """
    Scenario where a cube is moved to a target position.
    The cube is initialized at a random position above the ground.
    The target position is also initialized randomly within the screen bounds.
    The scenario is considered successful when the cube is within a certain radius of the target position.
    """

    def __init__(self, config: MoveCubeToTargetScenarioConfig):
        super().__init__(config)
        self.cube_body: Optional[pymunk.Body] = None
        self.cube_shape: Optional[pymunk.Shape] = None

        if self.config.target_pos is None:
            self.target_pos = np.random.uniform(
                low=[0, 0],
                high=[self.config.screen.width, self.config.screen.height],
            )
        else:
            self.target_pos = np.array(self.config.target_pos)

        if self.config.cube_initial_pos is not None:
            self.cube_initial_pos = np.array(self.config.cube_initial_pos)
        else:
            # Default initial position for the cube
            self.cube_initial_pos = np.array([
                np.random.uniform(
                    low=self.config.cube_size[0] / 2,
                    high=self.config.screen.width - self.config.cube_size[0] / 2,
                ),
                self.config.cube_size[1] / 2 + 10,  # Slightly above the ground
            ])

    def setup(self, space: pymunk.Space, robot: Robot) -> None:
        super().setup(space, robot)

        mass = 1.0
        moment = pymunk.moment_for_box(mass, self.config.cube_size)

        self.cube_body = pymunk.Body(mass, moment)
        self.cube_body.position = self.cube_initial_pos
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, self.config.cube_size)
        self.cube_shape.friction = 0.7
        self.cube_shape.elasticity = 0.3
        self.cube_shape.collision_type = COLLISION_TYPE_SCENARIO_OBJECT_START + 1
        space.add(self.cube_body, self.cube_shape)

        self.objects.append(
            {
                "body": self.cube_body,
                "shape": self.cube_shape,
            }
        )

    def get_reward(self, robot: Robot, action: Action) -> float:
        reward = 0.0
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            dist_to_target = np.linalg.norm(cube_pos - self.target_pos)
            reward -= float(dist_to_target * 0.01)
            if dist_to_target < self.config.success_radius:
                reward += 100.0

        # TODO: Add action penalty
        # action_penalty = np.sum(np.square(action)) * 0.001
        # reward -= action_penalty
        return reward

    def is_terminated(self, robot: Robot, current_step: int, max_steps: int) -> bool:
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            if np.linalg.norm(cube_pos - self.target_pos) < self.config.success_radius:
                print("MoveCubeScenario: Target reached!")
                return True
        return False

    def get_observation(self, robot: Robot) -> Observation:
        if not self.cube_body:
            raise ValueError("Scenario is not initialized.")

        return Observation(
            velocity=self.cube_body.velocity.length,
            position=(self.cube_body.position.x, self.cube_body.position.y),
            angle=self.cube_body.angle,
            angular_velocity=self.cube_body.angular_velocity,
        )

    def render(self, screen):
        target_center_pygame = pymunk_to_pygame_coord(
            self.target_pos, self.config.screen.height
        )
        pygame.draw.circle(
            screen,
            pygame.Color("lightgreen"),
            target_center_pygame,
            self.success_threshold,
            2,
        )
