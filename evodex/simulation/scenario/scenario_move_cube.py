import numpy as np
import pymunk
import pygame

from .core import GroundScenario, ScenarioRegistry
from .utils import COLLISION_TYPE_SCENARIO_OBJECT_START, pymunk_to_pygame_coord


class MoveCubeToTargetScenario(GroundScenario):
    def __init__(self, **config):
        super().__init__(**config)
        self.cube_body = None
        self.cube_shape = None

        if "target_pos" in config:
            self.target_pos = np.array(config["target_pos"])
        else:
            # Initializing target position randomly within the screen bounds
            self.target_pos = np.random.uniform(
                low=[0, 0],
                high=[self.screen_width, self.screen_height],
            )

        self.success_threshold = config.get("success_radius", 20)
        self.cube_size = config.get("cube_size", (20, 20))

        if "cube_initial_pos" in config:
            self.cube_initial_pos = config["cube_initial_pos"]
        else:
            # Default initial position for the cube
            self.cube_initial_pos = (
                np.random.uniform(
                    low=self.cube_size[0] / 2,
                    high=self.screen_width - self.cube_size[0] / 2,
                ),
                self.cube_size[1] / 2 + 10,  # Slightly above the ground
            )

    def setup(self, space, robot):
        super().setup(space, robot)

        mass = 1.0
        moment = pymunk.moment_for_box(mass, self.cube_size)

        self.cube_body = pymunk.Body(mass, moment)
        self.cube_body.position = self.cube_initial_pos
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, self.cube_size)
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

        return self.objects

    # TODO: Do we need observation?
    def get_reward(self, robot, action, observation):
        reward = 0.0
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            dist_to_target = np.linalg.norm(cube_pos - self.target_pos)
            reward -= dist_to_target * 0.01
            if dist_to_target < self.success_threshold:
                reward += 100.0
        action_penalty = np.sum(np.square(action)) * 0.001
        reward -= action_penalty
        return reward

    # TODO: Do we need observation?
    def is_terminated(self, robot, observation, current_step, max_steps):
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            if np.linalg.norm(cube_pos - self.target_pos) < self.success_threshold:
                print("MoveCubeScenario: Target reached!")
                return True
        return False

    def get_observation(self, robot):
        obs = []
        if self.cube_body:
            obs.extend(
                [
                    self.cube_body.position.x,
                    self.cube_body.position.y,
                    self.cube_body.angle,
                ]
            )
        else:
            obs.extend([0.0] * 3)
        return np.array(obs, dtype=np.float32)

    def get_goal(self):
        return self.target_pos

    def render(self, screen):
        target_center_pygame = pymunk_to_pygame_coord(
            self.target_pos, self.screen_height
        )
        pygame.draw.circle(
            screen,
            pygame.Color("lightgreen"),
            target_center_pygame,
            self.success_threshold,
            2,
        )


# Register the scenario
ScenarioRegistry.register(MoveCubeToTargetScenario)
