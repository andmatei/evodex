from abc import ABC, abstractmethod
import numpy as np
import pymunk
import pygame
from evodex.simulation.utils import (
    COLLISION_TYPE_SCENARIO_OBJECT_START,
    pymunk_to_pygame_coord,
)


class Scenario(ABC):
    def __init__(self, **config):
        self.config = config
        self.screen_width = config["screen_width"]
        self.screen_height = config["screen_height"]
        self.objects = []

    @abstractmethod
    def setup(self, space, robot):
        pass

    @abstractmethod
    def get_reward(self, robot, action, observation):
        pass

    @abstractmethod
    def is_terminated(self, robot, observation, current_step, max_steps):
        pass

    @abstractmethod
    def get_observation(self, robot) -> np.ndarray:
        pass

    @abstractmethod
    def render(self, screen):
        pass

    def is_truncated(self, robot, observation, current_step, max_steps):
        return current_step >= max_steps

    def clear_from_space(self, space):
        for item in reversed(self.objects):
            if isinstance(item, dict) and "body" in item and "shape" in item:
                if item["shape"] in space.shapes:
                    space.remove(item["shape"])
                if item["body"] in space.bodies:
                    if item["body"] is not space.static_body:
                        space.remove(item["body"])
            elif isinstance(item, pymunk.Constraint):
                if item in space.constraints:
                    space.remove(item)
        self.objects = []


class GroundScenario(Scenario):
    def __init__(self, **sim_config):
        super().__init__(**sim_config)
        self.ground_shape = None

    @abstractmethod
    def setup(self, space, robot):
        # Create a static ground segment
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(
            ground_body,
            (0, self.screen_height - 10),
            (self.screen_width, self.screen_height - 10),
            5,
        )
        ground_shape.friction = 1.0
        space.add(ground_body, ground_shape)

        self.objects.append(
            {
                "body": ground_body,
                "shape": ground_shape,
            }
        )


# --- Concrete Scenarios ---
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


class ScenarioRegistry:
    _registry = {}

    def __init__(self):
        raise RuntimeError(
            "ScenarioRegistry is a class for managing scenario classes, not meant to be instantiated."
        )

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ScenarioRegistry, cls).__new__(cls)
        return cls.instance

    @classmethod
    def register(cls, scenario_class):
        cls._registry[scenario_class.__name__] = scenario_class

    @classmethod
    def get(cls, name):
        return cls._registry.get(name, None)

    @classmethod
    def create(cls, name, **kwargs) -> Scenario:
        scenario_class = cls.get(name)
        if scenario_class is None:
            raise ValueError(f"Scenario class '{name}' not found in registry.")
        return scenario_class(**kwargs)


# Register the scenarios
ScenarioRegistry.register(MoveCubeToTargetScenario)
