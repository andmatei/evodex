from abc import ABC, abstractmethod
import numpy as np
import pymunk


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
