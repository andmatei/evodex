import pymunk
import pygame

from abc import ABC, abstractmethod
from typing import Type, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
from typing import Tuple

from .types import Observation

from evodex.simulation.robot import Robot, Action


class ScreenConfig(BaseModel):
    width: int = Field(..., description="Screen width")
    height: int = Field(..., description="Screen height")


class ScenarioConfig(BaseModel):
    name: str = Field(..., description="Scenario name")
    screen: ScreenConfig = Field(..., description="Screen config")
    robot_start_position: Tuple[float, float] = Field(..., description="Start position")


C = TypeVar("C", bound=ScenarioConfig)


class Scenario(Generic[C], ABC):
    def __init__(self, config: C):
        self.config = config
        self.objects: list[dict] = []

    @abstractmethod
    def setup(self, space: pymunk.Space, robot: Robot) -> None:
        pass

    @abstractmethod
    def get_reward(self, robot: Robot, action: Action) -> float:
        pass

    @abstractmethod
    def is_terminated(self, robot: Robot, current_step: int, max_steps: int) -> bool:
        pass

    @abstractmethod
    def get_observation(self, robot) -> Observation:
        pass

    @abstractmethod
    def render(self, screen: pygame.Surface) -> None:
        pass

    # TODO: Move this functionality to the simulation class that takes care of the steps
    def is_truncated(self, robot, observation, current_step, max_steps) -> bool:
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


class GroundScenario(Scenario[C], ABC):
    def __init__(self, config: C):
        super().__init__(config)
        self.ground_shape = None

    @abstractmethod
    def setup(self, space: pymunk.Space, robot: Robot) -> None:
        # Create a static ground segment
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(
            ground_body,
            (0, self.config.screen.height - 10),
            (self.config.screen.width, self.config.screen.height - 10),
            5,
        )
        ground_shape.friction = 1.0
        space.add(ground_body, ground_shape)

        # TODO: Add a schema to add only this types of dictionary
        self.objects.append(
            {
                "body": ground_body,
                "shape": ground_shape,
            }
        )


class ScenarioRegistry:
    _registry: dict[str, Type[Scenario]] = {}

    def __init__(self):
        raise RuntimeError(
            "ScenarioRegistry is a class for managing scenario classes, not meant to be instantiated."
        )

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ScenarioRegistry, cls).__new__(cls)
        return cls.instance

    @classmethod
    def register(cls, scenario_class: Type[Scenario]) -> Type[Scenario]:
        cls._registry[scenario_class.__name__] = scenario_class
        return scenario_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[Scenario]]:
        return cls._registry.get(name, None)

    @classmethod
    def create(cls, name: str, config: ScenarioConfig) -> Scenario:
        scenario_class = cls.get(name)
        if scenario_class is None:
            raise ValueError(f"Scenario class '{name}' not found in registry.")
        return scenario_class(config)
