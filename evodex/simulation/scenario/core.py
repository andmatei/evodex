import pymunk
import pygame
import inspect

from abc import ABC, abstractmethod
from typing import Annotated, Type, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
from typing import Tuple, Union

from .types import Observation

from evodex.simulation.robot import Robot, Action


class ScenarioDimensionsConfig(BaseModel):
    width: int = Field(..., description="Screen width")
    height: int = Field(..., description="Screen height")


class ScenarioConfig(BaseModel):
    name: str = Field(..., description="Scenario name")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    screen: ScenarioDimensionsConfig = Field(..., description="Scenario dimenisions")
    robot_start_position: Tuple[float, float] = Field(..., description="Start position")


C = TypeVar("C", bound=ScenarioConfig)


class Scenario(Generic[C], ABC):
    def __init__(self, config: C):
        self.config = config
        self.objects: list[pymunk.Body | pymunk.Shape | pymunk.Constraint] = []

    @abstractmethod
    def setup(self, space: pymunk.Space, robot: Robot) -> None:
        robot.add_to_space(space)
        robot.position = self.config.robot_start_position

    @abstractmethod
    def get_reward(self, robot: Robot, action: Action) -> float:
        pass

    @abstractmethod
    def is_terminated(self, robot: Robot) -> bool:
        pass

    @abstractmethod
    def get_observation(self, robot) -> Observation:
        pass

    @abstractmethod
    def render(self, screen: pygame.Surface) -> None:
        pass

    def clear_from_space(self, space):
        for item in reversed(self.objects):
            if isinstance(item, pymunk.Body):
                if item in space.bodies:
                    space.remove(item)
            elif isinstance(item, pymunk.Shape):
                if item in space.shapes:
                    space.remove(item)
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
        super().setup(space, robot)

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

        self.objects.extend([ground_body, ground_shape])


class ScenarioRegistry:
    _scenarios: dict[str, Type[Scenario]] = {}
    _configs: dict[str, Type[ScenarioConfig]] = {}

    @classmethod
    def register(cls, scenario_class: Type[Scenario]) -> Type[Scenario]:
        print(f"Registering scenario: {scenario_class.__name__}")

        sig = inspect.signature(scenario_class.__init__)
        config = sig.parameters.get("config", None)

        if config is None or not issubclass(config.annotation, ScenarioConfig):
            raise ValueError(
                f"Scenario class '{scenario_class.__name__}' must have a 'config' parameter of type ScenarioConfig."
            )

        config_class = config.annotation
        scenario_name = config_class.model_fields["name"].default
        if not scenario_name:
            raise ValueError(
                f"Cannot register {scenario_class.__name__}: "
                "Its config model must have a Literal 'name' with a default value."
            )

        cls._scenarios[scenario_name] = scenario_class
        cls._configs[scenario_name] = config_class
        return scenario_class

    @classmethod
    def parse_config(cls, config: dict) -> ScenarioConfig:
        """
        Parses a scenario configuration dictionary and returns the corresponding Scenario instance.

        Args:
            config (dict): The scenario configuration dictionary.

        Returns:
            Scenario: An instance of the registered Scenario class.
        """
        scenario_name = config.get("name")
        if scenario_name is None:
            raise ValueError("Scenario configuration must contain a 'name' field.")

        ConfigClass = cls._configs.get(scenario_name)
        if ConfigClass is None:
            raise ValueError(f"Scenario config class '{scenario_name}' not registered.")

        return ConfigClass(**config)

    @classmethod
    def load(cls, config: dict) -> Scenario:
        scenario_name = config.get("name")
        if scenario_name is None:
            raise ValueError("Scenario configuration must contain a 'name' field.")

        ConfigClass = cls._configs.get(scenario_name)
        if ConfigClass is None:
            raise ValueError(f"Scenario config class '{scenario_name}' not registered.")

        ScenarioClass = cls._scenarios.get(scenario_name)
        if ScenarioClass is None:
            raise ValueError(f"Scenario class '{scenario_name}' not registered.")

        scenario_config = ConfigClass(**config)
        return ScenarioClass(config=scenario_config)
