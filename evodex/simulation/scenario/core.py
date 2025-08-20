import pymunk
import pygame
import inspect
import numpy as np

from abc import ABC, abstractmethod
from typing import Annotated, Type, Optional, TypeVar, Generic
from pydantic import BaseModel, Field
from typing import Tuple, List

from .types import Goal, Observation
from .utils import COLLISION_TYPE_GRASPING_OBJECT

from evodex.simulation.robot import Robot, Action


class ScenarioDimensionsConfig(BaseModel):
    width: int = Field(..., description="Screen width")
    height: int = Field(..., description="Screen height")


class ScenarioConfig(BaseModel):
    name: str = Field(..., description="Scenario name")
    screen: ScenarioDimensionsConfig = Field(..., description="Scenario dimenisions")
    robot_start_position: Tuple[float, float] = Field(..., description="Start position")


C = TypeVar("C", bound=ScenarioConfig)


class Scenario(Generic[C], ABC):
    def __init__(self, config: C):
        self.config = config

        self._objects: List[pymunk.Body | pymunk.Shape | pymunk.Constraint] = []
        self._random: np.random.Generator = np.random.default_rng()

    @abstractmethod
    def setup(
        self, space: pymunk.Space, robot: Robot, seed: Optional[int] = None
    ) -> None:
        self._random = np.random.default_rng(seed)

        robot.collision.listen(COLLISION_TYPE_GRASPING_OBJECT)

        robot.add_to_space(space)
        robot.position = self.config.robot_start_position
        robot.angle = np.pi / 2

    @abstractmethod
    def get_reward(self, robot: Robot, action: Action) -> float:
        pass

    @abstractmethod
    def is_terminated(self, robot: Robot) -> bool:
        pass

    @abstractmethod
    def get_observation(self, robot: Robot) -> Observation:
        pass

    @abstractmethod
    def render(self, screen: pygame.Surface) -> None:
        pass

    @abstractmethod
    def get_goal(self, robot: Robot) -> Goal:
        pass

    @abstractmethod
    def get_achieved_goal(self, robot: Robot) -> Goal:
        pass

    def clear_from_space(self, space):
        for item in reversed(self._objects):
            if isinstance(item, pymunk.Body):
                if item in space.bodies:
                    space.remove(item)
            elif isinstance(item, pymunk.Shape):
                if item in space.shapes:
                    space.remove(item)
            elif isinstance(item, pymunk.Constraint):
                if item in space.constraints:
                    space.remove(item)
        self._objects = []


class GroundScenario(Scenario[C], ABC):
    def __init__(self, config: C):
        super().__init__(config)
        self.ground_shape = None

    @abstractmethod
    def setup(
        self, space: pymunk.Space, robot: Robot, seed: Optional[int] = None
    ) -> None:
        super().setup(space, robot, seed)

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

        self._objects.extend([ground_body, ground_shape])


class ScenarioRegistry:
    _scenarios: dict[str, Type[Scenario]] = {}
    _configs: dict[str, Type[ScenarioConfig]] = {}

    @classmethod
    def register(cls, scenario_class: Type[Scenario]) -> Type[Scenario]:
        print("Scenario registered:", scenario_class.__name__)
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
