from multiprocessing.util import abstract_sockets_supported
from re import S
import pymunk
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from ssl import OP_ENABLE_KTLS
from pydantic import Field, BaseModel
from typing import Generic, TypeVar, Any, Tuple, Optional, Literal, Type, Union

from .utils import COLLISION_TYPE_GRASPING_OBJECT

class ObjectType(str, Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    STAR = "star"
    CUSTOM = "custom"


class ObjectConfig(BaseModel):
    type: str = Field(..., description="Object's type")
    position: Optional[Tuple[float, float]] = Field(default=None, description="Object's initial position")
    mass: float = Field(1.0, description="Object's mass")
    friction: float = Field(0.7, description="Object's friction")
    elasticity: float = Field(0.2, description="Object's elasticity")


class ObjectRegistry:
    _registry: dict[str, Type[ObjectConfig]] = {}

    @classmethod
    def register(cls, object_config: Type[ObjectConfig]) -> None:
        object_type = object_config.model_fields["type"].default
        if not object_type:
            raise ValueError(
                f"Cannot register {object_config.__name__}: "
                "Its config model must have a Literal 'type' with a default value."
            )
        
        cls._registry[object_type] = object_config

    @classmethod
    def any(cls) -> type[ObjectConfig]:
        subclasses = ObjectConfig.__subclasses__()
        return Union[*subclasses] # type: ignore


@ObjectRegistry.register
class CubeConfig(ObjectConfig):
    type: Literal[ObjectType.CUBE] = ObjectType.CUBE
    size: Tuple[float, float] = Field(..., description="The (width, height) of the cube")


@ObjectRegistry.register
class SphereConfig(ObjectConfig):
    type: Literal[ObjectType.SPHERE] = ObjectType.SPHERE
    radius: float = Field(..., description="The radius of the sphere")


@ObjectRegistry.register
class StarConfig(ObjectConfig):
    type: Literal[ObjectType.STAR] = ObjectType.STAR
    outer_radius: float = Field(..., description="The outer radius of the star")
    inner_radius: float = Field(..., description="The inner radius of the star")
    num_points: int = Field(..., description="The number of points of the star")


@ObjectRegistry.register
class CustomConfig(ObjectConfig):
    type: Literal[ObjectType.CUSTOM] = ObjectType.CUSTOM
    vertices: Tuple[Tuple[float, float], ...] = Field(..., description="The vertices of the custom shape")

    
AnyObjectConfig: Type[ObjectConfig] = ObjectRegistry.any()

C = TypeVar("C", bound=ObjectConfig)

class Object(Generic[C], ABC):
    shape: pymunk.Shape
    body: pymunk.Body

    def __init__(self, config: C):
        self.config = config

        self.body, self.shape = self._create()

        self.shape.mass = self.config.mass
        self.shape.friction = self.config.friction
        self.shape.elasticity = self.config.elasticity
        self.shape.collision_type = COLLISION_TYPE_GRASPING_OBJECT

        if self.config.position:
            self.body.position = self.config.position

    @abstractmethod
    def _create(self) -> Tuple[pymunk.Body, pymunk.Shape]:
        pass

    def add_to_space(self, space: pymunk.Space) -> None:
        space.add(self.body, self.shape)

    def remove_from_space(self, space: pymunk.Space) -> None:
        space.remove(self.body, self.shape)

    @property
    def position(self) -> pymunk.Vec2d:
        return self.body.position

    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        self.body.position = pos

    @property
    def velocity(self) -> pymunk.Vec2d:
        return self.body.velocity

    @property
    def angular_velocity(self) -> float:
        return self.body.angular_velocity

class CubeObject(Object[CubeConfig]):
    def _create(self) -> Tuple[pymunk.Body, pymunk.Shape]:        
        moment = pymunk.moment_for_box(self.config.mass, self.config.size)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly.create_box(self.body, self.config.size)
        return body, shape


class SphereObject(Object[SphereConfig]):
    def _create(self):
        moment = pymunk.moment_for_circle(self.config.mass, 0, self.config.radius)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Circle(self.body, self.config.radius)
        return body, shape

class StarObject(Object[StarConfig]):
    def _create(self):        
        vertices = self._generate_star_vertices(self.config.outer_radius, self.config.inner_radius, self.config.num_points)
        moment = pymunk.moment_for_poly(self.config.mass, vertices)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly(self.body, vertices)
        return body, shape
    
    def _generate_star_vertices(self, outer_radius: float, inner_radius: float, num_points: int) -> list[Tuple[float, float]]:
        vertices = []
        angle_step = np.pi / num_points
        for i in range(2 * num_points):
            radius = outer_radius if i % 2 == 0 else inner_radius
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y))
        return vertices


class CustomObject(Object[CustomConfig]):
    def _create(self):        
        moment = pymunk.moment_for_poly(self.config.mass, self.config.vertices)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly(self.body, self.config.vertices)
        return body, shape

class ObjectFactory:
    @staticmethod
    def create(config: ObjectConfig) -> Object:
        if isinstance(config, CubeConfig):
            return CubeObject(config)
        elif isinstance(config, SphereConfig):
            return SphereObject(config)
        elif isinstance(config, StarConfig):
            return StarObject(config)
        elif isinstance(config, CustomConfig):
            return CustomObject(config)
        else:
            raise ValueError(f"Unsupported object config type: {type(config)}")