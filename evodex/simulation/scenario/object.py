import pymunk
import numpy as np

from abc import ABC, abstractmethod
from enum import Enum
from pydantic import Field, BaseModel
from typing import Generic, List, TypeVar, Tuple, Optional, Literal, Type, Union, get_origin, get_args

from .utils import COLLISION_TYPE_GRASPING_OBJECT

class ObjectType(str, Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    POLYGON = "polygon"
    STAR = "star"
    CUSTOM = "custom"


class ObjectConfig(BaseModel):
    type: str = Field(..., description="Object's type")
    position: Optional[Tuple[float, float]] = Field(default=None, description="Object's initial position")
    mass: float = Field(1.0, description="Object's mass")
    friction: float = Field(0.7, description="Object's friction")
    elasticity: float = Field(0.2, description="Object's elasticity")

class CubeConfig(ObjectConfig):
    type: Literal[ObjectType.CUBE] = ObjectType.CUBE
    size: Tuple[float, float] = Field(..., description="The (width, height) of the cube")

class SphereConfig(ObjectConfig):
    type: Literal[ObjectType.SPHERE] = ObjectType.SPHERE
    radius: float = Field(..., description="The radius of the sphere")

class PolygonConfig(ObjectConfig):
    type: Literal[ObjectType.POLYGON] = ObjectType.POLYGON
    radius: float = Field(..., description="The radius of the polygon")
    num_sides: int = Field(..., description="The number of sides of the polygon")

class StarConfig(ObjectConfig):
    type: Literal[ObjectType.STAR] = ObjectType.STAR
    outer_radius: float = Field(..., description="The outer radius of the star")
    inner_radius: float = Field(..., description="The inner radius of the star")
    num_points: int = Field(..., description="The number of points of the star")

class CustomConfig(ObjectConfig):
    type: Literal[ObjectType.CUSTOM] = ObjectType.CUSTOM
    vertices: Tuple[Tuple[float, float], ...] = Field(..., description="The vertices of the custom shape")

C = TypeVar("C", bound=ObjectConfig)

class Object(Generic[C], ABC):
    shape: pymunk.Shape
    body: pymunk.Body

    def __init__(self, config: C):
        self.config = config

        self.body, self.shapes = self._create()

        for shape in self.shapes:
            shape.mass = self.config.mass
            shape.friction = self.config.friction
            shape.elasticity = self.config.elasticity
            shape.collision_type = COLLISION_TYPE_GRASPING_OBJECT

        if self.config.position:
            self.body.position = self.config.position

    @abstractmethod
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
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
    
    @property
    def angle(self) -> float:
        return self.body.angle
    

class ObjectRegistry:
    _objects: dict[str, Type[Object]] = {}
    _configs: dict[str, Type[ObjectConfig]] = {}

    @classmethod
    def register(cls, object_class: Type[Object[C]]) -> Type[Object[C]]:
        config_class: Optional[Type[ObjectConfig]] = None
        # Look through the base classes for the generic Object[...] type
        for base in getattr(object_class, '__orig_bases__', []):
            if get_origin(base) is Object:
                config_class = get_args(base)[0]
                break
        
        if config_class is None:
            raise TypeError(
                f"Cannot register {object_class.__name__}: "
                "It must inherit from Object[SomeConfig]."
            )

        object_type = config_class.model_fields["type"].default
        if not object_type:
            raise ValueError(
                f"Cannot register {config_class.__name__}: "
                "Its config model must have a Literal 'type' with a default value."
            )
        
        cls._objects[object_type] = object_class
        cls._configs[object_type] = config_class
        return object_class
    
    @classmethod
    def create(cls, config: ObjectConfig) -> Object:
        object_class = cls._objects.get(config.type)
        if object_class is None:
            raise ValueError(f"Unknown object type: {config.type}")
        return object_class(config)

    @classmethod
    def any(cls) -> type[ObjectConfig]:
        subclasses = ObjectConfig.__subclasses__()
        return Union[*subclasses] # type: ignore


AnyObjectConfig: Type[ObjectConfig] = ObjectRegistry.any()


@ObjectRegistry.register
class CubeObject(Object[CubeConfig]):
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:        
        moment = pymunk.moment_for_box(self.config.mass, self.config.size)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly.create_box(body, self.config.size)
        return body, [shape]


@ObjectRegistry.register
class SphereObject(Object[SphereConfig]):
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
        moment = pymunk.moment_for_circle(self.config.mass, 0, self.config.radius)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Circle(body, self.config.radius)
        return body, [shape]


@ObjectRegistry.register
class PolygonObject(Object[PolygonConfig]):
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
        vertices = self._generate_polygon_vertices(self.config.num_sides, self.config.radius)
        moment = pymunk.moment_for_poly(self.config.mass, vertices)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly(body, vertices)
        return body, [shape]

    def _generate_polygon_vertices(self, num_sides: int, radius: float) -> list[Tuple[float, float]]:
        vertices = []
        angle_step = 2 * np.pi / num_sides
        for i in range(num_sides):
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y))
        return vertices


@ObjectRegistry.register
class StarObject(Object[StarConfig]):
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
        # Create a single body for the entire star
        moment = pymunk.moment_for_poly(self.config.mass, []) # Moment is complex, can be approximated
        body = pymunk.Body(self.config.mass, moment)
        
        # Generate the vertices for the star
        vertices = self._generate_star_vertices(
            self.config.outer_radius, 
            self.config.inner_radius, 
            self.config.num_points
        )

        shapes: list[pymunk.Shape] = []
        center_point = (0, 0)
        num_vertices = len(vertices)

        # Create the star from triangles, each sharing the center point
        for i in range(num_vertices):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % num_vertices] # The next vertex, wrapping around
            
            # Each triangle is a convex polygon
            triangle_verts = [center_point, p1, p2]
            
            # Create a shape for this triangle and add it to our list
            triangle_shape = pymunk.Poly(body, triangle_verts)
            shapes.append(triangle_shape)
            
        return body, shapes
    
    def _generate_star_vertices(self, outer_radius: float, inner_radius: float, num_points: int) -> list[Tuple[float, float]]:
        vertices = []
        # Correct angle step for a full circle
        angle_step = np.pi / num_points 
        for i in range(2 * num_points):
            radius = outer_radius if i % 2 == 0 else inner_radius
            angle = i * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append((x, y))
        return vertices


@ObjectRegistry.register
class CustomObject(Object[CustomConfig]):
    def _create(self) -> Tuple[pymunk.Body, List[pymunk.Shape]]:
        moment = pymunk.moment_for_poly(self.config.mass, self.config.vertices)
        body = pymunk.Body(self.config.mass, moment)
        
        shape = pymunk.Poly(body, self.config.vertices)
        return body, [shape]