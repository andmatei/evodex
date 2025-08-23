import pymunk

from typing import Union, Tuple
from pydantic import BaseModel, Field, field_validator


class Reference(BaseModel):
    position: pymunk.Vec2d = Field(
        default=pymunk.Vec2d(0.0, 0.0),
        description="The (x, y) position of the reference point.",
    )
    velocity: pymunk.Vec2d = Field(
        default=pymunk.Vec2d(0.0, 0.0),
        description="The (x, y) velocity of the reference point.",
    )
    angle: float = Field(default=0.0, description="The angle of the reference point.")
    angular_velocity: float = Field(
        default=0.0, description="The angular velocity of the reference point."
    )

    @field_validator("position", "velocity", mode="before")
    @classmethod
    def ensure_vec2d(cls, v: Union[pymunk.Vec2d, Tuple[float, float]]):
        if isinstance(v, tuple) and len(v) == 2:
            return pymunk.Vec2d(*v)
        return v

    @staticmethod
    def from_body(body: pymunk.Body) -> "Reference":
        return Reference(
            position=body.position,
            velocity=body.velocity,
            angle=body.angle,
            angular_velocity=body.angular_velocity,
        )
