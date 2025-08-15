import pymunk

from typing import List, Tuple
from .config import BaseConfig
from .constants import (
    CAT_ROBOT_BASE,
    MASK_ROBOT_BASE,
)
from evodex.simulation.robot.spaces import BaseObservation


class Base:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.is_touching = False

        moment = pymunk.moment_for_box(
            self.config.mass, (self.config.width, self.config.height)
        )
        self.body = pymunk.Body(
            mass=self.config.mass, moment=moment, body_type=pymunk.Body.KINEMATIC
        )

        self.body.position = (0, 0)
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.shape = pymunk.Poly.create_box(
            self.body, (self.config.width, self.config.height)
        )

        self.shape.filter = pymunk.ShapeFilter(
            group=0,
            categories=CAT_ROBOT_BASE,
            mask=MASK_ROBOT_BASE,
        )

        self.finger_attachment_points_local: List[pymunk.Vec2d] = []

    @property
    def angle(self) -> float:
        return self.body.angle

    @angle.setter
    def angle(self, angle: float) -> None:
        self.body.angle = angle

    @property
    def position(self) -> Tuple[float, float]:
        return self.body.position

    @position.setter
    def position(self, pos: Tuple[float, float]) -> None:
        self.body.position = pos

    def set_finger_count(self, num_fingers):
        self.finger_attachment_points_local = []
        if num_fingers == 0:
            return
        for i in range(num_fingers):
            local_x = self.config.width / 2
            vertical_spacing = (
                self.config.height / (num_fingers - 1)
                if num_fingers > 1
                else self.config.width
            )
            local_y = (
                -self.config.height / 2 + i * vertical_spacing if num_fingers > 1 else 0
            )
            self.finger_attachment_points_local.append(pymunk.Vec2d(local_x, local_y))

    def remove_from_space(self, space):
        """Remove the base from the pymunk space."""

        if self.shape in space.shapes:
            space.remove(self.shape)
        if self.body in space.bodies:
            space.remove(self.body)

    def add_to_space(self, space):
        """Add the base to the pymunk space."""
        space.add(self.body, self.shape)

    def get_observation(self) -> BaseObservation:
        return BaseObservation(
            position=self.body.position,
            velocity=self.body.velocity,
            angle=self.body.angle,
            angular_velocity=self.body.angular_velocity,
        )
