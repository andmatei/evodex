import numpy as np
import pymunk

from typing import Optional

from .base import Base
from .config import SegmentConfig
from .spaces import FingertipObservation, SegmentObservation
from .connection import Connection
from .utils import Reference
from .constants import (
    CAT_FINGER_BASE,
    CAT_FINGER_SEGMENT,
    MASK_ALL,
    MASK_FINGER_BASE,
)


class Segment:
    def __init__(
        self,
        config: SegmentConfig,
        is_fingertip=False,
        is_base=False,
    ):
        self.connection: Connection | None = None

        self.config = config
        self.is_fingertip = is_fingertip
        self.is_base = is_base
        self.is_touching = False

        moment = pymunk.moment_for_box(
            self.config.mass, (self.config.length, self.config.width)
        )
        self.body = pymunk.Body(self.config.mass, moment)

        self.shape = pymunk.Poly.create_box(
            self.body, (self.config.length, self.config.width)
        )
        self.shape.mass = self.config.mass

    def get_tip_position(self):
        """Get the position of the tip of the segment in world coordinates."""

        local_tip = (self.config.length / 2, 0)
        return self.body.local_to_world(local_tip)

    def get_tip_velocity(self):
        """Get the velocity of the tip of the segment in world coordinates."""

        v_linear = self.body.velocity
        r_local = pymunk.Vec2d(self.config.length / 2, 0)
        v_tangential = (
            r_local.rotated(self.body.angle).perpendicular()
            * self.body.angular_velocity
        )
        v_total = v_linear + v_tangential

        return v_total

    def get_tip_observation(
        self, reference: Optional[Reference] = None
    ) -> FingertipObservation:
        """Get the fingertip observation in world coordinates."""
        if not self.is_fingertip:
            raise ValueError("This segment is not a fingertip.")

        tip_position = self.get_tip_position()
        tip_velocity = self.get_tip_velocity()

        if reference is not None:
            tip_position -= reference.position
            tip_velocity -= reference.velocity

        return FingertipObservation(
            position=tip_position,
            velocity=tip_velocity,
        )

    def remove_from_space(self, space: pymunk.Space):
        """Remove the segment from the pymunk space."""
        if self.shape in space.shapes:
            space.remove(self.shape)
        if self.body in space.bodies:
            space.remove(self.body)

    def add_to_space(self, space: pymunk.Space):
        """Add the segment to the pymunk space."""
        space.add(self.body, self.shape)

    def set_filter_group(self, group: int):
        """Set the collision group for the segment."""
        self.shape.filter = pymunk.ShapeFilter(
            group=group,
            categories=CAT_FINGER_SEGMENT if not self.is_base else CAT_FINGER_BASE,
            mask=MASK_ALL if not self.is_base else MASK_FINGER_BASE,
        )

    def get_observation(self) -> SegmentObservation:
        if self.connection is None:
            raise ValueError("Segment connection is not set. Cannot get observation.")

        return SegmentObservation(
            joint_angle=self.connection.angle,
            joint_angular_velocity=self.body.angular_velocity,
            is_touching=self.is_touching,
        )

    def connect(self, other: "Segment") -> Connection:
        attach_point = other.get_tip_position()
        self.connection = self._connect(other.body, attach_point, other.angle)
        return self.connection

    def attach(self, base: Base, attach_point: pymunk.Vec2d) -> Connection:
        local_attach_point = base.body.world_to_local(attach_point)
        self.connection = self._connect(base.body, local_attach_point, base.angle)
        return self.connection

    def _connect(
        self, other: pymunk.Body, attach_point: pymunk.Vec2d, angle: float
    ) -> Connection:
        pos_x = attach_point.x + self.config.length / 2 * np.cos(angle)
        pos_y = attach_point.y + self.config.length / 2 * np.sin(angle)

        self.position = (pos_x, pos_y)
        self.angle = angle

        return Connection(other, self.body, attach_point, self.config.joint_angle_limit)

    @property
    def is_connected(self) -> bool:
        """Check if the segment is connected to another segment."""
        return self.connection is not None

    @property
    def position(self) -> tuple[float, float]:
        """Get the position of the segment's body."""
        return self.body.position.x, self.body.position.y

    @position.setter
    def position(self, value: tuple[float, float]):
        """Set the position of the segment's body."""
        self.body.position = pymunk.Vec2d(value[0], value[1])

    @property
    def angle(self) -> float:
        """Get the angle of the segment's body."""
        return self.body.angle

    @angle.setter
    def angle(self, value: float):
        """Set the angle of the segment's body."""
        self.body.angle = value
