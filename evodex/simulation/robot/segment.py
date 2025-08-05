import pymunk

from evodex.simulation.robot.spaces import SegmentObservation

from .config import SegmentConfig
from .spaces import SegmentObservation
from .constants import (
    CAT_FINGER_BASE,
    CAT_FINGER_SEGMENT,
    MASK_ALL,
    MASK_FINGER_BASE,
)


class Segment:
    def __init__(
        self,
        position,
        config: SegmentConfig,
        angle=0.0,
        is_fingertip=False,
        is_base=False,
    ):
        self.joint: pymunk.constraints.PivotJoint | None = None

        self.config = config
        self.is_fingertip = is_fingertip
        self.is_base = is_base

        moment = pymunk.moment_for_box(
            self.config.mass, (self.config.length, self.config.width)
        )
        self.body = pymunk.Body(self.config.mass, moment)
        self.body.position = position
        self.body.angle = angle

        self.shape = pymunk.Poly.create_box(
            self.body, (self.config.length, self.config.width)
        )
        self.shape.mass = self.config.mass

    def get_tip_position(self):
        """Get the position of the tip of the segment in world coordinates."""

        local_tip = (self.config.length / 2, 0)
        return self.body.local_to_world(local_tip)

    def remove_from_space(self, space):
        """Remove the segment from the pymunk space."""
        if self.shape in space.shapes:
            space.remove(self.shape)
        if self.body in space.bodies:
            space.remove(self.body)

    def add_to_space(self, space):
        """Add the segment to the pymunk space."""
        space.add(self.body, self.shape)

    def set_filter_group(self, group):
        """Set the collision group for the segment."""
        self.shape.filter = pymunk.ShapeFilter(
            group=group,
            categories=CAT_FINGER_SEGMENT if not self.is_base else CAT_FINGER_BASE,
            mask=MASK_ALL if not self.is_base else MASK_FINGER_BASE,
        )

    def get_observation(self) -> SegmentObservation:
        if self.joint is None:
            raise ValueError("Segment joint is not set. Cannot get observation.")

        relative_angle = self.joint.b.angle - self.joint.a.angle

        return SegmentObservation(
            joint_angle=relative_angle,
            joint_angular_velocity=self.body.angular_velocity,
            position=self.body.position,
            velocity=self.body.velocity,
        )
