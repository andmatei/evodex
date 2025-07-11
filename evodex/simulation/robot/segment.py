import pymunk

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
        length,
        width,
        angle=0.0,
        is_fingertip=False,
        is_base=False,
        mass=1.0,
    ):
        self.length = length
        self.width = width
        self.is_fingertip = is_fingertip
        self.is_base = is_base

        moment = pymunk.moment_for_box(mass, (length, width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.angle = angle

        self.shape = pymunk.Poly.create_box(self.body, (length, width))
        self.shape.mass = mass
        self.shape.elasticity = 0.3
        self.shape.friction = 0.8

    def get_tip_position(self):
        """Get the position of the tip of the segment in world coordinates."""

        local_tip = (self.length / 2, 0)
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