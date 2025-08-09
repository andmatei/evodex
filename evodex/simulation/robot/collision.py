import pymunk

from .segment import Segment
from .finger import Finger
from .constants import COLLISION_TYPE_ROBOT_SEGMENT

class SegmentCollisionHandler:
    _shape_to_segment_map: dict[pymunk.Shape, Segment]

    @classmethod
    def add_segment(cls, segment: Segment) -> None:
        """Add a segment to the collision handler."""
        if not hasattr(cls, '_shape_to_segment_map'):
            cls._shape_to_segment_map = {}
        cls._shape_to_segment_map[segment.shape] = segment

    @classmethod
    def add_finger(cls, finger: Finger) -> None:
        """Add a finger to the collision handler."""
        for segment in finger.segments:
            cls.add_segment(segment)

    @classmethod
    def register(cls, space: pymunk.Space) -> None:
        """Register collision handlers for the robot segments."""
        handler = space.add_wildcard_collision_handler(
            COLLISION_TYPE_ROBOT_SEGMENT
        )

        handler.begin = cls._begin_contact
        handler.separate = cls._end_contact

    @classmethod
    def _begin_contact(cls, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Handle the beginning of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes
        segment = cls._shape_to_segment_map.get(shape_a) or cls._shape_to_segment_map.get(shape_b)

        if segment:
            segment.is_touching = True

    @classmethod
    def _end_contact(cls, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict):
        """Handle the end of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes
        segment = cls._shape_to_segment_map.get(shape_a) or cls._shape_to_segment_map.get(shape_b)

        if segment:
            segment.is_touching = False