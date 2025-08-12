import pymunk

from .segment import Segment
from .finger import Finger
from .constants import COLLISION_TYPE_ROBOT_SEGMENT


# TODO: Fix collision
class SegmentCollisionHandler:
    _shape_to_segment_map: dict[pymunk.Shape, Segment]

    def __init__(self):
        """Initialize the collision handler."""
        self._shape_to_segment_map = {}

    def add_segment(self, segment: Segment) -> None:
        """Add a segment to the collision handler."""
        segment.shape.collision_type = COLLISION_TYPE_ROBOT_SEGMENT
        self._shape_to_segment_map[segment.shape] = segment

    def add_finger(self, finger: Finger) -> None:
        """Add a finger to the collision handler."""
        print(f"Registering finger {finger} with segments {len(finger.segments)}")
        for segment in finger.segments:
            self.add_segment(segment)

    def register(self, space: pymunk.Space) -> None:
        """Register collision handlers for the robot segments."""
        handler = space.add_wildcard_collision_handler(COLLISION_TYPE_ROBOT_SEGMENT)

        handler.begin = self._begin_contact
        handler.separate = self._end_contact

    def _begin_contact(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict
    ) -> bool:
        """Handle the beginning of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes

        if (
            shape_a.collision_type == COLLISION_TYPE_ROBOT_SEGMENT
            and shape_b.collision_type == COLLISION_TYPE_ROBOT_SEGMENT
        ):
            return True

        segment = self._shape_to_segment_map.get(
            shape_a
        ) or self._shape_to_segment_map.get(shape_b)

        if segment:
            segment.is_touching = True

        return True

    def _end_contact(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict
    ) -> None:
        """Handle the end of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes
        segment = self._shape_to_segment_map.get(
            shape_a
        ) or self._shape_to_segment_map.get(shape_b)

        if (
            shape_a.collision_type == COLLISION_TYPE_ROBOT_SEGMENT
            and shape_b.collision_type == COLLISION_TYPE_ROBOT_SEGMENT
        ):
            return

        if segment:
            segment.is_touching = False
