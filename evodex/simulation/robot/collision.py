import pymunk

from .segment import Segment
from .finger import Finger
from .constants import COLLISION_TYPE_ROBOT_SEGMENT


# TODO: Add method to register sensitivity to object type
class RobotCollisionHandler:
    _shape_to_segment_map: dict[pymunk.Shape, Segment]
    _object_types: set[int]

    def __init__(self):
        """Initialize the collision handler."""
        self._shape_to_segment_map = {}
        self._object_types = set()

    def track_segment(self, segment: Segment) -> None:
        """Add a segment to the collision handler."""
        segment.shape.collision_type = COLLISION_TYPE_ROBOT_SEGMENT
        self._shape_to_segment_map[segment.shape] = segment

    def track_finger(self, finger: Finger) -> None:
        """Add a finger to the collision handler."""
        print(f"Registering finger {finger} with segments {len(finger.segments)}")
        for segment in finger.segments:
            self.track_segment(segment)

    def listen(self, object_type: int) -> None:
        """Register a new object type for collision handling."""
        self._object_types.add(object_type)

    def activate(self, space: pymunk.Space) -> None:
        """Register collision handlers for the robot segments."""
        for object_type in self._object_types:
            handler = space.add_collision_handler(
                COLLISION_TYPE_ROBOT_SEGMENT, object_type
            )

            handler.begin = self._on_begin_contact
            handler.separate = self._on_end_contact

    def _on_begin_contact(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict
    ) -> bool:
        """Handle the beginning of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes

        segment = self._shape_to_segment_map.get(
            shape_a
        ) or self._shape_to_segment_map.get(shape_b)

        if segment:
            segment.is_touching = True

        return True

    def _on_end_contact(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: dict
    ) -> None:
        """Handle the end of a contact between a segment and an object."""
        shape_a, shape_b = arbiter.shapes
        segment = self._shape_to_segment_map.get(
            shape_a
        ) or self._shape_to_segment_map.get(shape_b)

        if segment:
            segment.is_touching = False
