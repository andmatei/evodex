import numpy as np
import pymunk

from typing import List, Optional

from .spaces import (
    ExtrinsicFingerObservation,
    FingertipObservation,
    IntrinsicFingerObservation,
)
from .config import FingerConfig
from .segment import Segment
from .connection import Connection
from .constants import FINGER_GROUP_START
from .base import Base


class Finger:
    def __init__(
        self, index: int, base: Base, attach_point: pymunk.Vec2d, config: FingerConfig
    ):
        self.segments: List[Segment] = []
        self.connections: List[Connection] = []
        self.index = index
        self.config = config

        self.num_segments = len(config.segments)
        self._build(base, attach_point)

    def _build(self, base: Base, attach_point: pymunk.Vec2d):
        """Build the finger segments and joints based on the configuration."""
        for i, segment_config in enumerate(self.config.segments):
            is_fingertip = i == self.num_segments - 1
            is_base = i == 0

            segment = Segment(
                segment_config,
                is_fingertip=is_fingertip,
                is_base=is_base,
            )
            segment.set_filter_group(self.index + FINGER_GROUP_START)
            connection = (
                segment.attach(base, attach_point)
                if is_base
                else segment.connect(self.segments[-1])
            )

            self.segments.append(segment)
            self.connections.append(connection)

    def act(self, rates: List[float]):
        if len(rates) != len(self.connections):
            raise ValueError(
                f"Action rates length {len(rates)} does not match finger length {len(self.connections)}."
            )
        for connection, rate in zip(self.connections, rates):
            connection.motor.rate = rate

    def get_intrinsic_observation(self) -> IntrinsicFingerObservation:
        return IntrinsicFingerObservation(
            segments=tuple(segment.get_observation() for segment in self.segments)
        )

    def get_extrinsic_observation(
        self, reference_frame: Optional[pymunk.Body] = None
    ) -> ExtrinsicFingerObservation:
        return ExtrinsicFingerObservation(
            tip=self.tip.get_tip_observation(reference_frame),
        )

    def remove_from_space(self, space):
        """Remove the finger and its segments from the pymunk space."""
        for segment in self.segments:
            segment.remove_from_space(space)
        for connection in self.connections:
            connection.remove_from_space(space)

    def add_to_space(self, space):
        """Add the finger and its segments to the pymunk space."""
        for segment in self.segments:
            segment.add_to_space(space)
        for connection in self.connections:
            connection.add_to_space(space)

    @property
    def tip(self) -> Segment:
        return self.segments[-1]
