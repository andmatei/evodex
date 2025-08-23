from typing import List, Optional

from .collision import RobotCollisionHandler
from .finger import Finger
from .base import Base
from .config import RobotConfig
from .spaces import Action, ExtrinsicObservation, IntrinsicObservation
from .utils import Reference


class Robot:
    def __init__(self, config: RobotConfig):
        self.config = config

        self.collision = RobotCollisionHandler()

        self.base = Base(self.config.base)
        self.base.set_finger_count(len(self.config.fingers))
        self.collision.track_base(self.base)

        self.fingers: List[Finger] = []
        for i, finger_config in enumerate(self.config.fingers):
            attach_point = self.base.finger_attachment_points_local[i]
            finger = Finger(
                i,
                self.base,
                attach_point,
                finger_config,
            )
            self.fingers.append(finger)

            # Register the finger segments for collision handling
            self.collision.track_finger(finger)

    @property
    def angle(self) -> float:
        """Get the robot's base rotation angle."""
        return self.base.angle

    @angle.setter
    def angle(self, angle: float) -> None:
        """Set the robot's base rotation angle."""
        self.base.angle = angle

    @property
    def position(self) -> tuple[float, float]:
        return self.base.position

    @position.setter
    def position(self, pos: tuple[float, float]) -> None:
        """Set the robot's base position."""
        self.base.position = pos

    def act(self, action: Action):
        """Apply actions to the robot base and its fingers."""
        # Apply base movement
        self.base.body.velocity = action.base.velocity
        self.base.body.angular_velocity = action.base.omega

        # Apply motor rates to fingers
        if len(action.fingers) != len(self.fingers):
            raise ValueError(
                f"Action fingers length {len(action.fingers)} does not match robot fingers count {len(self.fingers)}."
            )
        for i, finger_action in enumerate(action.fingers):
            try:
                self.fingers[i].act(finger_action)
            except ValueError as e:
                raise ValueError(f"Error in finger {i} action: {str(e)}") from e

    def get_intrinsic_observation(self) -> IntrinsicObservation:
        return IntrinsicObservation(
            fingers=tuple(finger.get_intrinsic_observation() for finger in self.fingers)
        )

    def get_extrinsic_observation(
        self, reference: Optional[Reference] = None
    ) -> ExtrinsicObservation:
        return ExtrinsicObservation(
            base=self.base.get_observation(),
            fingers=tuple(
                finger.get_extrinsic_observation(reference) for finger in self.fingers
            ),
        )

    def remove_from_space(self, space):
        """Remove the robot and its components from the pymunk space."""
        self.base.remove_from_space(space)
        for finger in self.fingers:
            finger.remove_from_space(space)

    def add_to_space(self, space):
        """Add the robot and its components to the pymunk space."""
        self.base.add_to_space(space)
        for finger in self.fingers:
            finger.add_to_space(space)

        # Register collision handlers
        self.collision.activate(space)
