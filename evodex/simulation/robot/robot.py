from .finger import Finger
from .base import Base
from .config import RobotConfig
from .spaces import Action, Observation


class Robot:
    def __init__(self, position, config: RobotConfig):
        self.config = config

        self.base = Base(position, self.config.base)
        self.base.set_finger_count(len(self.config.fingers))

        self.fingers: list[Finger] = []
        for i, finger_config in enumerate(self.config.fingers):
            attach_point = self.base.finger_attachment_points_local[i]
            finger = Finger(
                i,
                self.base.body,
                attach_point,
                finger_config,
            )
            self.fingers.append(finger)

        self.num_motors_per_finger: list[int] = [f.num_segments for f in self.fingers]
        self.total_motors = sum(self.num_motors_per_finger)

    def act(self, action: Action):
        """Apply actions to the robot base and its fingers."""
        # Apply base movement
        self.base.body.velocity = action.base.velocity
        self.base.body.angular_velocity = action.base.omega

        # Apply motor rates to fingers
        if len(action.fingers) != self.total_motors:
            return
        for i, finger_action in enumerate(action.fingers):
            self.fingers[i].act(finger_action)

    def get_observation(self) -> Observation:
        return Observation(
            base=self.base.get_observation(),
            fingers=[finger.get_observation() for finger in self.fingers],
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
