from .finger import Finger
from .base import Base
from .config import RobotConfig
from .spaces import Action, Observation


class Robot:
    def __init__(self, config: RobotConfig):
        self.config = config

        self.base = Base(self.config.base)
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
        # TODO: check if this is correct
        if len(action.fingers) != self.total_motors:
            return
        for i, finger_action in enumerate(action.fingers):
            self.fingers[i].act(finger_action)

    def get_observation(self):
        obs = {
            "base": [
                self.base.body.position.x,
                self.base.body.position.y,
                self.base.body.angle,
                self.base.body.velocity.x,
                self.base.body.velocity.y,
                self.base.body.angular_velocity,
            ],
            "fingers": [],
        }

        return obs

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

    def get_observation_space(self):
        """Get the observation space of the robot."""
        base_obs_size = 6
        finger_obs_size = sum(
            f.num_segments + 2
            for f in self.fingers  # +2 for fingertip position
        )

        return base_obs_size + finger_obs_size

    def get_action_space(self):
        """Get the action space of the robot."""
        base_action_size = 3
        finger_action_size = sum(f.num_segments for f in self.fingers)
        return base_action_size + finger_action_size
