from .finger import Finger
from .base import Base
from .config import RobotConfig

class Robot:
    def __init__(self, config):
        self.config = config

        self.base_config = config["base"]
        self.base = Base(self.base_config)
        self.base.set_finger_count(len(config["fingers"]))

        self.fingers = []
        for i, f_config in enumerate(config["fingers"]):
            attach_point = self.base.finger_attachment_points_local[i]
            finger = Finger(
                i,
                self.base.body,
                attach_point,
                f_config,
            )
            self.fingers.append(finger)

        self.num_motors_per_finger = [f.num_segments for f in self.fingers]
        self.total_motors = sum(self.num_motors_per_finger)

    def apply_actions(self, vx, vy, omega, motor_rates):
        """Apply actions to the robot base and its fingers."""
        # Apply base movement
        self.base.body.velocity = vx, vy
        self.base.body.angular_velocity = omega

        # Apply motor rates to fingers
        if len(motor_rates) != self.total_motors:
            return
        action_idx = 0
        for finger, num_motors in zip(self.fingers, self.num_motors_per_finger):
            finger.set_motor_rates(motor_rates[action_idx : action_idx + num_motors])
            action_idx += num_motors

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
        for finger in self.fingers:
            finger_obs = finger.get_joint_angles()
            fingertip_pos = finger.get_fingertip_position()
            finger_obs.extend(
                [fingertip_pos.x, fingertip_pos.y] if fingertip_pos else [0, 0]
            )
            obs["fingers"].append(finger_obs)

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
