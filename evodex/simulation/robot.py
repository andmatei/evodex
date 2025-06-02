import numpy as np
import pymunk

FINGER_GROUP_START = 1

CAT_ROBOT_BASE = 1 << 0
CAT_FINGER_BASE = 1 << 1
CAT_FINGER_SEGMENT = 1 << 2
CAT_SCENARIO_OBJECT = 1 << 3

MASK_ROBOT_BASE = CAT_SCENARIO_OBJECT | CAT_FINGER_SEGMENT
MASK_FINGER_BASE = CAT_SCENARIO_OBJECT | CAT_FINGER_SEGMENT | CAT_FINGER_BASE
MASK_ALL = CAT_SCENARIO_OBJECT | CAT_ROBOT_BASE | CAT_FINGER_BASE | CAT_FINGER_SEGMENT


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


class Finger:
    def __init__(self, index, base, attach_point, config):
        self.segments = []
        self.joints = []
        self.motors = []
        self.index = index
        self.config = config

        self.num_segments = config["num_segments"]
        self._build(base, attach_point)

    def _build(self, base, attach_point):
        """Build the finger segments and joints based on the configuration."""
        prev_body = base
        current_attach_point = base.local_to_world(attach_point)
        for i in range(self.num_segments):
            segment_length = (
                self.config["segment_lengths"][i]
                if "segment_lengths" in self.config
                else self.config["segment_length"]
            )
            segment_width = (
                self.config["segment_widths"][i]
                if "segment_widths" in self.config
                else self.config["segment_width"]
            )
            is_fingertip = i == self.num_segments - 1
            if_base = i == 0

            if i == 0:
                initial_angle = base.angle
                segment_pos_x = current_attach_point[0] + (segment_length / 2) * np.cos(
                    initial_angle
                )
                segment_pos_y = current_attach_point[1] + (segment_length / 2) * np.sin(
                    initial_angle
                )
            else:
                prev_segment_tip = self.segments[-1].get_tip_position()
                initial_angle = self.segments[-1].body.angle
                segment_pos_x = prev_segment_tip[0] + (segment_length / 2) * np.cos(
                    initial_angle
                )
                segment_pos_y = prev_segment_tip[1] + (segment_length / 2) * np.sin(
                    initial_angle
                )
            segment = Segment(
                (segment_pos_x, segment_pos_y),
                segment_length,
                segment_width,
                angle=initial_angle,
                is_fingertip=is_fingertip,
                is_base=if_base,
            )
            segment.set_filter_group(self.index + FINGER_GROUP_START)
            self.segments.append(segment)

            joint_pos_world = (
                current_attach_point if i == 0 else self.segments[-2].get_tip_position()
            )

            anchor_a_local_for_joint = prev_body.world_to_local(joint_pos_world)
            anchor_b_local_for_joint = segment.body.world_to_local(joint_pos_world)

            joint = pymunk.constraints.PivotJoint(
                prev_body,
                segment.body,
                anchor_a_local_for_joint,
                anchor_b_local_for_joint,
            )

            min_angle_rel = self.config["joint_angle_limit_min"]
            max_angle_rel = self.config["joint_angle_limit_max"]
            limit_joint = pymunk.RotaryLimitJoint(
                prev_body, segment.body, min_angle_rel, max_angle_rel
            )
            self.joints.extend([joint, limit_joint])

            motor = pymunk.SimpleMotor(prev_body, segment.body, 0)
            motor.max_force = self.config.get("motor_max_force", 5e7)
            self.motors.append(motor)

            prev_body = segment.body
            current_attach_point = segment.get_tip_position()

    def set_motor_rates(self, rates):
        if len(rates) != len(self.motors):
            return
        for motor, rate in zip(self.motors, rates):
            motor.rate = rate

    def get_joint_angles(self):
        angles = []
        for motor_idx, motor in enumerate(self.motors):
            angles.append(motor.b.angle - motor.a.angle)
        return angles

    def get_fingertip_position(self):
        return self.segments[-1].get_tip_position() if self.segments else None

    def remove_from_space(self, space):
        """Remove the finger and its segments from the pymunk space."""
        for segment in self.segments:
            segment.remove_from_space(space)
        for joint in self.joints:
            if joint in space.constraints:
                space.remove(joint)
        for motor in self.motors:
            if motor in space.constraints:
                space.remove(motor)

    def add_to_space(self, space):
        """Add the finger and its segments to the pymunk space."""
        for segment in self.segments:
            space.add(segment.body, segment.shape)
        for joint in self.joints:
            space.add(joint)
        for motor in self.motors:
            space.add(motor)


class Base:
    def __init__(self, config):
        self.width = config["width"]
        self.height = config["height"]
        self.initial_position = config.get("initial_position", (100, 100))

        mass = config.get("mass", 1.0)
        moment = pymunk.moment_for_box(mass, (self.width, self.height))
        self.body = pymunk.Body(
            mass=mass, moment=moment, body_type=pymunk.Body.KINEMATIC
        )

        self.body.position = self.initial_position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))

        self.shape.elasticity = 0.1
        self.shape.friction = 0.9
        self.shape.filter = pymunk.ShapeFilter(
            group=0,
            categories=CAT_ROBOT_BASE,
            mask=MASK_ROBOT_BASE,
        )

        self.finger_attachment_points_local = []

    def set_finger_count(self, num_fingers):
        self.finger_attachment_points_local = []
        if num_fingers == 0:
            return
        for i in range(num_fingers):
            local_x = self.width / 2
            vertical_spacing = (
                self.height / (num_fingers - 1) if num_fingers > 1 else self.width
            )
            local_y = -self.height / 2 + i * vertical_spacing if num_fingers > 1 else 0
            self.finger_attachment_points_local.append((local_x, local_y))

    def remove_from_space(self, space):
        """Remove the base from the pymunk space."""

        if self.shape in space.shapes:
            space.remove(self.shape)
        if self.body in space.bodies:
            space.remove(self.body)

    def add_to_space(self, space):
        """Add the base to the pymunk space."""
        space.add(self.body, self.shape)


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
