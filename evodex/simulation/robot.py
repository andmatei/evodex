import math
import numpy as np
import pymunk
from evodex.simulation.utils import (
    COLLISION_TYPE_GROUND,
    COLLISION_TYPE_ROBOT_BASE,
    COLLISION_TYPE_ROBOT_SEGMENT_START,
    COLLISION_TYPE_SCENARIO_OBJECT_START,
    COLLISION_TYPE_SCENARIO_STATIC_START,
)


class Segment:
    def __init__(
        self,
        space,
        position,
        length,
        width,
        angle=0.0,
        is_fingertip=False,
        tip_config=None,
        mass=1.0,
        collision_type=COLLISION_TYPE_ROBOT_SEGMENT_START,
    ):
        self.length = length
        self.width = width
        self.is_fingertip = is_fingertip
        moment = pymunk.moment_for_box(mass, (length, width))
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.angle = angle
        if is_fingertip and tip_config:
            if tip_config.get("shape") == "circle":
                radius = tip_config.get("radius", width / 2)
                offset = (length / 2, 0)
                self.shape = pymunk.Circle(self.body, radius, offset)
            elif tip_config.get("shape") == "rectangle":
                tip_size = tip_config.get("size", (length * 0.5, width))
                vertices = [
                    (-tip_size[0] / 2 + length / 2, -tip_size[1] / 2),
                    (tip_size[0] / 2 + length / 2, -tip_size[1] / 2),
                    (tip_size[0] / 2 + length / 2, tip_size[1] / 2),
                    (-tip_size[0] / 2 + length / 2, tip_size[1] / 2),
                ]
                self.shape = pymunk.Poly(self.body, vertices)
            else:
                self.shape = pymunk.Poly.create_box(self.body, (length, width))
        else:
            self.shape = pymunk.Poly.create_box(self.body, (length, width))
        self.shape.mass = mass
        self.shape.elasticity = 0.3
        self.shape.friction = 0.8
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)

    def get_tip_position(self):
        local_tip = (self.length / 2, 0)
        return self.body.local_to_world(local_tip)


class Finger:
    def __init__(
        self,
        space,
        base_body,
        base_attach_point,
        finger_config,
        finger_index,
        robot_segment_collision_id_counter,
    ):
        self.space = space
        self.segments = []
        self.joints = []
        self.motors = []
        self.finger_config = finger_config
        self.num_segments = finger_config["num_segments"]
        self.collision_id_counter = robot_segment_collision_id_counter
        prev_body = base_body
        current_attach_point_world = base_body.local_to_world(base_attach_point)
        for i in range(self.num_segments):
            segment_length = finger_config["segment_length"]
            segment_width = finger_config["segment_width"]
            is_fingertip = i == self.num_segments - 1
            tip_config = (
                {
                    "shape": finger_config.get("fingertip_shape", "circle"),
                    "radius": finger_config.get("fingertip_radius", segment_width / 2),
                    "size": finger_config.get(
                        "fingertip_size", (segment_length * 0.5, segment_width)
                    ),
                }
                if is_fingertip
                else None
            )
            if i == 0:
                initial_angle = base_body.angle
                segment_pos_x = current_attach_point_world[0] + (
                    segment_length / 2
                ) * math.cos(initial_angle)
                segment_pos_y = current_attach_point_world[1] + (
                    segment_length / 2
                ) * math.sin(initial_angle)
            else:
                prev_segment_tip = self.segments[-1].get_tip_position()
                initial_angle = self.segments[-1].body.angle
                segment_pos_x = prev_segment_tip[0] + (segment_length / 2) * math.cos(
                    initial_angle
                )
                segment_pos_y = prev_segment_tip[1] + (segment_length / 2) * math.sin(
                    initial_angle
                )
            current_collision_type = next(self.collision_id_counter)
            segment = Segment(
                space,
                (segment_pos_x, segment_pos_y),
                segment_length,
                segment_width,
                angle=initial_angle,
                is_fingertip=is_fingertip,
                tip_config=tip_config,
                collision_type=current_collision_type,
            )
            self.segments.append(segment)
            joint_pos_world = (
                current_attach_point_world
                if i == 0
                else self.segments[-2].get_tip_position()
            )
            anchor_a_local_for_joint = prev_body.world_to_local(joint_pos_world)
            anchor_b_local_for_joint = segment.body.world_to_local(joint_pos_world)
            joint = pymunk.PivotJoint(
                prev_body,
                segment.body,
                anchor_a_local_for_joint,
                anchor_b_local_for_joint,
            )
            min_angle_rel = finger_config["joint_angle_limit_min"]
            max_angle_rel = finger_config["joint_angle_limit_max"]
            limit_joint = pymunk.RotaryLimitJoint(
                prev_body, segment.body, min_angle_rel, max_angle_rel
            )
            space.add(joint, limit_joint)
            self.joints.extend([joint, limit_joint])
            motor = pymunk.SimpleMotor(prev_body, segment.body, 0)
            motor.max_force = finger_config.get("motor_max_force", 5e7)
            space.add(motor)
            self.motors.append(motor)
            prev_body = segment.body
            current_attach_point_world = segment.get_tip_position()

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


class Base:
    def __init__(self, space, config, collision_type=COLLISION_TYPE_ROBOT_BASE):
        self.width = config["width"]
        self.height = config["height"]
        self.is_kinematic = config.get("kinematic", False)
        self.initial_position = config.get("initial_position", (100, 100))
        if self.is_kinematic:
            self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            shape_mass = 10.0
        else:
            mass = 10.0
            moment = pymunk.moment_for_box(mass, (self.width, self.height))
            self.body = pymunk.Body(mass, moment)
            shape_mass = mass
        self.body.position = self.initial_position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.shape = pymunk.Poly.create_box(self.body, (self.width, self.height))
        if self.is_kinematic:
            self.shape.mass = shape_mass
        else:
            self.shape.mass = shape_mass
        self.shape.elasticity = 0.2
        self.shape.friction = 0.9
        self.shape.collision_type = collision_type
        space.add(self.body, self.shape)
        self.finger_attachment_points_local = []

    def set_finger_attachment_points(self, num_fingers):
        self.finger_attachment_points_local = []
        if num_fingers == 0:
            return
        for i in range(num_fingers):
            local_x = self.width / 2
            vertical_spacing = self.height / (num_fingers + 1)
            local_y = -self.height / 2 + (i + 1) * vertical_spacing
            self.finger_attachment_points_local.append((local_x, local_y))


class Robot:
    def __init__(self, space, config):
        self.space = space
        self.config = config
        self.base_config = config["base"]
        self.base = Base(space, self.base_config)
        self.base.set_finger_attachment_points(len(config["fingers"]))
        self.fingers = []
        num_total_segments = sum(f["num_segments"] for f in config["fingers"])
        self.robot_segment_collision_id_counter = (
            COLLISION_TYPE_ROBOT_SEGMENT_START + i for i in range(num_total_segments)
        )
        for i, f_config in enumerate(config["fingers"]):
            if i < len(self.base.finger_attachment_points_local):
                attach_point = self.base.finger_attachment_points_local[i]
                finger = Finger(
                    space,
                    self.base.body,
                    attach_point,
                    f_config,
                    i,
                    self.robot_segment_collision_id_counter,
                )
                self.fingers.append(finger)
        self.num_motors_per_finger = [f.num_segments for f in self.fingers]
        self.total_motors = sum(self.num_motors_per_finger)

    def apply_actions(self, actions):
        if len(actions) != self.total_motors:
            return
        action_idx = 0
        for finger, num_motors in zip(self.fingers, self.num_motors_per_finger):
            finger.set_motor_rates(actions[action_idx : action_idx + num_motors])
            action_idx += num_motors

    def get_observation(self):
        obs = [
            self.base.body.position.x,
            self.base.body.position.y,
            self.base.body.angle,
            self.base.body.velocity.x,
            self.base.body.velocity.y,
            self.base.body.angular_velocity,
        ]
        for finger in self.fingers:
            obs.extend(finger.get_joint_angles())
            fingertip_pos = finger.get_fingertip_position()
            obs.extend([fingertip_pos.x, fingertip_pos.y] if fingertip_pos else [0, 0])
        return np.array(obs, dtype=np.float32)

    def remove_from_space(self, space):
        items_to_remove = [(self.base.body, self.base.shape)] + [
            (seg.body, seg.shape) for f in self.fingers for seg in f.segments
        ]
        for body, shape in items_to_remove:
            if shape in space.shapes:
                space.remove(shape)
            if body in space.bodies:
                space.remove(body)
        constraints_to_remove = [c for f in self.fingers for c in f.joints + f.motors]
        for constraint in constraints_to_remove:
            if constraint in space.constraints:
                space.remove(constraint)
