import numpy as np
import pymunk

from typing import List

from evodex.simulation.robot.spaces import FingerObservation
from .config import FingerConfig
from .segment import Segment
from .constants import FINGER_GROUP_START


class Finger:
    def __init__(self, index, base, attach_point, config: FingerConfig):
        self.segments: List[Segment] = []
        self.joints: List[pymunk.constraints.PivotJoint] = []
        self.motors: List[pymunk.SimpleMotor] = []
        self.index = index
        self.config = config

        self.num_segments = len(config.segments)
        self._build(base, attach_point)

    def _build(self, base, attach_point):
        """Build the finger segments and joints based on the configuration."""
        prev_body = base
        current_attach_point = base.local_to_world(attach_point)
        for i, segment_config in enumerate(self.config.segments):
            is_fingertip = i == self.num_segments - 1
            if_base = i == 0

            if i == 0:
                initial_angle = base.angle
                segment_pos_x = current_attach_point[0] + (
                    segment_config.length / 2
                ) * np.cos(initial_angle)
                segment_pos_y = current_attach_point[1] + (
                    segment_config.length / 2
                ) * np.sin(initial_angle)
            else:
                prev_segment_tip = self.segments[-1].get_tip_position()
                initial_angle = self.segments[-1].body.angle
                segment_pos_x = prev_segment_tip[0] + (
                    segment_config.length / 2
                ) * np.cos(initial_angle)
                segment_pos_y = prev_segment_tip[1] + (
                    segment_config.length / 2
                ) * np.sin(initial_angle)
            segment = Segment(
                (segment_pos_x, segment_pos_y),
                segment_config,
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

            min_angle_rel = segment_config.joint_angle_limit[0]
            max_angle_rel = segment_config.joint_angle_limit[1]
            limit_joint = pymunk.RotaryLimitJoint(
                prev_body, segment.body, min_angle_rel, max_angle_rel
            )
            self.joints.extend([joint, limit_joint])

            motor = pymunk.SimpleMotor(prev_body, segment.body, 0)
            self.motors.append(motor)

            prev_body = segment.body
            current_attach_point = segment.get_tip_position()

    def act(self, rates: List[float]):
        if len(rates) != len(self.motors):
            return
        for motor, rate in zip(self.motors, rates):
            motor.rate = rate

    def get_observation(self):
        segments_obs = [segment.get_observation() for segment in self.segments]
        fingertip_obs = self.get_fingertip_position()
        return FingerObservation(
            segments=segments_obs,
            fingertip_position=fingertip_obs,
        )

    def get_fingertip_position(self):
        return self.segments[-1].get_tip_position()

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
