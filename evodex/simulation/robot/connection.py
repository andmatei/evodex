import pymunk
from dataclasses import dataclass


class Connection:
    body_a: pymunk.Body
    body_b: pymunk.Body

    def __init__(
        self, body_a: pymunk.Body, body_b: pymunk.Body, joint_pos: pymunk.Vec2d
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.joint_pos = joint_pos

    def _connect(self) -> None:
        """Connect the two bodies with a pivot joint."""
        anchor_a_local_for_joint = self.body_a.world_to_local(self.joint_pos)
        anchor_b_local_for_joint = self.body_b.world_to_local(self.joint_pos)

        self.joint = pymunk.constraints.PivotJoint(
            self.body_a,
            self.body_b,
            anchor_a_local_for_joint,
            anchor_b_local_for_joint,
        )

        # TODO: Add rotary limit joint if needed

        self.motor = pymunk.SimpleMotor(self.body_a, self.body_b, 0)
        self.motor.max_force = 10000000  # TODO: Add either in config or as

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add the connection joint and motor to the pymunk space."""
        space.add(self.joint, self.motor)

    def remove_from_space(self, space: pymunk.Space) -> None:
        """Remove the connection joint and motor from the pymunk space."""
        if self.joint in space.constraints:
            space.remove(self.joint)
        if self.motor in space.constraints:
            space.remove(self.motor)
