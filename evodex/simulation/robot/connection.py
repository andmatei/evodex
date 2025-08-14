import pymunk
from dataclasses import dataclass


class Connection:
    """A connection between two bodies with a joint and a motor."""

    def __init__(
        self,
        body_a: pymunk.Body,
        body_b: pymunk.Body,
        joint_pos: pymunk.Vec2d,
        angle_limit: tuple[float, float],
    ):
        self.body_a = body_a
        self.body_b = body_b
        self.joint_pos = joint_pos
        self.angle_limit = angle_limit

        self._connect()

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

        self.limit_joint = pymunk.RotaryLimitJoint(
            self.body_a, self.body_b, *self.angle_limit
        )

        self.motor = pymunk.SimpleMotor(self.body_a, self.body_b, 0)
        self.motor.max_force = 10000000  # TODO: Add either in config or as

    @property
    def rate(self) -> float:
        """Get the current motor rate."""
        return self.motor.rate

    @rate.setter
    def rate(self, value: float) -> None:
        """Set the motor rate."""
        self.motor.rate = value

    @property
    def angle(self) -> float:
        """Get the angle of the joint."""
        return self.body_a.angle - self.body_b.angle

    def add_to_space(self, space: pymunk.Space) -> None:
        """Add the connection joint and motor to the pymunk space."""
        space.add(
            self.joint,
            self.motor,
            self.limit_joint,
        )

    def remove_from_space(self, space: pymunk.Space) -> None:
        """Remove the connection joint and motor from the pymunk space."""
        if self.joint in space.constraints:
            space.remove(self.joint)
        if self.motor in space.constraints:
            space.remove(self.motor)
        if self.limit_joint in space.constraints:
            space.remove(self.limit_joint)
