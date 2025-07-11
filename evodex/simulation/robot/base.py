import pymunk
from .constants import (
    CAT_ROBOT_BASE,
    MASK_ROBOT_BASE,
)

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