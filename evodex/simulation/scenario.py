from abc import ABC, abstractmethod
import numpy as np
import pymunk
from evodex.simulation.utils import (
    COLLISION_TYPE_SCENARIO_OBJECT_START,
    COLLISION_TYPE_SCENARIO_STATIC_START,
)


class Scenario(ABC):
    def __init__(self, sim_config):
        self.sim_config = sim_config
        self.screen_width = sim_config["screen_width"]
        self.screen_height = sim_config["screen_height"]
        self.pymunk_objects = []

    @abstractmethod
    def setup(self, space):
        pass

    @abstractmethod
    def get_reward(self, robot, action, observation):
        pass

    @abstractmethod
    def is_terminated(self, robot, observation, current_step, max_steps):
        pass

    def is_truncated(self, robot, observation, current_step, max_steps):
        return current_step >= max_steps

    @abstractmethod
    def get_scenario_observation(self, robot):
        return np.array([], dtype=np.float32)

    def clear_from_space(self, space):
        for item in reversed(self.pymunk_objects):
            if isinstance(item, dict) and "body" in item and "shape" in item:
                if item["shape"] in space.shapes:
                    space.remove(item["shape"])
                if item["body"] in space.bodies:
                    if item["body"] is not space.static_body:
                        space.remove(item["body"])
            elif isinstance(item, pymunk.Constraint):
                if item in space.constraints:
                    space.remove(item)
        self.pymunk_objects = []

    def reset_objects(self):
        for obj_info in self.pymunk_objects:
            if (
                isinstance(obj_info, dict)
                and "initial_pos" in obj_info
                and obj_info.get("body")
                and obj_info["body"].body_type == pymunk.Body.DYNAMIC
            ):
                obj_info["body"].position = obj_info["initial_pos"]
                obj_info["body"].velocity = (0, 0)
                obj_info["body"].angular_velocity = 0
                obj_info["body"].angle = 0


# --- Concrete Scenarios ---
class MoveCubeToTargetScenario(Scenario):
    def __init__(self, sim_config):
        super().__init__(sim_config)
        self.cube_body = None
        self.cube_shape = None
        self.target_pos = np.array(
            [sim_config["screen_width"] * 0.75, sim_config["screen_height"] * 0.75]
        )
        self.target_radius = 30
        self.cube_size = (30, 30)
        self.cube_initial_pos = (
            sim_config["screen_width"] * 0.25,
            sim_config["screen_height"] * 0.5,
        )
        self.success_threshold = 20

    def setup(self, space):
        self.clear_from_space(space)
        mass = 1.0
        moment = pymunk.moment_for_box(mass, self.cube_size)
        self.cube_body = pymunk.Body(mass, moment)
        self.cube_body.position = self.cube_initial_pos
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, self.cube_size)
        self.cube_shape.friction = 0.7
        self.cube_shape.elasticity = 0.3
        self.cube_shape.collision_type = COLLISION_TYPE_SCENARIO_OBJECT_START + 1
        space.add(self.cube_body, self.cube_shape)
        cube_obj_info = {
            "body": self.cube_body,
            "shape": self.cube_shape,
            "initial_pos": self.cube_initial_pos,
            "type": "dynamic_cube",
        }
        self.pymunk_objects.append(cube_obj_info)
        return [cube_obj_info]

    def get_reward(self, robot, action, observation):
        reward = 0.0
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            dist_to_target = np.linalg.norm(cube_pos - self.target_pos)
            reward -= dist_to_target * 0.01
            if dist_to_target < self.success_threshold:
                reward += 100.0
        action_penalty = np.sum(np.square(action)) * 0.001
        reward -= action_penalty
        return reward

    def is_terminated(self, robot, observation, current_step, max_steps):
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            if np.linalg.norm(cube_pos - self.target_pos) < self.success_threshold:
                print("MoveCubeScenario: Target reached!")
                return True
        return False

    def get_scenario_observation(self, robot):
        obs = []
        if self.cube_body:
            obs.extend(
                [
                    self.cube_body.position.x,
                    self.cube_body.position.y,
                    self.cube_body.velocity.x,
                    self.cube_body.velocity.y,
                    self.cube_body.angle,
                ]
            )
        else:
            obs.extend([0.0] * 5)
        obs.extend(self.target_pos)
        return np.array(obs, dtype=np.float32)


class ExtractCubeFromTubeScenario(Scenario):
    def __init__(self, sim_config):
        super().__init__(sim_config)
        self.cube_body = None
        self.cube_shape = None
        self.cube_size = (20, 20)
        self.tube_opening_y = sim_config["screen_height"] * 0.3
        self.tube_height = 150
        self.tube_width = 40
        self.tube_wall_thickness = 5
        self.tube_center_x = sim_config["screen_width"] * 0.5
        self.cube_initial_pos = (
            self.tube_center_x,
            self.tube_opening_y + self.tube_height - self.cube_size[1] / 2 - 5,
        )
        self.extraction_target_y = self.tube_opening_y - self.cube_size[1]
        self.tube_parts_shapes = []

    def setup(self, space):
        self.clear_from_space(space)
        self.tube_parts_shapes = []
        wall_friction = 0.1
        wall_elasticity = 0.1
        left_wall_verts = [
            (
                self.tube_center_x - self.tube_width / 2 - self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height,
            ),
            (
                self.tube_center_x - self.tube_width / 2,
                self.tube_opening_y + self.tube_height,
            ),
            (self.tube_center_x - self.tube_width / 2, self.tube_opening_y),
            (
                self.tube_center_x - self.tube_width / 2 - self.tube_wall_thickness,
                self.tube_opening_y,
            ),
        ]
        left_wall_shape = pymunk.Poly(space.static_body, left_wall_verts)
        left_wall_shape.friction = wall_friction
        left_wall_shape.elasticity = wall_elasticity
        left_wall_shape.collision_type = COLLISION_TYPE_SCENARIO_STATIC_START + 2
        space.add(left_wall_shape)
        self.pymunk_objects.append(
            {
                "body": space.static_body,
                "shape": left_wall_shape,
                "type": "static_tube_part",
            }
        )
        self.tube_parts_shapes.append(left_wall_shape)
        right_wall_verts = [
            (
                self.tube_center_x + self.tube_width / 2,
                self.tube_opening_y + self.tube_height,
            ),
            (
                self.tube_center_x + self.tube_width / 2 + self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height,
            ),
            (
                self.tube_center_x + self.tube_width / 2 + self.tube_wall_thickness,
                self.tube_opening_y,
            ),
            (self.tube_center_x + self.tube_width / 2, self.tube_opening_y),
        ]
        right_wall_shape = pymunk.Poly(space.static_body, right_wall_verts)
        right_wall_shape.friction = wall_friction
        right_wall_shape.elasticity = wall_elasticity
        right_wall_shape.collision_type = COLLISION_TYPE_SCENARIO_STATIC_START + 3
        space.add(right_wall_shape)
        self.pymunk_objects.append(
            {
                "body": space.static_body,
                "shape": right_wall_shape,
                "type": "static_tube_part",
            }
        )
        self.tube_parts_shapes.append(right_wall_shape)
        bottom_wall_verts = [
            (
                self.tube_center_x - self.tube_width / 2 - self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height + self.tube_wall_thickness,
            ),
            (
                self.tube_center_x + self.tube_width / 2 + self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height + self.tube_wall_thickness,
            ),
            (
                self.tube_center_x + self.tube_width / 2 + self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height,
            ),
            (
                self.tube_center_x - self.tube_width / 2 - self.tube_wall_thickness,
                self.tube_opening_y + self.tube_height,
            ),
        ]
        bottom_wall_shape = pymunk.Poly(space.static_body, bottom_wall_verts)
        bottom_wall_shape.friction = wall_friction
        bottom_wall_shape.elasticity = wall_elasticity
        bottom_wall_shape.collision_type = COLLISION_TYPE_SCENARIO_STATIC_START + 1
        space.add(bottom_wall_shape)
        self.pymunk_objects.append(
            {
                "body": space.static_body,
                "shape": bottom_wall_shape,
                "type": "static_tube_part",
            }
        )
        self.tube_parts_shapes.append(bottom_wall_shape)
        mass = 0.5
        moment = pymunk.moment_for_box(mass, self.cube_size)
        self.cube_body = pymunk.Body(mass, moment)
        self.cube_body.position = self.cube_initial_pos
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, self.cube_size)
        self.cube_shape.friction = 0.5
        self.cube_shape.elasticity = 0.2
        self.cube_shape.collision_type = COLLISION_TYPE_SCENARIO_OBJECT_START + 2
        space.add(self.cube_body, self.cube_shape)
        cube_obj_info = {
            "body": self.cube_body,
            "shape": self.cube_shape,
            "initial_pos": self.cube_initial_pos,
            "type": "dynamic_cube",
        }
        self.pymunk_objects.append(cube_obj_info)
        return self.pymunk_objects

    def get_reward(self, robot, action, observation):
        reward = 0.0
        if self.cube_body:
            dist_to_extraction_y = abs(
                self.cube_body.position.y - self.extraction_target_y
            )
            reward -= dist_to_extraction_y * 0.02
            if (
                self.cube_body.position.y
                < self.extraction_target_y + self.cube_size[1] / 2
            ):
                reward += 200.0
                if self.cube_body.position.y < self.extraction_target_y:
                    horizontal_dist_from_tube_center = abs(
                        self.cube_body.position.x - self.tube_center_x
                    )
                    reward += horizontal_dist_from_tube_center * 0.005
        action_penalty = np.sum(np.square(action)) * 0.001
        reward -= action_penalty
        return reward

    def is_terminated(self, robot, observation, current_step, max_steps):
        if self.cube_body:
            if self.cube_body.position.y < self.extraction_target_y:
                print("ExtractCubeScenario: Cube extracted!")
                return True
        return False

    def get_scenario_observation(self, robot):
        obs = []
        if self.cube_body:
            obs.extend(
                [
                    self.cube_body.position.x,
                    self.cube_body.position.y,
                    self.cube_body.velocity.x,
                    self.cube_body.velocity.y,
                    self.cube_body.angle,
                ]
            )
        else:
            obs.extend([0.0] * 5)
        obs.extend(
            [self.tube_center_x, self.tube_opening_y, self.tube_width, self.tube_height]
        )
        return np.array(obs, dtype=np.float32)
