import math
import numpy as np
import pymunk
import pygame
import pymunk.pygame_util
from evodex.simulation.robot import Robot
from evodex.simulation.utils import (
    COLLISION_TYPE_GROUND,
    COLLISION_TYPE_ROBOT_BASE,
    COLLISION_TYPE_ROBOT_SEGMENT_START,
    COLLISION_TYPE_SCENARIO_OBJECT_START,
    COLLISION_TYPE_SCENARIO_STATIC_START,
    pymunk_to_pygame_coord,
    pygame_to_pymunk_coord,
)

from evodex.simulation.scenario import (
    MoveCubeToTargetScenario,
    ExtractCubeFromTubeScenario,
)


class Simulation:
    def __init__(self, robot_config, scenario_instance):
        self.robot_config = robot_config
        self.sim_config = robot_config["simulation"]
        self.scenario = scenario_instance
        self.key_move_speed = self.sim_config.get(
            "key_move_speed", 150
        )  # Get from config

        pygame.init()
        self.screen_width = self.sim_config["screen_width"]
        self.screen_height = self.sim_config["screen_height"]
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Robotic Hand Simulation")
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.space = pymunk.Space()
        self.space.gravity = self.sim_config["gravity"]
        self.space.iterations = 20
        self.dt = self.sim_config["dt"]

        self.robot = None
        self.scenario_pymunk_elements = []
        self._add_persistent_static_elements()

        # For keyboard control of the base
        self.base_target_vx = 0.0
        self.base_target_vy = 0.0

        self.reset_simulation()

    def _add_persistent_static_elements(self):
        ground_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(
            ground_body,
            (0, self.screen_height - 10),
            (self.screen_width, self.screen_height - 10),
            5,
        )
        ground_shape.friction = 1.0
        ground_shape.collision_type = COLLISION_TYPE_GROUND
        self.space.add(ground_body, ground_shape)
        self.persistent_static_elements = [(ground_body, ground_shape)]

    def handle_pygame_event(self, event):
        """Handles Pygame events, specifically for keyboard control of the base."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.base_target_vx = -self.key_move_speed
            elif event.key == pygame.K_RIGHT:
                self.base_target_vx = self.key_move_speed
            elif (
                event.key == pygame.K_UP
            ):  # Pymunk Y is down, so negative VY moves up screen
                self.base_target_vy = -self.key_move_speed
            elif event.key == pygame.K_DOWN:  # Positive VY moves down screen
                self.base_target_vy = self.key_move_speed
            elif event.key == pygame.K_r:  # Allow reset via 'r' key
                print("Resetting simulation via 'r' key...")
                self.reset_simulation()

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT and self.base_target_vx < 0:
                self.base_target_vx = 0.0
            elif event.key == pygame.K_RIGHT and self.base_target_vx > 0:
                self.base_target_vx = 0.0
            elif event.key == pygame.K_UP and self.base_target_vy < 0:
                self.base_target_vy = 0.0
            elif event.key == pygame.K_DOWN and self.base_target_vy > 0:
                self.base_target_vy = 0.0

    def reset_simulation(self, new_robot_config=None, new_scenario_instance=None):
        # Reset keyboard-controlled velocities
        self.base_target_vx = 0.0
        self.base_target_vy = 0.0

        if new_robot_config:
            self.robot_config = new_robot_config
            self.sim_config = self.robot_config.get("simulation", self.sim_config)
            self.space.gravity = self.sim_config.get("gravity", self.space.gravity)
            self.key_move_speed = self.sim_config.get("key_move_speed", 150)

        if new_scenario_instance:
            if self.scenario:
                self.scenario.clear_from_space(self.space)
            self.scenario = new_scenario_instance

        if self.robot:
            self.robot.remove_from_space(self.space)

        if self.scenario and not new_scenario_instance:
            self.scenario.clear_from_space(self.space)

        if self.scenario:
            self.scenario_pymunk_elements = self.scenario.setup(self.space)
        else:
            self.scenario_pymunk_elements = []

        self.robot = Robot(self.robot_config)
        self.robot.add_to_space(self.space)

        if self.scenario:
            self.scenario.reset_objects()

    def step(self, actions=None):
        if actions is not None and self.robot:
            self.robot.apply_actions(actions)

        if self.robot:
            self.robot.base.body.velocity = (self.base_target_vx, self.base_target_vy)

        self.space.step(self.dt)

    def render(self):
        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.draw_options)

        if isinstance(self.scenario, MoveCubeToTargetScenario):
            target_center_pygame = pymunk_to_pygame_coord(
                self.scenario.target_pos, self.screen_height
            )
            pygame.draw.circle(
                self.screen,
                pygame.Color("lightgreen"),
                target_center_pygame,
                self.scenario.target_radius,
                2,
            )
        elif isinstance(self.scenario, ExtractCubeFromTubeScenario):
            pass

        pygame.display.flip()
        self.clock.tick(1.0 / self.dt)

    def close(self):
        pygame.quit()
