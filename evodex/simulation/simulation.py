import pymunk
import pygame
import pymunk.pygame_util
from typing import Optional

from evodex.simulation.scenario.core import Scenario
from .config import SimulatorConfig
from .robot import Robot, RobotConfig, Action, Observation
from .scenario import ScenarioRegistry, ScenarioConfig


# TODO: deal with the config and move to a separate file
class ManualController:
    def __init__(self, config: dict):
        self.config = config

        self.key_move_speed = self.config.get("key_move_speed", 150)
        self.key_angular_speed = self.config.get("key_angular_speed", 1.5)

        self.angular_mode = False

        self.base_target_vx = 0.0
        self.base_target_vy = 0.0
        self.base_target_omega = 0.0

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.angular_mode = True
                self.base_target_vx = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.base_target_omega = -self.key_angular_speed
                elif keys[pygame.K_RIGHT]:
                    self.base_target_omega = self.key_angular_speed
            elif event.key == pygame.K_UP:
                self.base_target_vy = -self.key_move_speed
            elif event.key == pygame.K_DOWN:
                self.base_target_vy = self.key_move_speed
            elif event.key == pygame.K_LEFT:
                if self.angular_mode:
                    self.base_target_omega = -self.key_angular_speed
                    self.base_target_vx = 0.0
                else:
                    self.base_target_vx = -self.key_move_speed
                    self.base_target_omega = 0.0
            elif event.key == pygame.K_RIGHT:
                if self.angular_mode:
                    self.base_target_omega = self.key_angular_speed
                    self.base_target_vx = 0.0
                else:
                    self.base_target_vx = self.key_move_speed
                    self.base_target_omega = 0.0

        elif event.type == pygame.KEYUP:
            if (event.key == pygame.K_UP and self.base_target_vy < 0) or (
                event.key == pygame.K_DOWN and self.base_target_vy > 0
            ):
                self.base_target_vy = 0.0
            elif event.key == pygame.K_SPACE:
                self.angular_mode = False
                self.base_target_omega = 0.0

                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    self.base_target_vx = -self.key_move_speed
                elif keys[pygame.K_RIGHT]:
                    self.base_target_vx = self.key_move_speed
            if self.angular_mode:
                if (event.key == pygame.K_LEFT and self.base_target_omega < 0) or (
                    event.key == pygame.K_RIGHT and self.base_target_omega > 0
                ):
                    self.base_target_omega = 0.0
            else:
                if (event.key == pygame.K_LEFT and self.base_target_vx < 0) or (
                    event.key == pygame.K_RIGHT and self.base_target_vx > 0
                ):
                    self.base_target_vx = 0.0

    def get_actions(self):
        return {
            "base_vx": self.base_target_vx,
            "base_vy": self.base_target_vy,
            "base_omega": self.base_target_omega,
        }


class Simulator:
    robot: Optional[Robot] = None
    scenario: Optional[Scenario] = None

    def __init__(
        self,
        robot_config: RobotConfig,
        scenario_config: ScenarioConfig,
        config: SimulatorConfig,
    ):
        self.robot_config = robot_config
        self.scenario_config = scenario_config
        self.sim_config = config

        # TODO: integrate keyboard control

        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                self.sim_config.simulation.screen_width,
                self.sim_config.simulation.screen_height,
            )
        )
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        pygame.display.set_caption("Robotic Hand Simulation")

        self.space = pymunk.Space()
        self.space.gravity = self.sim_config.simulation.gravity
        self.dt = self.sim_config.simulation.dt

        self.reset()

    def handle_pygame_event(self, event):
        """Handles Pygame events, specifically for keyboard control of the base."""
        if event.type == pygame.QUIT:
            pygame.quit()
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

            if event.key == pygame.K_r:
                print("Resetting simulation...")
                self.reset()
                return

    def reset(
        self,
        new_robot_config: Optional[RobotConfig] = None,
        new_scenario_config: Optional[ScenarioConfig] = None,
    ):
        # if self.keyboard_mode:
        #     self.base_target_vx = 0.0
        #     self.base_target_vy = 0.0
        #     self.base_target_omega = 0.0

        if self.robot is not None:
            self.robot.remove_from_space(self.space)

        if self.scenario is not None:
            self.scenario.clear_from_space(self.space)

        if new_robot_config:
            self.robot_config = new_robot_config

        if new_scenario_config:
            self.scenario_config = new_scenario_config

        self.scenario = ScenarioRegistry.create(**self.scenario_config.model_dump())

        # TODO: add start position to the scenario
        self.robot = Robot(self.scenario_config.robot_start_position, self.robot_config)
        self.robot.add_to_space(self.space)

        self.scenario.setup(self.space, self.robot)

    def step(self, actions: Optional[Action] = None):
        if actions is not None and self.robot:
            self.robot.act(actions)

        # TODO: integrate keyboard control
        # if self.robot and self.keyboard_mode:
        #     self.robot.base.body.velocity = (self.base_target_vx, self.base_target_vy)
        #     self.robot.base.body.angular_velocity = self.base_target_omega

        self.space.step(self.dt)

    def render(self):
        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.draw_options)

        if self.scenario:
            self.scenario.render(self.screen)

        pygame.display.flip()
        self.clock.tick(1.0 / self.dt)

    def close(self):
        pygame.quit()
