from re import A
import numpy as np
import gymnasium as gym
import pymunk
import pygame
from gymnasium import spaces
from typing import Optional

from .renderer import Renderer
from .robot import Robot, RobotConfig, Action
from .scenario import Scenario, ScenarioRegistry
from .config import EnvConfig
from .types import Observation, RobotObservation


class RobotHandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    robot: Optional[Robot] = None
    scenario: Optional[Scenario] = None
    renderer: Optional[Renderer] = None

    def __init__(
        self,
        robot_config: dict,
        scenario_config: dict,
        env_config: dict,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.robot_config = RobotConfig(**robot_config)
        self.env_config = EnvConfig(**env_config)
        self.scenario_config = scenario_config
        self.render_mode = render_mode

        # Initialize the physical simulation
        self.space = pymunk.Space()
        self.space.gravity = self.env_config.simulation.gravity
        self.dt = self.env_config.simulation.dt

        # Define action and observation spaces
        self.action_space = spaces.Dict(
            {
                "base": spaces.Dict(
                    {
                        "velocity": spaces.Box(
                            low=self.robot_config.limits.velocity.min,
                            high=self.robot_config.limits.velocity.max,
                            shape=(2,),
                            dtype=np.float32,
                        ),
                        "omega": spaces.Box(
                            low=self.robot_config.limits.omega.min,
                            high=self.robot_config.limits.omega.max,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "fingers": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=self.robot_config.limits.motor_rate.min,
                                    high=self.robot_config.limits.motor_rate.max,
                                    shape=(1,),
                                    dtype=np.float32,
                                )
                                for _ in range(len(finger))
                            ]
                        )
                        for finger in self.robot_config.fingers
                    ]
                ),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        "extrinsic": spaces.Dict(
                            {
                                "robot": spaces.Dict(
                                    {
                                        "base": spaces.Dict(
                                            {
                                                "position": spaces.Box(
                                                    low=-np.inf,
                                                    high=np.inf,
                                                    shape=(2,),
                                                    dtype=np.float32,
                                                ),
                                                "angle": spaces.Box(
                                                    low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                                                ),
                                                "velocity": spaces.Box(
                                                    low=-np.inf,
                                                    high=np.inf,
                                                    shape=(2,),
                                                    dtype=np.float32,
                                                ),
                                                "angular_velocity": spaces.Box(
                                                    low=-np.inf,
                                                    high=np.inf,
                                                    shape=(1,),
                                                    dtype=np.float32,
                                                ),
                                            }
                                        ),
                                        "fingertips": spaces.Tuple(
                                            [
                                                spaces.Dict(
                                                    {
                                                        "tip": spaces.Dict(
                                                            {
                                                                "position": spaces.Box(
                                                                    low=-np.inf,
                                                                    high=np.inf,
                                                                    shape=(2,),
                                                                    dtype=np.float32,
                                                                ),
                                                                "velocity": spaces.Box(
                                                                    low=-np.inf,
                                                                    high=np.inf,
                                                                    shape=(2,),
                                                                    dtype=np.float32,
                                                                ),
                                                            }
                                                        ),
                                                    }
                                                )
                                                for finger in self.robot_config.fingers
                                            ]
                                        ),
                                    }
                                ),
                                "object": spaces.Dict(
                                    {
                                        "velocity": spaces.Box(
                                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                                        ),
                                        "position": spaces.Box(
                                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                                        ),
                                        "angle": spaces.Box(
                                            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                                        ),
                                        "angular_velocity": spaces.Box(
                                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                                        ),
                                        "size": spaces.Box(
                                            low=0, high=np.inf, shape=(2,), dtype=np.float32
                                        ),
                                    }
                                ),
                            }
                        ),
                        "intrinsic": spaces.Dict(
                            {
                                "fingers": spaces.Tuple(
                                    [
                                        spaces.Tuple(
                                            [
                                                spaces.Dict(
                                                    {
                                                        "joint_angle": spaces.Box(
                                                            low=-np.pi,
                                                            high=np.pi,
                                                            shape=(1,),
                                                            dtype=np.float32,
                                                        ),
                                                        "joint_angular_velocity": spaces.Box(
                                                            low=-np.inf,
                                                            high=np.inf,
                                                            shape=(1,),
                                                            dtype=np.float32,
                                                        ),
                                                        "is_touching": spaces.MultiBinary(
                                                            n=1
                                                        ),
                                                    }
                                                )
                                                for _ in finger.segments
                                            ]
                                        )
                                        for finger in self.robot_config.fingers
                                    ]
                                )
                            }
                        ),
                    }
                ),
                "achieved_goal": spaces.Dict(
                    {
                        "position": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                        "angle": spaces.Box(
                            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                        ),
                        "angular_velocity": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "velocity": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                    }
                ),
                "desired_goal": spaces.Dict(
                    {
                        "position": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                        "angle": spaces.Box(
                            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                        ),
                        "angular_velocity": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                        ),
                        "velocity": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                    }
                ),
            }
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        if self.robot:
            self.robot.remove_from_space(self.space)
        if self.scenario:
            self.scenario.clear_from_space(self.space)

        self.robot = Robot(self.robot_config)
        self.scenario = ScenarioRegistry.load(self.scenario_config)
        self.scenario.setup(self.space, self.robot)

        self.step_count = 0

        return self._get_observation().model_dump(), {}

    def step(self, action: dict):
        if self.robot is None:
            raise ValueError("Robot is not initialized. Call reset() first.")

        if self.scenario is None:
            raise ValueError("Scenario is not initialized. Call reset() first.")

        # Apply actions to the robot
        robot_action = Action(**action)
        self.robot.act(robot_action)

        # Step the simulation
        self.space.step(self.dt)
        self.step_count += 1

        # Get observation
        obs = self._get_observation()

        # Calculate reward and terminated (if applicable)
        reward = self.scenario.get_reward(self.robot, robot_action)
        terminated = self.scenario.is_terminated(self.robot)
        truncated = (
            self.step_count >= self.env_config.simulation.max_steps
            if self.env_config.simulation.max_steps
            else False
        )

        return obs.model_dump(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.scenario is None:
                raise ValueError("Scenario is not initialized. Call reset() first.")

            if self.renderer is None:
                self.renderer = Renderer(config=self.env_config.render)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.renderer.close()
                    return

            # Draw logic
            self.renderer.render(self.space, self.scenario)

    def close(self):
        if self.renderer:
            self.renderer.close()
        self.renderer = None

    def _get_observation(self) -> Observation:
        if not self.robot or not self.scenario:
            raise ValueError("Robot or scenario is not initialized.")

        intrinsic_obs = self.robot.get_intrinsic_observation()
        extrinsic_obs = self.scenario.get_observation(robot=self.robot)
        goal = self.scenario.get_goal(robot=self.robot)
        achieved_goal = self.scenario.get_achieved_goal(robot=self.robot)

        return Observation(
            observation=RobotObservation(
                extrinsic=extrinsic_obs,
                intrinsic=intrinsic_obs,
            ),
            achieved_goal=achieved_goal,
            desired_goal=goal,
        )
