import numpy as np
import gymnasium as gym
import pymunk
from gymnasium import spaces
from typing import Optional
from .robot import Robot, RobotConfig, Action
from .scenario import Scenario, ScenarioConfig, ScenarioRegistry
from .simulation import SimulatorConfig
from .types import Observation


class RobotHandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    robot: Optional[Robot] = None
    scenario: Optional[Scenario] = None

    def __init__(
        self,
        robot_config: RobotConfig,
        scenario_config: ScenarioConfig,
        env_config: SimulatorConfig,
    ):
        super().__init__()

        self.robot_config = robot_config
        self.scenario_config = scenario_config
        self.env_config = env_config

        self.space = pymunk.Space()
        self.space.gravity = self.env_config.simulation.gravity
        self.dt = self.env_config.simulation.dt

        # Define action and observation spaces
        # TODO: Add constraints from ActionScaleConfig
        self.action_space = spaces.Dict(
            {
                "base": spaces.Dict(
                    {
                        "velocity": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                        "omega": spaces.Box(
                            low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32
                        ),
                    }
                ),
                "fingers": spaces.Tuple(
                    [
                        spaces.Tuple(
                            [
                                spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
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
                        "fingers": spaces.Tuple(
                            [
                                spaces.Dict(
                                    {
                                        "segments": spaces.Tuple(
                                            [
                                                spaces.Dict(
                                                    {
                                                        "position": spaces.Box(
                                                            low=-np.inf,
                                                            high=np.inf,
                                                            shape=(2,),
                                                            dtype=np.float32,
                                                        ),
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
                                                        "velocity": spaces.Box(
                                                            low=-np.inf,
                                                            high=np.inf,
                                                            shape=(1,),
                                                            dtype=np.float32,
                                                        ),
                                                    }
                                                )
                                                for _ in finger.segments
                                            ]
                                        ),
                                        "fingertip_position": spaces.Box(
                                            low=-np.inf,
                                            high=np.inf,
                                            shape=(2,),
                                            dtype=np.float32,
                                        ),
                                    }
                                )
                                for finger in self.robot_config.fingers
                            ]
                        ),
                    }
                ),
                "scenario": spaces.Dict(
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
                    }
                ),
            }
        )

        self.reset()

    def reset(self):
        if self.robot:
            self.robot.remove_from_space(self.space)
        if self.scenario:
            self.scenario.clear_from_space(self.space)

        self.robot = Robot(position=(0, 0), config=self.robot_config)
        self.robot.add_to_space(self.space)

        self.scenario = ScenarioRegistry.create(self.scenario_config)
        self.scenario.setup(self.space, self.robot)

        self.step_count = 0

        return self.get_observation().model_dump(), {}

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
        obs = self.get_observation()

        # Calculate reward and terminated (if applicable)
        reward = self.scenario.get_reward(self.robot, robot_action)
        terminated = self.scenario.is_terminated(self.robot)
        truncated = (
            self.step_count >= self.env_config.simulation.max_steps
            if self.env_config.simulation.max_steps
            else False
        )

        return obs.model_dump(), reward, terminated, truncated, {}

    def get_observation(self) -> Observation:
        if not self.robot or not self.scenario:
            raise ValueError("Robot or scenario is not initialized.")

        robot_obs = self.robot.get_observation()
        scenario_obs = self.scenario.get_observation(robot=self.robot)

        return Observation(robot=robot_obs, scenario=scenario_obs)
