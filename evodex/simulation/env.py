import numpy as np
import gymnasium as gym
from gymnasium import spaces
from evodex.simulation.simulation import Simulation
from evodex.simulation.config import (
    DEFAULT_ROBOT_CONFIG,
    DEFAULT_SCENARIO_CONFIG,
    DEFAULT_SIMULATOR_CONFIG,
)


class RobotHandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        robot_config=DEFAULT_ROBOT_CONFIG,
        scenario_config=DEFAULT_SCENARIO_CONFIG,
        sim_config=DEFAULT_SIMULATOR_CONFIG,
        max_episode_steps=1000,
    ):
        super().__init__()
        self.robot_config = robot_config

        self.robot_config = robot_config
        self.scenario_config = scenario_config
        self.sim_config = sim_config

        self.simulation = Simulation(
            robot_config=self.robot_config,
            scenario_config=self.scenario_config,
            config=self.sim_config,
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self._setup_action_space()
        self._setup_observation_space()

    def _setup_action_space(self):
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.simulation.robot.get_action_space(),),
            dtype=np.float32,
        )

    def _setup_observation_space(self):
        robot_obs_space = self.simulation.robot.get_observation_space()
        scenario_obs_space = self.simulation.scenario.get_observation(
            self.simulation.robot
        ).shape

        combined_obs_sample = sce
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=combined_obs_sample.shape, dtype=np.float32
        )

    def step(self, action):
        self.simulation.step(action)

        self.current_step += 1

        robot_observation = self.simulation.robot.get_observation()
        scenario_observation = self.simulation.scenario.get_observation(
            self.simulation.robot
        )
        observation = np.concatenate((robot_observation, scenario_observation))

        reward = self.simulation.scenario.get_reward(
            self.simulation.robot, action, scenario_observation
        )

        terminated = self.simulation.scenario.is_terminated(
            self.simulation.robot,
            scenario_observation,
            self.current_step,
            self.max_episode_steps,
        )

        truncated = self.simulation.scenario.is_truncated(
            self.simulation.robot,
            scenario_observation,
            self.current_step,
            self.max_episode_steps,
        )

        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed=None,
        new_robot_config=None,
        new_scenario_config=None,
        new_sim_config=None,
    ):
        super().reset(seed=seed)
        if new_robot_config:
            self.robot_config = new_robot_config
        if new_scenario_config:
            self.scenario_config = new_scenario_config
        if new_sim_config:
            self.sim_config = new_sim_config
            self.simulation = Simulation(
                robot_config=self.robot_config,
                scenario_config=self.scenario_config,
                config=self.sim_config,
            )

        self.simulation.reset(
            new_robot_config=self.robot_config,
            new_scenario_config=self.scenario_config,
        )

        self.current_step = 0
        robot_observation = self.simulation.robot.get_observation()
        scenario_observation = self.simulation.scenario.get_observation(
            self.simulation.robot
        )
        observation = np.concatenate((robot_observation, scenario_observation))
        return observation, {}

    def render(self):
        if self.render_mode == "human":
            self.simulation.render()

    def close(self):
        self.simulation.close()
