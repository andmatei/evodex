import math
import numpy as np
from evodex.simulation.utils import (
    COLLISION_TYPE_GROUND,
    COLLISION_TYPE_ROBOT_BASE,
    COLLISION_TYPE_ROBOT_SEGMENT_START,
    COLLISION_TYPE_SCENARIO_OBJECT_START,
    COLLISION_TYPE_SCENARIO_STATIC_START,
)
import gymnasium as gym
from gymnasium import spaces
from evodex.simulation.robot import Robot
from evodex.simulation.scenario import (
    MoveCubeToTargetScenario,
    ExtractCubeFromTubeScenario,
)
from evodex.simulation.simulation import Simulation
from evodex.simulation.config import DEFAULT_ROBOT_CONFIG


class RobotHandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        robot_config=DEFAULT_ROBOT_CONFIG,
        scenario_instance=None,
        max_episode_steps=1000,
    ):
        super().__init__()
        self.robot_config_initial = robot_config.copy()
        self.scenario_instance_initial_type = (
            scenario_instance.__class__
            if scenario_instance
            else MoveCubeToTargetScenario
        )
        self.current_robot_config = robot_config
        self.current_sim_config = self.current_robot_config["simulation"]
        if scenario_instance is None:
            self.current_scenario_instance = self.scenario_instance_initial_type(
                self.current_sim_config
            )
        else:
            self.current_scenario_instance = scenario_instance
        self.simulation = Simulation(
            self.current_robot_config, self.current_scenario_instance
        )
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self._setup_action_space()
        self._setup_observation_space()
        self.render_mode = "human"

    def _setup_action_space(self):
        self.total_motors = sum(
            f["num_segments"] for f in self.current_robot_config["fingers"]
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.total_motors,), dtype=np.float32
        )

    def _setup_observation_space(self):
        robot_obs_sample = self.simulation.robot.get_observation()
        scenario_obs_sample = self.current_scenario_instance.get_scenario_observation(
            self.simulation.robot
        )
        combined_obs_sample = np.concatenate((robot_obs_sample, scenario_obs_sample))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=combined_obs_sample.shape, dtype=np.float32
        )

    def step(self, action):
        scaled_action = action * math.pi
        self.simulation.step(
            scaled_action
        )  # Simulation step now handles base velocity from keyboard
        self.current_step += 1
        robot_observation = self.simulation.robot.get_observation()
        scenario_observation = self.current_scenario_instance.get_scenario_observation(
            self.simulation.robot
        )
        observation = np.concatenate((robot_observation, scenario_observation))
        reward = self.current_scenario_instance.get_reward(
            self.simulation.robot, scaled_action, observation
        )
        terminated = self.current_scenario_instance.is_terminated(
            self.simulation.robot,
            observation,
            self.current_step,
            self.max_episode_steps,
        )
        truncated = self.current_scenario_instance.is_truncated(
            self.simulation.robot,
            observation,
            self.current_step,
            self.max_episode_steps,
        )
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        new_robot_config_dict = self.current_robot_config
        new_scenario_inst = self.current_scenario_instance
        if options:
            if "robot_config" in options:
                new_robot_config_dict = options["robot_config"]
                self.current_robot_config = new_robot_config_dict
                self.current_sim_config = self.current_robot_config["simulation"]
                self._setup_action_space()
            if "scenario_instance" in options:
                new_scenario_inst = options["scenario_instance"]
                if (
                    not hasattr(new_scenario_inst, "sim_config")
                    or new_scenario_inst.sim_config != self.current_sim_config
                ):
                    new_scenario_inst = new_scenario_inst.__class__(
                        self.current_sim_config
                    )
                self.current_scenario_instance = new_scenario_inst
            elif "scenario_class" in options:
                scenario_class = options["scenario_class"]
                new_scenario_inst = scenario_class(self.current_sim_config)
                self.current_scenario_instance = new_scenario_inst
        if self.current_scenario_instance.sim_config != self.current_sim_config:
            self.current_scenario_instance = self.current_scenario_instance.__class__(
                self.current_sim_config
            )
        self.simulation.reset_simulation(
            new_robot_config=self.current_robot_config,
            new_scenario_instance=self.current_scenario_instance,
        )
        self._setup_observation_space()
        self.current_step = 0
        robot_observation = self.simulation.robot.get_observation()
        scenario_observation = self.current_scenario_instance.get_scenario_observation(
            self.simulation.robot
        )
        observation = np.concatenate((robot_observation, scenario_observation))
        info = {}
        return observation, info

    def render(self):
        if self.render_mode == "human":
            self.simulation.render()

    def close(self):
        self.simulation.close()
