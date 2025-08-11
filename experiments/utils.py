import yaml
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import RescaleAction
from typing import Optional

from evodex.simulation import RobotHandEnv
from evodex.simulation.wrapper import flatten_env


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def make_env(
    robot_config: dict,
    scenario_config: dict,
    simulator_config: dict,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """
    Create a custom environment for the robot hand simulation.

    Args:
        robot_config (dict): Configuration for the robot.
        scenario_config (dict): Configuration for the scenario.
        simulator_config (dict): Configuration for the simulator.
        render_mode (str): Rendering mode for the environment.

    Returns:
        gym.Env: The custom environment instance.
    """
    env: gym.Env = flatten_env(
        RobotHandEnv(
            robot_config=robot_config,
            scenario_config=scenario_config,
            env_config=simulator_config,
            render_mode=render_mode,
        ),
        observation=True,
        action=True,
    )
    env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))
    return env
