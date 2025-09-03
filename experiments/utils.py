import yaml
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import RescaleAction
from typing import Optional, Any
from enum import Enum

from evodex.simulation import RobotHandEnv, BaseMaskWrapper
from evodex.simulation.wrapper import flatten_env


class EnvMask(Enum):
    NONE = 0
    BASE = 1
    FINGERS = 2


def clean_data(d: Any) -> Any:
    """
    Recursively remove empty lists and dictionaries from a nested dictionary.

    Args:
        d (dict): The input dictionary to clean.

    Returns:
        dict: The cleaned dictionary.
    """
    if isinstance(d, dict):
        return {k: clean_data(v) for k, v in d.items()}
    if isinstance(d, tuple) or isinstance(d, list):
        return [clean_data(v) for v in d]
    return d


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


def save_config(config: dict, path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        path (str): Path to the YAML configuration file.
        config (dict): Configuration dictionary to save.
    """
    clean_config = clean_data(config)
    with open(path, "w") as file:
        yaml.dump(clean_config, file)


def make_env(
    robot_config: dict,
    scenario_config: dict,
    simulator_config: dict,
    render_mode: Optional[str] = None,
    mask: EnvMask = EnvMask.NONE,
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

    robot_env: RobotHandEnv = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode=render_mode,
    )

    env: gym.Env = robot_env
    if mask == EnvMask.BASE:
        env = BaseMaskWrapper(robot_env)

    env = flatten_env(
        env,
        observation=True,
        action=True,
    )
    env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))
    return env
