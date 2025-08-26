import yaml
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import RescaleAction
from typing import Optional
from enum import Enum

from evodex.simulation import RobotHandEnv, BaseMaskWrapper
from evodex.simulation.wrapper import flatten_env


class EnvMask(Enum):
    NONE = 0
    BASE = 1
    FINGERS = 2


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
