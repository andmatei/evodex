import yaml
import gymnasium as gym
import numpy as np
import os

from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from evodex.simulation import RobotHandEnv
from evodex.simulation.robot.spaces import Action, BaseAction
from evodex.simulation.wrapper import flatten


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


# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/base_robot.yaml")
    scenario_config = load_config("configs/move_cube_scenario.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    env: gym.Env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode="human",  # Set to 'human' for visual rendering
    )

    action = Action(
        base=BaseAction(
            velocity=(0.0, 0.0),  # Base velocity in x and y directions
            omega=0.0,  # Base angular velocity
        ),
        fingers=[[0.0, 0.0], [0.0, 0.0, 0.2]],  # Example finger motor rates
    )

    obs, _ = env.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(action.model_dump())
        env.render()
