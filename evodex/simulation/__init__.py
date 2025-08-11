from .env import RobotHandEnv
from .types import Observation

import gymnasium as gym

gym.register(
    id="RobotHandEnv-v0",
    entry_point="evodex.simulation.env:RobotHandEnv",
)
