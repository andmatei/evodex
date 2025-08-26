from .env import RobotHandEnv, BaseMaskWrapper
from .types import Observation
from .robot import Action

import gymnasium as gym

gym.register(
    id="RobotHandEnv-v0",
    entry_point="evodex.simulation.env:RobotHandEnv",
)
