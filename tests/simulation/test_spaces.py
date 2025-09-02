import gymnasium as gym
import pytest

from tests.utils import load_config
from evodex.simulation import RobotHandEnv, Observation, Action


# Adjust these paths if your test configs are elsewhere
ROBOT_CONFIG_PATH = "tests/configs/test_robot.yaml"
SCENARIO_CONFIG_PATH = "tests/configs/test_scenario.yaml"
SIMULATOR_CONFIG_PATH = "tests/configs/test_simulator.yaml"


def test_env_observation_and_action_spaces() -> None:
    robot_config = load_config(ROBOT_CONFIG_PATH)
    scenario_config = load_config(SCENARIO_CONFIG_PATH)
    simulator_config = load_config(SIMULATOR_CONFIG_PATH)

    env: gym.Env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode=None,
    )

    obs, _ = env.reset()
    # Check observation is in observation space
    assert env.observation_space.contains(obs), "Observation not in observation space"
    # Check observation satisfies pydantic model
    Observation.model_validate(obs)
    # Sample a new observation and check it's valid
    obs = env.observation_space.sample()
    # Validate against pydantic model
    Observation.model_validate(obs)

    # Sample action and check it's in action space
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Sampled action not in action space"
    # Validate action against pydantic model
    Action.model_validate(action)

    env.close()
