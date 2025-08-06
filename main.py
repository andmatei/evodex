import yaml

from stable_baselines3.common.env_checker import check_env

from evodex.simulation import RobotHandEnv
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
    robot_config = load_config("configs/base_robot.yaml")
    scenario_config = load_config("configs/move_cube_scenario.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    env = flatten(
        RobotHandEnv(
            robot_config=robot_config,
            scenario_config=scenario_config,
            env_config=simulator_config,
        ),
        observation=True,
        action=True,
    )

    print(env.reset()[0])

    check_env(env, warn=True)

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
