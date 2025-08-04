import yaml

from evodex.simulation import RobotHandEnv, SimulatorConfig
from evodex.simulation.robot import RobotConfig
from evodex.simulation.scenario import ScenarioConfig


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
    robot_config_dict = load_config("configs/base_robot.yaml")
    scenario_config_dict = load_config("configs/move_cube_scenario.yaml")
    simulator_config_dict = load_config("configs/base_simulator.yaml")

    robot_config = RobotConfig(**robot_config_dict)
    scenario_config = ScenarioConfig(**scenario_config_dict)
    simulator_config = SimulatorConfig(**simulator_config_dict)

    env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
    )

    env.reset()

    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
