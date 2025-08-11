import gymnasium as gym

from evodex.simulation import RobotHandEnv
from evodex.simulation.wrapper import flatten_env

from experiments.utils import load_config


if __name__ == "__main__":
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/base_robot.yaml")
    scenario_config = load_config("configs/move_cube_scenario.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    train_env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode=None,
    )

    action = train_env.action_space.sample()
    print(f"Sampled action: {action}")
    print(
        f"Flattened action space: {gym.spaces.utils.flatten_space(train_env.action_space)}"
    )
    print(
        f"Flattened action: {gym.spaces.utils.flatten(train_env.action_space, action)}"
    )
