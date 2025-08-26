import gymnasium as gym

from evodex.simulation import RobotHandEnv, Observation

from experiments.utils import load_config

# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/robot/base_robot.yaml")
    scenario_config = load_config("configs/scenario/move_to_target_scenario.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    env: gym.Env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode="human",  # Set to 'human' for visual rendering
    )

    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
