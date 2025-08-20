import gymnasium as gym

from evodex.simulation import RobotHandEnv
from evodex.simulation.robot.spaces import Action, BaseAction

from experiments.utils import load_config

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

    action = env.action_space.sample()

    obs, _ = env.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
