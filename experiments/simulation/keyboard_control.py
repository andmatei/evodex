import gymnasium as gym

from evodex.simulation import RobotHandEnv
from evodex.simulation.robot.spaces import Action, BaseAction

from experiments.utils import load_config

# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/robot/base_robot.yaml")
    scenario_config = load_config("configs/scenario/move_to_target_scenario.yaml")
    simulator_config = load_config("configs/keyboard_control_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode="human",  # Set to 'human' for visual rendering
    )

    obs, _ = env.reset()
    while True:
        action = env.controller.get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
