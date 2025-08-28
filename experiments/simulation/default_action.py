import gymnasium as gym

from evodex.simulation import RobotHandEnv
from evodex.simulation.robot.spaces import Action, BaseAction

from experiments.utils import load_config

# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/robot/base_robot.yaml")
    scenario_config = load_config("configs/scenario/grasping/cube.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    env: gym.Env = RobotHandEnv(
        robot_config=robot_config,
        scenario_config=scenario_config,
        env_config=simulator_config,
        render_mode="human",  # Set to 'human' for visual rendering
    )

    TIME_SWAP = 100
    timestep = 0

    obs, _ = env.reset()
    while True:
        if timestep % TIME_SWAP == 0:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        timestep += 1
