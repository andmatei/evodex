import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from experiments.utils import load_config, make_env, EnvMask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/move_to_target/")
    parser.add_argument(
        "--scenario-config",
        "-sc",
        type=str,
        default="configs/scenario/move_to_target_scenario.yaml",
    )
    parser.add_argument(
        "--robot-config", "-rc", type=str, default="configs/robot/base_robot.yaml"
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    robot_config = args.robot_config
    scenario_config = args.scenario_config

    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config(robot_config)
    scenario_config = load_config(scenario_config)
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    eval_env = make_vec_env(
        lambda: make_env(
            robot_config=robot_config,
            scenario_config=scenario_config,
            simulator_config=simulator_config,
            render_mode="human",
            mask=EnvMask.BASE,
        ),
        n_envs=1,
    )

    # 3. Define the PPO Agent
    print("➡️  Loading the trained PPO model...")
    model_path = os.path.join(log_dir, "model.latest.zip")

    model = PPO.load(model_path, env=eval_env)

    # 4. Evaluate the Trained Agent
    print("➡️  Starting evaluation...")
    obs = eval_env.reset()
    for _ in range(10000):
        # The model's predict method gets the best action
        action, _ = model.predict(obs, deterministic=True)  # type: ignore
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        if done.any():
            print("   Evaluation episode finished.")
            obs = eval_env.reset()

    eval_env.close()
    print("🏁 Evaluation finished.")
