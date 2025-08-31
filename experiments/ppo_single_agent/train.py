import torch
import os
import argparse

from experiments.utils import load_config, make_env

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs/ppo_single_agent/")
    parser.add_argument(
        "--scenario-config",
        "-sc",
        type=str,
        default="configs/scenario/grasping/cube.yaml",
    )
    parser.add_argument(
        "--robot-config", "-rc", type=str, default="configs/robot/base_robot.yaml"
    )

    args = parser.parse_args()
    log_dir = args.log_dir
    scenario_config = args.scenario_config
    robot_config = args.robot_config

    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config(robot_config)
    scenario_config = load_config(scenario_config)
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    train_env = make_vec_env(
        lambda: make_env(
            robot_config=robot_config,
            scenario_config=scenario_config,
            simulator_config=simulator_config,
        ),
        n_envs=4,
    )

    check_env(
        make_env(
            robot_config=robot_config,
            scenario_config=scenario_config,
            simulator_config=simulator_config,
        ),
        warn=True,
    )

    # 3. Define the PPO Agent
    os.makedirs(log_dir, exist_ok=True)

    print("➡️  Defining the PPO model...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",  # Standard policy for continuous action/observation spaces
        train_env,  # The vectorized environment
        verbose=1,  # Print training progress
        tensorboard_log=log_dir,  # Directory for TensorBoard logs
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )

    # 4. Train the Agent
    print("🚀 Starting training...")
    TRAINING_TIMESTEPS = 20_000_000
    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    # 5. Save the Trained Model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = os.path.join(log_dir, f"model_{timestamp}.zip")
    model_path_latest = os.path.join(log_dir, "model.latest.zip")

    model.save(model_path_latest)
    model.save(model_path)

    print(f"✅ Training complete! Model saved to {model_path_latest}")
