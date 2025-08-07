import yaml
import gymnasium as gym
import numpy as np
import os

from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
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
    # 1. Load Configurations
    print("➡️  Loading configurations...")
    robot_config = load_config("configs/base_robot.yaml")
    scenario_config = load_config("configs/move_cube_scenario.yaml")
    simulator_config = load_config("configs/base_simulator.yaml")

    # 2. Create and Check the Custom Environment
    print("➡️  Initializing environment...")
    env: gym.Env = flatten(
        RobotHandEnv(
            robot_config=robot_config,
            scenario_config=scenario_config,
            env_config=simulator_config,
            render_mode="human",  # Set to 'human' for visual rendering
        ),
        observation=True,
        action=True,
    )
    env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))
    vec_env = make_vec_env(
        lambda: env,
        n_envs=1,
    )

    check_env(env, warn=True)

    # 3. Define the PPO Agent
    # We'll save logs and the trained model in a dedicated directory
    log_dir = "logs/base_robot_hand/"
    os.makedirs(log_dir, exist_ok=True)

    print("➡️  Defining the PPO model...")
    model = PPO(
        "MlpPolicy",  # Standard policy for continuous action/observation spaces
        vec_env,  # The vectorized environment
        verbose=1,  # Print training progress
        tensorboard_log=log_dir,  # Directory for TensorBoard logs
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # 4. Train the Agent
    print("🚀 Starting training...")
    # The total number of steps the agent will be trained for
    TRAINING_TIMESTEPS = 25000
    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    # 5. Save the Trained Model
    model_path = os.path.join(log_dir, "ppo_robot_hand_model.zip")
    model.save(model_path)
    print(f"✅ Training complete! Model saved to {model_path}")

    # --- Evaluate the Trained Agent ---
    print("\n👀 Starting evaluation of the trained agent...")
    del model  # remove the trained model from memory

    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    print(type(obs))  # Check the observation type
    for _ in range(1000):  # Run evaluation for 1000 steps
        # The model's predict method gets the best action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        if done:
            print("   Evaluation episode finished.")
            obs = vec_env.reset()

    vec_env.close()
    print("🏁 Evaluation finished.")
