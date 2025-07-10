from evodex.simulation.scenario import (
    MoveCubeToTargetScenario,
    ExtractCubeFromTubeScenario,
)
from evodex.simulation.env import RobotHandEnv
from evodex.simulation.simulation import Simulation
import numpy as np
from evodex.simulation.config import DEFAULT_ROBOT_CONFIG
import pygame

# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    current_run_robot_config = DEFAULT_ROBOT_CONFIG.copy()
    current_run_robot_config["base"]["initial_position"] = (
        400,
        300,
    )  # Center the base initially
    current_run_robot_config["simulation"]["key_move_speed"] = 200  # Adjusted speed

    sim_config_global = current_run_robot_config["simulation"]
    # scenario_to_run = ExtractCubeFromTubeScenario(sim_config_global)
    scenario_to_run = MoveCubeToTargetScenario(sim_config_global)

    env = RobotHandEnv(
        robot_config=current_run_robot_config,
        scenario_instance=scenario_to_run,
        max_episode_steps=700,
    )

    print(f"Running with scenario: {scenario_to_run.__class__.__name__}")
    print(f"Robot base initial position: {env.simulation.robot.base.body.position}")
    print("Use arrow keys to move the robot base. Press 'r' to reset the simulation.")

    for episode in range(5):  # Increased episodes for more testing
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        # Manually control or let agent act
        manual_control_active_this_episode = (
            True  # Set to False to use agent actions (e.g. random)
        )

        while not (terminated or truncated):
            # Handle Pygame events for keyboard control and window closing
            quit_attempted = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    truncated = True
                    quit_attempted = True
                    break
                env.simulation.handle_pygame_event(
                    event
                )  # Process keyboard input for base movement
            if quit_attempted:
                break

            if manual_control_active_this_episode:
                action = np.zeros(
                    env.action_space.shape
                )  # No motor actions if manually controlling base
            else:
                action = env.action_space.sample()  # Agent takes random actions

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            env.render()

        print(
            f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Steps = {step_count}"
        )
        if terminated and not truncated:
            print("  Terminated (task specific condition).")
        if truncated and step_count < env.max_episode_steps and quit_attempted:
            print("  Truncated (QUIT event).")
        elif truncated:
            print(
                "  Truncated (max steps reached or QUIT event without task completion)."
            )

        if truncated and quit_attempted:  # If QUIT event happened during episode
            break
    env.close()
    print("Simulation finished.")
