from evodex.simulation.isaac.robot.config import RobotConfig
from evodex.simulation.isaac.robot.utils import load_config, save_urdf

if __name__ == "__main__":
    # --- Main script logic ---
    # Load the configuration
    robot_config_path = "./configs/robot/3d/hand_robot.yaml"
    robot_config = load_config(robot_config_path)

    # Save the URDF file
    output_path = "./generated/robot.urdf"
    save_urdf(robot_config, output_path)