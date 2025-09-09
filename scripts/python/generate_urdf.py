import argparse

from evodex.simulation.isaac.robot.utils import load_config, save_urdf

parser = argparse.ArgumentParser(
    description="Generate a URDF file from a robot configuration."
)
parser.add_argument(
    "--config",
    type=str,
    default="./configs/robot/3d/gripper_robot.yaml",
    help="Path to the robot configuration file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="./generated/robot.urdf",
    help="Path to the output URDF file.",
)
args = parser.parse_args()


def main():
    # Load the configuration
    robot_config = load_config(args.config)

    # Save the URDF file
    save_urdf(robot_config, args.output)


if __name__ == "__main__":
    main()
