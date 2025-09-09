import os

isaaclab_home = os.environ.get("ISAACLAB_HOME")
if isaaclab_home:
    os.chdir(isaaclab_home)

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a URDF into USD format."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True  # ensure headless mode

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from evodex.simulation.isaac.robot.utils import load_config
from evodex.simulation.isaac.robot.urdf import save_urdf
from evodex.simulation.isaac.robot.usd import convert_urdf_to_usd
from evodex.core.paths import PROJECT_ROOT

if __name__ == "__main__":
    # --- Main script logic ---
    # Load the configuration
    robot_config_path = f"{PROJECT_ROOT}/configs/robot/3d/gripper_robot.yaml"
    robot_config = load_config(robot_config_path)

    # Save the URDF file
    output_path = f"{PROJECT_ROOT}/generated/test.urdf"
    save_urdf(robot_config, output_path)

    # Convert the URDF to USD
    usd_output_path = f"{PROJECT_ROOT}/generated/test2.usd"
    convert_urdf_to_usd(output_path, usd_output_path)
    print(f"✅ Successfully converted URDF to USD file at: {usd_output_path}")
