import os
from pathlib import Path

isaaclab_home = os.environ.get("ISAACLAB_HOME")
if isaaclab_home:
    os.chdir(isaaclab_home)

import argparse

from isaaclab.app import AppLauncher
from evodex.core.paths import PROJECT_ROOT

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Generate a URDF files from robot configurations."
)
parser.add_argument(
    "--configs",
    nargs="+",
    type=str,
    required=True,
    help="Path to the robot configuration files.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=f"{PROJECT_ROOT}/generated/",
    help="Path to the output directory for URDF and USD files.",
)
args = parser.parse_args()

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

if __name__ == "__main__":
    # --- Main script logic ---
    # Load the configuration

    robot_config_paths = args.configs
    for robot_config_path in robot_config_paths:
        if not os.path.isabs(robot_config_path):
            robot_config_path = os.path.abspath(f"{PROJECT_ROOT}/{robot_config_path}")

        output_name = Path(robot_config_path).stem

        output_path = f"{args.output_dir}/{output_name}"
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(f"{PROJECT_ROOT}/{output_path}")

        urdf_path = f"{output_path}.urdf"
        usd_output_path = f"{output_path}.usd"

        # Load the configuration
        robot_config = load_config(robot_config_path)

        # Save the URDF file
        save_urdf(robot_config, urdf_path)

        # Convert the URDF to USD
        convert_urdf_to_usd(urdf_path, usd_output_path)
        print(
            f"✅ Successfully converted robot config to USD file at: {usd_output_path}"
        )
