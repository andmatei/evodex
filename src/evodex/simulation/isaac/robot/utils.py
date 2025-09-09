import yaml
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from .config import RobotConfig

CURRENT_DIR = Path(__file__).parent
TEMPLATE_FILE = "robot.urdf.j2"
TEMPLATES_DIR = CURRENT_DIR / "templates"


def load_config(file_path: str) -> RobotConfig:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return RobotConfig(**config)


def save_urdf(config: RobotConfig, o_path: str) -> None:
    """
    Generates a URDF file from a RobotConfig object using a template file.

    Args:
        config (RobotConfig): The validated Pydantic model of the robot.
        template_path (str): The path to the Jinja2 URDF template file.
        output_path (str): The file path where the URDF will be saved.
    """
    try:
        # 1. Set up Jinja2 environment to load templates from the correct directory.
        env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR), trim_blocks=True, lstrip_blocks=True
        )
        template = env.get_template(TEMPLATE_FILE)

        # 2. Render the template with the config object
        urdf_content = template.render(config=config)

        # 3. Ensure the output directory exists
        output_path = Path(o_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Write the rendered content to the specified file
        with output_path.open("w") as f:
            f.write(urdf_content)

        print(f"✅ Successfully generated URDF file at: {output_path}")
    except Exception as e:
        print(f"❌ An error occurred during URDF generation: {e}")


def convert_urdf_to_usd(urdf_path: str, output_path: str) -> UrdfConverter:
    """
    Converts a URDF file to USD format using Isaac Sim's command-line tool.

    Args:
        urdf_path (str): The path to the input URDF file.
        usd_path (str): The path where the output USD file will be saved.
    """
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)

    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)

    output_dir = os.path.dirname(output_path)
    output_file_name = os.path.basename(output_path)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name=output_file_name,
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=50.0,
                damping=1.0,
            ),
            target_type="position",
            drive_type="acceleration",
        ),
    )

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    return urdf_converter
