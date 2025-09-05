import yaml
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .config import RobotConfig

THIS_DIR = Path(__file__).parent
TEMPLATE_FILE = "robot.urdf.j2"
TEMPLATES_DIR = THIS_DIR / "templates"


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
