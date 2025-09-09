import yaml
import os
from pathlib import Path

from .config import RobotConfig

CURRENT_DIR = Path(__file__).parent
TEMPLATE_FILE = "robot.urdf.j2"
TEMPLATES_DIR = CURRENT_DIR / "templates"


def load_config(file_path: str) -> RobotConfig:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return RobotConfig(**config)
