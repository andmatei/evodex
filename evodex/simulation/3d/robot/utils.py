import yaml

from .config import RobotConfig

def load_config(file_path: str) -> RobotConfig:
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return RobotConfig(**config)

def save_urdf(config: RobotConfig, path: str) -> None:
    pass