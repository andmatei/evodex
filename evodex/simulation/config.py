import numpy as np
from typing import Tuple, Optional
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from evodex.simulation.utils import NormalizedScale

DEFAULT_ACTION_LIMITS = {
    "velocity": {"min": -100.0, "max": 100.0},
    "omega": {"min": -np.pi / 4, "max": np.pi / 4},
    "motor_rate": {"min": -np.pi / 4, "max": np.pi / 4},
}

DEFAULT_ROBOT_CONFIG = {
    "base": {
        "width": 30,
        "height": 100,
    },
    "fingers": [
        {
            "num_segments": 3,
            "segment_lengths": [50, 40, 30],
            "segment_widths": [30, 25, 20],
            "motor_stiffness": 1e7,
            "motor_damping": 1e5,
            "joint_angle_limit_min": -np.pi / 4,
            "joint_angle_limit_max": np.pi / 4,
            "fingertip_shape": "circle",
            "fingertip_radius": 7,
            "fingertip_size": (10, 5),
            "motor_max_force": 5e7,
        },
        {
            "num_segments": 1,
            "segment_length": 50,
            "segment_width": 12,
            "motor_stiffness": 1e7,
            "motor_damping": 1e5,
            "joint_angle_limit_min": -np.pi / 3,
            "joint_angle_limit_max": np.pi / 3,
            "fingertip_shape": "rectangle",
            "fingertip_radius": 6,
            "fingertip_size": (12, 6),
            "motor_max_force": 5e7,
        },
    ],
}

DEFAULT_SCENARIO_CONFIG = {
    "type": "MoveCubeToTargetScenario",  # Default scenario type
    "target_position": (600, 300),  # Default target position for scenarios
}

DEFAULT_SIMULATION_CONFIG = {
    "dt": 1.0 / 60.0,
    "gravity": (0, 900),
    "screen_width": 800,
    "screen_height": 600,
}

DEFAULT_RENDER_CONFIG = {
    "enabled": True,
    "fps": 60,
    "draw_options": {
        "draw_fps": True,
    },
}

DEFAULT_LOGGING_CONFIG = {
    "enabled": True,
    "log_level": "INFO",
    "log_file": "simulation.log",
}

DEFAULT_KEYBOARD_CONTROL_CONFIG = {
    "enabled": True,
    "move_speed": 150,
    "angular_speed": 1.5,
}


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SimulationConfig(BaseModel):
    dt: float = Field(..., description="Simulation time step")
    gravity: Tuple[float, float] = Field(..., description="Simulation gravity")
    screen_width: int = Field(..., description="Simulation screen width")
    screen_height: int = Field(..., description="Simulation screen height")
    max_steps: Optional[int] = Field(
        None, description="Maximum number of simulation steps"
    )


class DrawOptionsConfig(BaseModel):
    draw_fps: bool = Field(default=False, description="Draw fps")
    draw_collision_shapes: bool = Field(
        default=False, description="Draw collision shapes"
    )
    draw_constraints: bool = Field(default=False, description="Draw constraints")
    draw_kinematics: bool = Field(default=False, description="Draw kinematics")


class RenderConfig(BaseModel):
    enabled: bool = Field(default=True, description="Render enabled")
    fps: int = Field(default=60, description="Render fps")
    draw_options: DrawOptionsConfig = Field(
        default=DrawOptionsConfig(), description="Draw options"
    )


class LoggingConfig(BaseModel):
    enabled: bool = Field(default=True, description="Logging enabled")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_file: str = Field(default="simulation.log", description="Logging file")


class KeyboardControlConfig(BaseModel):
    enabled: bool = Field(default=True, description="Keyboard control enabled")
    move_speed: float = Field(default=150, description="Keyboard move speed")
    angular_speed: float = Field(default=1.5, description="Keyboard angular speed")


class SimulatorConfig(BaseModel):
    simulation: SimulationConfig = Field(..., description="Simulation config")
    render: RenderConfig = Field(..., description="Render config")
    logging: LoggingConfig = Field(..., description="Logging config")
    keyboard_control: KeyboardControlConfig = Field(
        ..., description="Keyboard control config"
    )

    @model_validator(mode="after")
    def validate_config(self):
        if self.keyboard_control.enabled and not self.render.enabled:
            raise ValueError("Keyboard control is enabled but render is disabled")
        return self
