import numpy as np
from typing import Tuple, Optional
from pydantic import BaseModel, Field, model_validator
from enum import Enum

DEFAULT_SCENARIO_CONFIG = {
    "type": "MoveCubeToTargetScenario",  # Default scenario type
    "target_position": (600, 300),  # Default target position for scenarios
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


# TODO: Add optional fields
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
