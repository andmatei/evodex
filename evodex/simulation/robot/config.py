from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple, List
import numpy as np


class GlobalSegmentConfig(BaseModel):
    mass: Optional[float] = Field(None, ge=0.0)
    motor_stiffness: Optional[float] = Field(None, ge=0.0)
    motor_damping: Optional[float] = Field(None, ge=0.0)
    joint_angle_limit: Optional[Tuple[float, float]] = Field(None)

    @model_validator(mode="after")
    def validate_joint_angle_limit(self) -> "GlobalSegmentConfig":
        if self.joint_angle_limit is not None:
            min_val, max_val = self.joint_angle_limit
            if min_val > max_val:
                raise ValueError(
                    "joint_angle_limit.min must be less than joint_angle_limit.max"
                )
            if min_val < -np.pi or max_val > np.pi:
                raise ValueError("joint_angle_limit must be between -pi and pi")
        return self
    

class SegmentConfig(GlobalSegmentConfig):
    length: float = Field(..., ge=0.0)
    width: float = Field(..., ge=0.0)


class FingerConfig(BaseModel):
    segments: List[SegmentConfig]
    defaults: Optional[GlobalSegmentConfig] = None

    @model_validator(mode="after")
    def validate_segments(self) -> "FingerConfig":
        if len(self.segments) == 0:
            raise ValueError("segments must be non-empty")
        return self

    @model_validator(mode="after")
    def validate_defaults(self) -> "FingerConfig":
        if self.defaults is not None:
            default_values = self.defaults.model_dump(exclude_none=True)
            for segment in self.segments:
                for key, value in default_values.items():
                    if getattr(segment, key, None) is None:
                        setattr(segment, key, value)

        global_fields = GlobalSegmentConfig.model_fields.keys()
        for i, segment in enumerate(self.segments):
            for field in global_fields:
                if getattr(segment, field) is None:
                    raise ValueError(f"Segment {i} must have '{field}' defined or in defaults")
        return self
    

class BaseConfig(BaseModel):
    width: float = Field(..., ge=0.0)
    height: float = Field(..., ge=0.0)
    mass: float = Field(..., gt=0.0)


class RobotConfig(BaseModel):
    base: BaseConfig
    fingers: List[FingerConfig]