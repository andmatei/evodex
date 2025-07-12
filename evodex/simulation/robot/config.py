from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple, List, Any
import numpy as np


class GlobalSegmentConfig(BaseModel):
    mass: Optional[float] = Field(default=None, ge=0.0)
    motor_stiffness: Optional[float] = Field(default=None, ge=0.0)
    motor_damping: Optional[float] = Field(default=None, ge=0.0)
    joint_angle_limit: Optional[Tuple[float, float]] = Field(default=None)

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


class SegmentConfig(BaseModel):
    length: float = Field(..., ge=0.0)
    width: float = Field(..., ge=0.0)

    mass: float = Field(..., ge=0.0)
    motor_stiffness: float = Field(..., ge=0.0)
    motor_damping: float = Field(..., ge=0.0)
    joint_angle_limit: Tuple[float, float]

    @model_validator(mode="after")
    def validate_joint_angle_limit(self) -> "SegmentConfig":
        if self.joint_angle_limit is not None:
            min_val, max_val = self.joint_angle_limit
            if min_val > max_val:
                raise ValueError(
                    "joint_angle_limit.min must be less than joint_angle_limit.max"
                )
            if min_val < -np.pi or max_val > np.pi:
                raise ValueError("joint_angle_limit must be between -pi and pi")
        return self


class FingerConfig(BaseModel):
    segments: List[SegmentConfig]
    defaults: Optional[GlobalSegmentConfig] = None

    @model_validator(mode="after")
    def validate_segments(self) -> "FingerConfig":
        if len(self.segments) == 0:
            raise ValueError("segments must be non-empty")
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        defaults = data.get("defaults")
        segments = data.get("segments")

        if not defaults or not segments:
            return data

        default_values = GlobalSegmentConfig.model_validate(defaults).model_dump(
            exclude_none=True
        )
        for segment in segments:
            for key, value in default_values.items():
                segment.setdefault(key, value)

        return data


class BaseConfig(BaseModel):
    width: float = Field(..., ge=0.0)
    height: float = Field(..., ge=0.0)
    mass: float = Field(..., gt=0.0)


class RobotConfig(BaseModel):
    base: BaseConfig
    fingers: List[FingerConfig]
