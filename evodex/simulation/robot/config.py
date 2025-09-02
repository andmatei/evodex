from pydantic import BaseModel, Field, model_validator
from typing import Optional, Tuple, Any
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
    segments: Tuple[SegmentConfig, ...]
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

    def __len__(self) -> int:
        return len(self.segments)


class BaseConfig(BaseModel):
    width: float = Field(..., ge=0.0)
    height: float = Field(..., ge=0.0)
    mass: float = Field(..., gt=0.0)


class LimitConfig(BaseModel):
    min: float = Field(..., description="Minimum limit value")
    max: float = Field(..., description="Maximum limit value")

    @model_validator(mode="after")
    def validate_limits(self) -> "LimitConfig":
        if self.min >= self.max:
            raise ValueError("min must be less than max")
        return self


# TODO: Include this as metadata into the action object
class ActionLimitsConfig(BaseModel):
    velocity: LimitConfig = Field(..., description="Scale for base velocity ")
    omega: LimitConfig = Field(..., description="Scale for base angular velocity")
    motor_rate: LimitConfig = Field(..., description="Scale for finger motor rates")


class RobotConfig(BaseModel):
    base: BaseConfig
    fingers: Tuple[FingerConfig, ...]
    limits: ActionLimitsConfig = Field(
        default_factory=lambda: ActionLimitsConfig(
            velocity=LimitConfig(min=-100, max=100),
            omega=LimitConfig(min=-np.pi / 4, max=np.pi / 4),
            motor_rate=LimitConfig(min=-np.pi / 4, max=np.pi / 4),
        )
    )

    def __len__(self) -> int:
        return len(self.fingers)
