import math
from pydantic import BaseModel, Field, model_validator, computed_field
from typing import List, Sequence, Tuple, Literal, Union, Optional
from abc import ABC, abstractmethod
from functools import cached_property
from enum import Enum

# =============================================================== #
#                       Shape Configurations                      #
# =============================================================== #
# Defines the different types of geometric shapes that can be used
# for visual or collision properties of a link.

class Inertia(BaseModel):
    """Holds the calculated values for the inertia tensor."""
    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0

    def __mul__(self, other: float) -> "Inertia":
        return Inertia(
            ixx=self.ixx * other,
            ixy=self.ixy * other,
            ixz=self.ixz * other,
            iyy=self.iyy * other,
            iyz=self.iyz * other,
            izz=self.izz * other,
        )

    def __rmul__(self, other: float) -> "Inertia":
        return self.__mul__(other)


class GeometryType(str, Enum):
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"


class GeometryConfig(BaseModel, ABC):
    type: GeometryType = Field(..., description="Type of the shape")

    @abstractmethod
    def _calculate_unit_inertia(self) -> Inertia:
        pass

    @computed_field
    @property
    def unit_inertia(self) -> Inertia:
        return self._calculate_unit_inertia()


class BoxConfig(GeometryConfig):
    type: Literal[GeometryType.BOX] = GeometryType.BOX
    width: float = Field(..., description="Width of the box along the X-axis")
    length: float = Field(..., description="Length of the box along the Y-axis")
    depth: float = Field(..., description="Height of the box along the Z-axis")

    def _calculate_unit_inertia(self) -> Inertia:
        x, y, z = self.width, self.length, self.depth
        ixx = (1/12) * (y**2 + z**2)
        iyy = (1/12) * (x**2 + z**2)
        izz = (1/12) * (x**2 + y**2)
        return Inertia(ixx=ixx, iyy=iyy, izz=izz)


class CylinderConfig(GeometryConfig):
    type: Literal[GeometryType.CYLINDER] = GeometryType.CYLINDER
    radius: float = Field(..., description="Radius of the cylinder")
    length: float = Field(..., description="Depth (height) of the cylinder")

    def _calculate_unit_inertia(self) -> Inertia:
        r, h = self.radius, self.length
        ixx = (1/12) * (3 * r**2 + h**2)
        iyy = ixx
        izz = 0.5 * r**2
        return Inertia(ixx=ixx, iyy=iyy, izz=izz)


class CapsuleConfig(GeometryConfig):
    type: Literal[GeometryType.CAPSULE] = GeometryType.CAPSULE
    radius: float = Field(..., description="Radius of the capsule")
    length: float = Field(..., description="Depth (height) of the capsule")

    def _calculate_unit_inertia(self) -> Inertia:
        r, l = self.radius, self.length
        ixx = (1/12) * (3*r**2 + l**2)
        iyy = ixx
        izz = 0.5 * r**2
        return Inertia(ixx=ixx, iyy=iyy, izz=izz)


class SphereConfig(GeometryConfig):
    type: Literal[GeometryType.SPHERE] = GeometryType.SPHERE
    radius: float = Field(..., description="Radius of the sphere")

    def _calculate_unit_inertia(self) -> Inertia:
        r = self.radius
        i = (2/5) * r**2
        return Inertia(ixx=i, iyy=i, izz=i)


AllGeometryConfigs = Union[BoxConfig, SphereConfig, CylinderConfig, CapsuleConfig]

# =============================================================== #
#                       Link Configurations                       #
# =============================================================== #
# Defines the properties of a single link in the robot's kinematic chain.

class LinkConfig(BaseModel):
    """A generic configuration for any link (e.g., a finger segment)."""
    name: str = Field(..., description="A unique name for this link part (e.g., 'proximal').")
    mass: float = Field(..., gt=0, description="The mass of the link in kilograms.")
    geometry: AllGeometryConfigs = Field(..., description="The geometry of the link.", discriminator="type")
    
    @computed_field
    @property
    def inertia(self) -> Inertia:
        return self.geometry.unit_inertia * self.mass


class BaseConfig(LinkConfig):
    """Configuration for the robot's base (palm)."""
    pass

# =============================================================== #
#                      Finger Configurations                      #
# =============================================================== #
# Defines the structure of a complete finger, from its attachment
# point on the base to its segments and fingertip.
class FingerAttachmentConfig(BaseModel):
    angle: float = Field(..., description="Angle in radians around the Z-axis where the finger attaches to the base.")
    radius: float = Field(..., description="The radial distance from the base center to the finger attachment point.")
    z_offset: float = Field(default=0.0, description="The vertical offset (Z-axis) of the finger attachment point.")
    yaw_offset: float = Field(default=0.0, description="The local rotation of the finger around its own axis in radians.")

    @computed_field
    @property
    def origin(self) -> Optional[Tuple[float, float, float]]:
        if self.radius is not None and self.z_offset is not None:
            x = self.radius * math.cos(self.angle)
            y = self.radius * math.sin(self.angle)
            z = self.z_offset
            return (x, y, z)
        return None


# TODO: Add segment override
class FingerDefaultsConfig(BaseModel):
    angle_limit: Tuple[float, float] = Field(..., description="The min and max angle limits for the finger joints.")
    damping: float = Field(..., description="The damping factor for the finger joints.")


class FingerConfig(BaseModel):
    defaults: FingerDefaultsConfig = Field(..., description="Default properties applied to all segments unless overridden.")
    attachment: FingerAttachmentConfig = Field(..., description="Attachment configuration for the finger.")
    segments: Tuple[LinkConfig, ...] = Field(..., description="A list of link configurations representing the finger's segments.")
    fingertip: Optional[LinkConfig] = Field(None, description="Optional fingertip configuration.")

    @model_validator(mode="after")
    def validate_segments(self) -> "FingerConfig":
        if len(self.segments) == 0:
            raise ValueError("segments must be non-empty")
        return self


# =============================================================== #
#                       Top-Level Configuration                   #
# =============================================================== #
# The main models that encompass the entire robot morphology.

class RobotConfig(BaseModel):
    base: BaseConfig
    fingers: Tuple[FingerConfig, ...]