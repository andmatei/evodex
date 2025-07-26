from pydantic import BaseModel, Field
from typing import Any, Dict, Tuple


class Observation(BaseModel):
    """
    Base class for observations in scenarios.
    This class can be extended to include specific observation data.
    """

    velocity: Tuple[float, float] = Field(
        ..., description="Velocity of the object being observed"
    )
    position: Tuple[float, float] = Field(
        ..., description="Position of the object being observed"
    )
    angle: float = Field(..., description="Angle of the object being observed")
    angular_velocity: float = Field(
        ..., description="Angular velocity of the object being observed"
    )

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional data related to the observation, if any",
    )
