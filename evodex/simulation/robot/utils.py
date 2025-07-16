from typing import Tuple, Optional
from pydantic import BaseModel, Field


def scale(
    value: float, scale: Tuple[float, float], domain: Tuple[float, float] = (-1.0, 1.0)
) -> float:
    """
    Scales a value from the range [domain[0], domain[1]] to the range [scale[0], scale[1]].

    Args:
        value (float): The value to be scaled.
        scale (Tuple[float, float]): The target scale range.
        domain (Tuple[float, float], optional): The original domain range. Defaults to [-1.0, 1.0].

    Returns:
        float: The scaled value.
    """
    return scale[0] + (value - domain[0]) * (scale[1] - scale[0]) / (
        domain[1] - domain[0]
    )

def in_interval(
    value: float, interval: Tuple[float, float]
) -> bool:
    return interval[0] <= value <= interval[1]


class Scale(BaseModel):
    domain: Tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Domain for scaling values"
    )

    target: Tuple[float, float] = Field(
        ..., description="Target scale range for values"
    )

    def rescale(self, value: float, normalise: bool = False):
        """
        Scales a value from the domain to the target range or vice versa if inverse is True.

        Args:
            value (float): The value to be scaled.
            inverse (bool): If True, scales from target to domain; otherwise scales from domain to target.

        Returns:
            float: The scaled value.
        """
        if normalise:
            return scale(value, self.domain, self.target)
        else:
            return scale(value, self.target, self.domain)
