from typing import Tuple
from pydantic import BaseModel, Field

def in_interval(
    value: float, interval: Tuple[float, float]
) -> bool:
    return interval[0] <= value <= interval[1]


class Scale(BaseModel):
    domain: Tuple[float, float] = Field(
        ..., description="Domain for scaling values"
    )

    target: Tuple[float, float] = Field(
        ..., description="Target scale range for values"
    )

    def rescale(self, value: float, inverse: bool = False):
        """
        Scales a value from the domain to the target range or vice versa if inverse is True.

        Args:
            value (float): The value to be scaled.
            inverse (bool): If True, scales from target to domain; otherwise scales from domain to target.

        Returns:
            float: The scaled value.
        """
        if not inverse:
            return value * self.gain + self.offset
        return (value - self.offset) / self.gain if self.gain != 0 else value
        
    
    @property
    def gain(self) -> float:
        """
        Returns the gain factor for scaling.
        """
        return (self.target[1] - self.target[0]) / (self.domain[1] - self.domain[0]) if self.domain[1] != self.domain[0] else 1.0
    
    @property
    def offset(self) -> float:
        """
        Returns the offset for scaling.
        """
        return self.target[0] - self.gain * self.domain[0] if self.domain[1] != self.domain[0] else 0.0


class NormalizedScale(Scale):
    """
    A scale that normalizes values to the range [0, 1].
    """
    def __init__(self, target: Tuple[float, float], **data):
        """
        Initializes the NormalizedScale with the specified target range.
        """
        super().__init__(domain=(-1.0, 1.0), target=target, **data)

    def scale(self, value: float) -> float:
        """
        Rescales a value to the normalized range [0, 1] or back.
        """
        return super().rescale(value)