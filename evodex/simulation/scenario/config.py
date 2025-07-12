from pydantic import BaseModel, Field
from typing import Tuple


class ScenarioConfig(BaseModel):
    name: str = Field(..., description="Scenario name")
    start_position: Tuple[float, float] = Field(..., description="Start position")
