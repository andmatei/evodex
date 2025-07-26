from .robot import Observation as RobotObservation
from .scenario import Observation as ScenarioObservation

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Base class for observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    robot: RobotObservation = Field(
        ..., description="Intrinsic observation data of the robot"
    )
    scenario: ScenarioObservation = Field(
        ..., description="Extrinsic observation data from the scenario"
    )
