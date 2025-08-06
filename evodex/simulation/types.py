from .robot import Observation as RobotObservation
from .scenario import Observation as ScenarioObservation

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Base class for observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    observation: RobotObservation = Field(
        ..., description="Intrinsic observation data of the robot"
    )
    achieved_goal: ScenarioObservation = Field(
        ..., description="Extrinsic observation data from the scenario"
    )

    desired_goal: ScenarioObservation = Field(
        ..., description="Goal observation data from the scenario"
    )
