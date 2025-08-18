from .robot import Observation as RobotObservation
from .scenario import Observation as ScenarioObservation
from .scenario import Goal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    Base class for observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    extrinsic: ScenarioObservation
    intrinsic: RobotObservation


class HERObservation(BaseModel):
    """
    Base class for observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    observation: Observation = Field(
        ..., description="Observation data from the robot and environment"
    )
    achieved_goal: Goal = Field(..., description="Achieved goal observation data")

    desired_goal: Goal = Field(
        ..., description="Goal observation data from the scenario"
    )
