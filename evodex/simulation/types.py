from .robot import IntrinsicObservation
from .scenario import Observation as ExtrinsicObservation
from .scenario import Goal

from pydantic import BaseModel, Field

class RobotObservation(BaseModel):
    """
    Base class for robot observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    extrinsic: ExtrinsicObservation
    intrinsic: IntrinsicObservation


class Observation(BaseModel):
    """
    Base class for observations in the simulation environment.
    This class can be extended to include specific observation data.
    """

    observation: RobotObservation
    desired_goal: Goal
    achieved_goal: Goal