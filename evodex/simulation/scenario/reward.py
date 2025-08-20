import numpy as np

from abc import ABC, abstractmethod
from typing import List, Optional

from .core import Scenario

from evodex.simulation.robot import Robot


class RewardFunction(ABC):
    def __init__(self, robot: Robot, scenario: Scenario, weight: float = 1.0):
        self._robot = robot
        self._scenario = scenario
        self._weight = weight

    @abstractmethod
    def _calculate_reward(self) -> float:
        pass

    def __call__(self) -> float:
        """
        Compute the reward given the robot and scenario observation.
        """
        return self._calculate_reward() * self._weight


class CompositeRewardFunction(RewardFunction):
    def __init__(
        self,
        robot: Robot,
        scenario: Scenario,
        reward_functions: List[RewardFunction] = [],
    ):
        super().__init__(robot, scenario)
        self._reward_functions = reward_functions

    def _calculate_reward(self) -> float:
        return sum(rf() for rf in self._reward_functions)

    def add(self, reward_function: RewardFunction) -> None:
        self._reward_functions.append(reward_function)

    def extend(self, reward_functions: List[RewardFunction]) -> None:
        self._reward_functions.extend(reward_functions)


class RewardBuilder:
    def __init__(
        self,
        robot: Robot,
        scenario: Scenario,
        reward_functions: List[RewardFunction] = [],
    ):
        self._robot = robot
        self._scenario = scenario
        self._reward_functions = reward_functions

    def add(self, reward_function: RewardFunction) -> "RewardBuilder":
        self._reward_functions.append(reward_function)
        return self

    def build(self) -> RewardFunction:
        return CompositeRewardFunction(
            self._robot, self._scenario, self._reward_functions
        )


class GraspingReward(RewardFunction):
    def _calculate_reward(self) -> float:
        intrinsic_obs = self._robot.get_intrinsic_observation()

        contact_points = sum(
            1
            for finger in intrinsic_obs.fingers
            for segment in finger.segments
            if segment.is_touching
        )

        is_grasping = (
            contact_points > 1
        )  # Grasping is defined as at least 2 contact points
        if is_grasping:
            return contact_points

        return 0.0


class TargetReward(RewardFunction):
    def _calculate_reward(self) -> float:
        achieved_goal = self._scenario.get_achieved_goal(self._robot)
        goal = self._scenario.get_goal(self._robot)

        goal_position = np.array(goal.position)
        achieved_position = np.array(achieved_goal.position)
        distance = np.linalg.norm(achieved_position - goal_position)

        return -float(distance)
