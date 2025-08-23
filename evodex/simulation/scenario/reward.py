import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .core import Scenario
from .types import Observation as ScenarioObservation, Goal

from evodex.simulation.robot import Robot, IntrinsicObservation as RobotObservation


class RewardFunction(ABC):
    def __init__(self, weight: float = 1.0):
        self._weight = weight
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        pass

    def reset(self):
        self._state = {}

    def __call__(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        """
        Compute the reward given the robot and scenario observation.
        """
        return (
            self._calculate_reward(intrinsic_obs, extrinsic_obs, achieved_goal, goal)
            * self._weight
        )


class CompositeRewardFunction(RewardFunction):
    def __init__(
        self,
        reward_functions: List[RewardFunction] = [],
    ):
        super().__init__()
        self._reward_functions = reward_functions

    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        return sum(
            rf(intrinsic_obs, extrinsic_obs, achieved_goal, goal)
            for rf in self._reward_functions
        )

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
        return CompositeRewardFunction(self._reward_functions)


class GraspingReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
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
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        goal_position = np.array(goal.position)
        achieved_position = np.array(achieved_goal.position)
        distance = np.linalg.norm(achieved_position - goal_position)

        prev_distance = self._state.get("prev_distance")
        reward = 0.0
        if prev_distance is not None:
            reward = prev_distance - distance

        self._state["prev_distance"] = distance

        return reward


class LiftReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        return 0.0
