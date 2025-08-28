import re
import numpy as np

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Type

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


class RewardRegistry:
    _registry: Dict[str, Type[RewardFunction]] = {}

    @classmethod
    def _normalise_name(cls, name: str) -> str:
        # Alter the name of the reward function to match snake case
        name = name.removesuffix("Reward")
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return name

    @classmethod
    def register(cls, reward_function: Type[RewardFunction]) -> None:
        name = cls._normalise_name(reward_function.__name__)
        print(f"Registering reward function: {name}")
        cls._registry[name] = reward_function

    @classmethod
    def get(cls, name: str) -> Type[RewardFunction]:
        reward_function = cls._registry.get(name)
        if reward_function is None:
            raise ValueError(f"Reward function '{name}' not found.")
        return reward_function


class RewardConfig(BaseModel):
    name: str = Field(..., description="Name of the reward function")
    weight: float = Field(1.0, description="Weight of the reward function")


class RewardBuilder:
    def __init__(
        self,
        reward_functions: List[RewardFunction] = [],
    ):
        self._reward_functions = reward_functions

    def add(self, reward_function: RewardFunction) -> "RewardBuilder":
        self._reward_functions.append(reward_function)
        return self

    def build(self) -> RewardFunction:
        return CompositeRewardFunction(self._reward_functions)
    
    @staticmethod
    def from_config(reward_configs: List[RewardConfig]) -> RewardFunction:
        builder = RewardBuilder()
        for config in reward_configs:
            reward_class = RewardRegistry.get(config.name)
            reward_function = reward_class(weight=config.weight)
            builder.add(reward_function)
        return builder.build()


@RewardRegistry.register
class GraspReward(RewardFunction):
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


@RewardRegistry.register
class MoveReward(RewardFunction):
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


@RewardRegistry.register
class LiftReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        return 0.0


@RewardRegistry.register
class StabilityReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        return 0.0


@RewardRegistry.register
class ReachReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        return 0.0


@RewardRegistry.register
class SuccessReward(RewardFunction):
    def _calculate_reward(
        self,
        intrinsic_obs: RobotObservation,
        extrinsic_obs: ScenarioObservation,
        achieved_goal: Goal,
        goal: Goal,
    ) -> float:
        if np.linalg.norm(np.array(achieved_goal.position) - np.array(goal.position)) < 0.1:
            return 1.0
        return 0.0
