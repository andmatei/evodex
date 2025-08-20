import numpy as np
import pymunk
import pygame

from pydantic import Field
from typing import Tuple, Optional, Literal

from .core import GroundScenario, ScenarioRegistry, ScenarioConfig
from .utils import COLLISION_TYPE_GRASPING_OBJECT, pymunk_to_pygame_coord
from .types import Goal, Observation, ObjectObservation

from evodex.simulation.robot import Robot, Action


# TODO: Add the goal in the scenario config
class MoveCubeToTargetScenarioConfig(ScenarioConfig):
    name: Literal["move_cube_to_target"] = "move_cube_to_target"
    target_pos: Optional[Tuple[float, float]] = Field(
        None, description="Target position"
    )
    success_radius: float = Field(..., description="Success radius")
    cube_size: Tuple[float, float] = Field(
        ..., description="Cube size"
    )  # TODO: Extrapolate to more types of objects
    cube_initial_pos: Optional[Tuple[float, float]] = Field(
        None, description="Cube initial position"
    )


@ScenarioRegistry.register
class MoveCubeToTargetScenario(GroundScenario[MoveCubeToTargetScenarioConfig]):
    """
    Scenario where a cube is moved to a target position.
    The cube is initialized at a random position above the ground.
    The target position is also initialized randomly within the screen bounds.
    The scenario is considered successful when the cube is within a certain radius of the target position.
    """

    def __init__(self, config: MoveCubeToTargetScenarioConfig):
        super().__init__(config)
        self.cube_body: Optional[pymunk.Body] = None
        self.cube_shape: Optional[pymunk.Shape] = None

        # TODO: Separate scenario and mini reward functions
        self.prev_dist_cube_to_target: Optional[float] = None
        self.prev_dist_hand_to_cube: Optional[float] = None

    def setup(
        self, space: pymunk.Space, robot: Robot, seed: Optional[int] = None
    ) -> None:
        super().setup(space, robot, seed)

        if self.config.target_pos is None:
            # Random target position within the screen bounds
            self.target_pos = self._random.uniform(
                low=[0, 0],
                high=[self.config.screen.width, self.config.screen.height],
            ).tolist()
        else:
            self.target_pos = self.config.target_pos

        if self.config.cube_initial_pos is not None:
            self.cube_initial_pos = self.config.cube_initial_pos
        else:
            # Random initial position for the cube
            self.cube_initial_pos = np.array(
                [
                    np.random.uniform(
                        low=self.config.cube_size[0] / 2,
                        high=self.config.screen.width - self.config.cube_size[0] / 2,
                    ),
                    self.config.cube_size[1] / 2 + 10,  # Slightly above the ground
                ]
            ).tolist()

        mass = 1.0
        moment = pymunk.moment_for_box(mass, self.config.cube_size)

        self.cube_body = pymunk.Body(mass, moment)
        self.cube_body.position = self.cube_initial_pos
        self.cube_shape = pymunk.Poly.create_box(self.cube_body, self.config.cube_size)
        self.cube_shape.friction = 0.7
        self.cube_shape.elasticity = 0.3
        self.cube_shape.collision_type = COLLISION_TYPE_GRASPING_OBJECT
        space.add(self.cube_body, self.cube_shape)

        self._objects.extend([self.cube_body, self.cube_shape])

        # TODO: Move this to a reward function builder
        self.prev_norm_dist_hand_to_cube = None
        self.prev_norm_dist_cube_to_target = None
        self.max_distance = np.linalg.norm(
            [self.config.screen.width, self.config.screen.height]
        )

    def get_reward(self, robot: Robot, action: Action) -> float:
        """
        Calculates a dense, shaped reward for the grasping task.
        This is the standard method for goal-conditioned environments, especially with HER.
        """

        # TODO: Check if initialised (add is initialised method or property)
        if self.cube_body is None:
            return 0.0

        # --- Weights for different reward components (hyperparameters to tune) ---
        REACH_WEIGHT = 0.5
        GRASP_WEIGHT = 0.25
        LIFT_WEIGHT = 0.5
        MOVE_WEIGHT = 1.0
        STABILITY_PENALTY = 0.05
        ACTION_PENALTY = 0.001
        SUCCESS_BONUS = 250.0

        total_reward = 0.0

        achieved_goal = self.get_achieved_goal(robot)
        desired_goal = self.get_goal(robot)
        robot_extrinsic_obs = robot.get_extrinsic_observation()
        robot_intrinsic_obs = robot.get_intrinsic_observation()

        # --- 1. Reaching Reward: Encourage hand to move towards the cube ---
        fingertip_positions = [
            np.array(f.tip.position, dtype=np.float32)
            for f in robot_extrinsic_obs.fingers
        ]
        cube_pos = np.array(
            [self.cube_body.position.x, self.cube_body.position.y], dtype=np.float32
        )
        dist_hand_to_cube = (
            min([np.linalg.norm(pos - cube_pos) for pos in fingertip_positions])
            / self.max_distance
        )

        if self.prev_dist_hand_to_cube is not None:
            total_reward += REACH_WEIGHT * (
                self.prev_dist_hand_to_cube - dist_hand_to_cube
            )
        self.prev_dist_hand_to_cube = float(dist_hand_to_cube)

        # --- 2. Grasping Reward: Encourage contact with the cube ---
        # Assumes your robot_state includes contact info
        contact_points = sum(
            1
            for finger in robot_intrinsic_obs.fingers
            for segment in finger.segments
            if segment.is_touching
        )
        is_grasping = (
            contact_points > 1
        )  # Grasping is defined as at least 2 contact points
        if is_grasping:
            total_reward += GRASP_WEIGHT * contact_points

        # --- 3. Lifting Reward: Encourage lifting the cube off the ground ---
        # Only provide this reward if the cube is being grasped
        if is_grasping:
            ground_height = self.config.screen.height - 10  # From GroundScenario
            lift_height = cube_pos[1] - (ground_height - self.config.cube_size[1])
            if lift_height > 0:
                total_reward += LIFT_WEIGHT * lift_height

        # --- 4. Moving to Target Reward: Encourage moving the cube to the goal ---
        target_pos = np.array(desired_goal.position)
        dist_cube_to_target = np.linalg.norm(cube_pos - target_pos) / self.max_distance

        if self.prev_dist_cube_to_target is not None:
            total_reward += MOVE_WEIGHT * (
                self.prev_dist_cube_to_target - dist_cube_to_target
            )
        self.prev_dist_cube_to_target = float(dist_cube_to_target)

        # --- 5. Stability Penalty: Encourage placing the cube gently ---
        if dist_cube_to_target < self.config.success_radius * 1.5:
            vel_penalty = np.linalg.norm(achieved_goal.velocity)
            ang_vel_penalty = abs(achieved_goal.angular_velocity)
            total_reward -= STABILITY_PENALTY * (vel_penalty + ang_vel_penalty)

        # --- 6. Action Penalty: Encourage smooth, efficient actions ---
        # action_cost = np.sum(np.square(action.base.velocity)) + abs(action.base.omega)
        # for finger_action in action.fingers:
        #     action_cost += np.sum(np.square(finger_action))
        # total_reward -= ACTION_PENALTY * action_cost

        # --- 7. Success Bonus: Large reward for achieving the final goal ---
        if dist_cube_to_target < self.config.success_radius:
            total_reward += SUCCESS_BONUS

        return total_reward

    def is_terminated(self, robot: Robot) -> bool:
        if self.cube_body:
            cube_pos = np.array([self.cube_body.position.x, self.cube_body.position.y])
            if np.linalg.norm(cube_pos - self.target_pos) < self.config.success_radius:
                return True
        return False

    def get_observation(self, robot: Robot) -> Observation:
        if not self.cube_body:
            raise ValueError("Scenario is not initialized.")

        return Observation(
            object=ObjectObservation(
                position=(self.cube_body.position.x, self.cube_body.position.y),
                velocity=(self.cube_body.velocity.x, self.cube_body.velocity.y),
                angle=self.cube_body.angle,
                angular_velocity=self.cube_body.angular_velocity,
                size=self.config.cube_size,
            ),
            robot=robot.get_extrinsic_observation(self.cube_body),
        )

    def get_goal(self, robot: Robot) -> Goal:
        """Get the target position as the goal."""
        return Goal(
            position=self.target_pos,
            velocity=(0, 0),  # No velocity goal for static target
            angle=0.0,  # No angle goal for static target
            angular_velocity=0.0,  # No angular velocity goal for static target
        )

    def get_achieved_goal(self, robot: Robot) -> Goal:
        """Get the current position of the cube as the achieved goal."""
        if not self.cube_body:
            raise ValueError("Scenario is not initialized.")

        return Goal(
            position=(self.cube_body.position.x, self.cube_body.position.y),
            velocity=(self.cube_body.velocity.x, self.cube_body.velocity.y),
            angle=self.cube_body.angle,
            angular_velocity=self.cube_body.angular_velocity,
        )

    def render(self, screen):
        target_center_pygame = pymunk_to_pygame_coord(
            self.target_pos, self.config.screen.height
        )
        pygame.draw.circle(
            screen,
            pygame.Color("lightgreen"),
            target_center_pygame,
            self.config.success_radius,
            2,
        )
