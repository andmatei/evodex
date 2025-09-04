"""
Isaac Lab environment for the Evodex dexterous hand to grasp a cube.
This file contains the main RL logic for the task.
"""
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import FrameTransformer

from .config import GraspEnvConfig


class GraspEnv(ManagerBasedRLEnv):
    """
    An environment where a dexterous hand learns to grasp a cube and move it to a target.
    """
    cfg: GraspEnvConfig

    def _setup_scene(self):
        """Loads assets and sets up the scene using the configuration object."""
        self.scene = InteractiveScene(self.cfg.scene)
        self.robot = Articulation(self.cfg.robot)
        self.cube = RigidObject(self.cfg.cube)
        self.target = RigidObject(self.cfg.target)
        
        # Add assets to the scene manager for easy access
        self.scene.add_asset("robot", self.robot)
        self.scene.add_asset("cube", self.cube)
        self.scene.add_asset("target", self.target)

        # Add frame transformer to get fingertip positions
        # IMPORTANT: This regex must match the final segment of your URDF links.
        # Example: "f._s." matches "f1_s2", "f2_s3" etc.
        self.fingertip_frames = FrameTransformer(prim_path="{ENV_REGEX_PATH}/Robot/f._s.")
        self.scene.add_asset("fingertips", self.fingertip_frames)

    def _get_observations(self) -> dict:
        """Return observations for the policy."""
        # Get positions of objects in the world frame
        cube_pos_w = self.cube.data.root_pos_w
        target_pos_w = self.target.data.root_pos_w
        fingertip_pos_w, _ = self.fingertip_frames.get_transforms(indices=...)

        # Concatenate all observations for the policy
        obs = torch.cat((
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            fingertip_pos_w.view(self.num_envs, -1),
            cube_pos_w,
            target_pos_w,
        ), dim=-1)
        
        return {"policy": obs}

    def _compute_rewards(self, actions) -> torch.Tensor:
        """Compute the reward for the current step."""
        # Get positions relative to the environment origin for consistent rewards
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        target_pos = self.target.data.root_pos_w - self.scene.env_origins
        fingertip_pos, _ = self.fingertip_frames.get_transforms(indices=...)
        fingertip_pos = fingertip_pos - self.scene.env_origins.unsqueeze(1)

        # -- Stage 1: Reaching ---
        dist_to_cube = torch.norm(fingertip_pos - cube_pos.unsqueeze(1), dim=-1).mean(dim=1)
        reach_reward = 1.0 / (1.0 + 8.0 * dist_to_cube**2)

        # -- Stage 2: Grasping ---
        is_close = (dist_to_cube < 0.08).float()
        grasp_reward = (self.robot.data.joint_pos.mean(dim=1) * -1) * is_close

        # -- Stage 3: Lifting ---
        lift_reward = torch.clamp(cube_pos[:, 2] - 0.05, min=0.0, max=0.5)

        # -- Stage 4: Moving ---
        dist_to_target = torch.norm(cube_pos - target_pos, dim=-1)
        move_reward = (1.0 / (1.0 + 3.0 * dist_to_target**2)) * (lift_reward > 0.1).float()
        
        # -- Success Bonus ---
        is_success = (dist_to_target < 0.05).float()
        success_bonus = is_success * 25.0

        # Total reward is a weighted sum of the stages
        total_reward = (
            reach_reward * 0.5
            + grasp_reward * 0.5
            + lift_reward * 2.0
            + move_reward * 4.0
            + success_bonus
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if the episode should terminate."""
        cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
        target_pos = self.target.data.root_pos_w - self.scene.env_origins
        
        is_dropped = cube_pos[:, 2] < 0.02
        is_success = torch.norm(cube_pos - target_pos, dim=-1) < 0.05
        
        time_out = self.episode_length_buf >= self.max_episode_length
        
        return is_dropped | is_success, time_out

