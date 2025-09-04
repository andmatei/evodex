"""
Configuration for the Dexterous Hand Grasping Environment using Isaac Lab.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Get the path to this file to construct relative paths for assets
from pathlib import Path
CURRENT_DIR = Path(__file__).parent


class GraspEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the grasp environment."""

    def __post_init__(self):
        """Post-initialization checks and setup."""
        # General settings for the environment
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (1.2, 1.2, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # Scene configuration
        self.scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

        # Asset: Robot Hand
        self.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_PATH}/Robot",
            spawn=sim_utils.UsdFileCfg(
                # Path to your generated URDF file
                usd_path=str(CURRENT_DIR / "assets/dexterous_hand_v1.urdf"),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)), # Start slightly higher
        )

        # Asset: Cube to be grasped
        self.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_PATH}/Cube",
            spawn=sim_utils.UsdFileCfg(
                # Using a standard cube asset from Isaac Lab's Nucleus server
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Props/Blocks/block_instanceable.usd",
                scale=(0.06, 0.06, 0.06), # A 6cm cube
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.03)), # Start in front of the hand
        )

        # Asset: Target location (visual only)
        self.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_PATH}/Target",
            spawn=sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.5)), # Target location
        )

        # Actions: Define how the policy's output maps to the robot
        self.actions.robot = sim_utils.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
        )

