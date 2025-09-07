"""
Configuration for the Dexterous Hand Grasping Environment using Isaac Lab.
"""

from __future__ import annotations
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

from isaaclab.managers import (
    ActionTermCfg,
    ActionTerm,
    ObservationGroupCfg,
    ObservationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import (
    ArticulationCfg,
    Articulation,
    RigidObjectCfg,
    DeformableObjectCfg,
    AssetBaseCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get the path to this file to construct relative paths for assets
from pathlib import Path

CURRENT_DIR = Path(__file__).parent


@configclass
class GraspingSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class GripperCfg(ArticulationCfg):
    """Configuration for the dexterous hand robot."""

    spawn = sim_utils.UsdFileCfg(
        # Path to your generated URDF file
        usd_path=str(CURRENT_DIR / "assets/dexterous_hand_v1.urdf"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),  # Start above the ground
        rot=(1.0, 0.0, 0.0, 0.0),  # No rotation
    )


@configclass
class BaseVelocityAction(ActionTerm):
    cfg: BaseVelocityActionCfg

    def __init__(self, cfg: BaseVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[self.cfg.asset_name]
        self._action = torch.zeros((self.num_envs, 6), device=self.device)

    @property
    def action_size(self) -> int:
        return 6

    def process_actions(self, actions: torch.Tensor):
        self._action[:, :3] = actions[:, :3] * self.cfg.linear_velocity_scale
        self._action[:, 3:] = actions[:, 3:] * self.cfg.angular_velocity_scale

    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._action)


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    """Configuration for controlling the base velocity of the gripper."""

    class_type: type = BaseVelocityAction

    linear_velocity_scale: float = 0.1  # Scale for linear velocity commands
    angular_velocity_scale: float = 0.1  # Scale for angular velocity commands


@configclass
class ActionCfg:
    base_action: BaseVelocityActionCfg = BaseVelocityActionCfg(
        asset_name="gripper",
    )
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="gripper", joint_names=[".*"], scale=1.0
    )


@configclass
class CommandsConfig:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="cube",
        resampling_time_range=(10, 20),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.1, 0.1),
            pos_z=(0.025, 0.025),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    gripper_pose = mdp.UniformPoseCommandCfg(
        asset_name="gripper",
        body_name="base",
        resampling_time_range=(10, 20),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),
            pos_y=(-0.2, 0.2),
            pos_z=(0.4, 0.6),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )

    # TODO: Check if this is correct
    target_pose = mdp.UniformPoseCommandCfg(
        asset_name="target",
        body_name="base",
        resampling_time_range=(10, 20),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),
            pos_y=(-0.2, 0.2),
            pos_z=(0.4, 0.6),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        fingertip_pos = ObservationTermCfg(
            func=mdp.body_pose_w, body_names=[".*fingertip.*"]
        )  # TODO: Calcualte the relative distance to the object
        object_pos = ObservationTermCfg(func=mdp.body_pose_w, body_names=["cube"])
        target_pos = ObservationTermCfg(func=mdp.body_pose_w, body_names=["target"])

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    # TODO: Check if this is correct and randomise the object, target and robot poses
    reset_object_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


# TODO: Implement reward shaping
@configclass
class RewardCfg:
    """Configuration for the reward function."""

    pass


@configclass
class TerminationCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


# TODO: Implement curriculum learning
@configclass
class CurriculumCfg:
    """Configuration for curriculum learning."""

    pass


@configclass
class GraspingEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspingSceneCfg = GraspingSceneCfg(num_envs=2048, env_spacing=2.0)

    observations: ObservationCfg = ObservationCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandsConfig = CommandsConfig()

    rewards: RewardCfg = RewardCfg()
    terminations: TerminationCfg = TerminationCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0

        # self.sim.dt = 1.0 / 120
        # self.sim.render_interval = self.decimation
