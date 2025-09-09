"""
Configuration for the Dexterous Hand Grasping Environment using Isaac Lab.
"""

from __future__ import annotations
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import evodex.simulation.isaac.tasks.grasping.mdp as mdp

from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
    DeformableObjectCfg,
    AssetBaseCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from evodex.core.paths import GENERATED_DIR

from .mdp.action import BaseVelocityActionCfg

CURRENT_DIR = Path(__file__).parent


@configclass
class GripperCfg(ArticulationCfg):
    """Configuration for the dexterous hand robot."""

    spawn = sim_utils.UsdFileCfg(
        usd_path=str(GENERATED_DIR / "gripper.usd"),
        activate_contact_sensors=False,  # TODO: Enable if needed
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
    actuators = {
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"], damping=0.1, friction=0.01, stiffness=3.0
        ),
    }


@configclass
class GraspingSceneCfg(InteractiveSceneCfg):
    robot = GripperCfg(prim_path="{ENV_REGEX_NS}/Robot")

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    object_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/base"),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*fingertip.*"
            ),
        ],
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
class ActionCfg:
    base_action: BaseVelocityActionCfg = BaseVelocityActionCfg(
        asset_name="robot",
    )
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0
    )


# @configclass
# class CommandsConfig:
#     object_pose = mdp.UniformPoseCommandCfg(
#         asset_name="cube",
#         resampling_time_range=(10, 20),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.1, 0.1),
#             pos_y=(-0.1, 0.1),
#             pos_z=(0.025, 0.025),
#             roll=(0.0, 0.0),
#             pitch=(0.0, 0.0),
#             yaw=(0.0, 0.0),
#         ),
#     )

#     gripper_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name="base",
#         resampling_time_range=(10, 20),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.2, 0.2),
#             pos_y=(-0.2, 0.2),
#             pos_z=(0.4, 0.6),
#             roll=(-3.14, 3.14),
#             pitch=(-3.14, 3.14),
#             yaw=(-3.14, 3.14),
#         ),
#     )

#     # TODO: Check if this is correct
#     target_pose = mdp.UniformPoseCommandCfg(
#         asset_name="target",
#         body_name="base",
#         resampling_time_range=(10, 20),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             pos_x=(-0.2, 0.2),
#             pos_y=(-0.2, 0.2),
#             pos_z=(0.4, 0.6),
#             roll=(-3.14, 3.14),
#             pitch=(-3.14, 3.14),
#             yaw=(-3.14, 3.14),
#         ),
#     )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        @configclass
        class GripperStateCfg:
            base_linear_vel = ObservationTermCfg(func=mdp.root_lin_vel_w)
            base_angular_vel = ObservationTermCfg(func=mdp.root_ang_vel_w)

            joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
            joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

            base_pos_rel = ObservationTermCfg(
                func=mdp.pos_rel,
                params={
                    "frame_cfg": SceneEntityCfg("object"),
                    "entity_cfg": SceneEntityCfg("robot", body_name="base"),
                },
            )
            base_quat_rel = ObservationTermCfg(
                func=mdp.quat_rel,
                params={
                    "frame_cfg": SceneEntityCfg("object"),
                    "entity_cfg": SceneEntityCfg("robot", body_name="base"),
                },
            )

            # fingertips_pos_rel = ObservationTermCfg(

            # )

        @configclass
        class ObjectStateCfg:
            pass

        @configclass
        class TargetStateCfg:
            pass

        fingertip_pos = ObservationTermCfg(
            func=mdp.body_pose_w, body_names=[".*fingertip.*"]
        )  # TODO: Calcualte the relative distance to the object
        object_pos = ObservationTermCfg(func=mdp.body_pose_w, body_names=["cube"])
        target_pos = ObservationTermCfg(func=mdp.body_pose_w, body_names=["target"])

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

#     # TODO: Check if this is correct and randomise the object, target and robot poses
#     reset_object_position = EventTermCfg(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
#             "velocity_range": {},
#             "asset_cfg": SceneEntityCfg("object", body_names="Object"),
#         },
#     )


# # TODO: Implement reward shaping
# @configclass
# class RewardCfg:
#     """Configuration for the reward function."""

#     pass


# @configclass
# class TerminationCfg:
#     time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

#     object_dropping = TerminationTermCfg(
#         func=mdp.root_height_below_minimum,
#         params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
#     )


# # TODO: Implement curriculum learning
# @configclass
# class CurriculumCfg:
#     """Configuration for curriculum learning."""

#     pass


# @configclass
# class GraspingEnvCfg(ManagerBasedRLEnvCfg):
#     scene: GraspingSceneCfg = GraspingSceneCfg(num_envs=2048, env_spacing=2.0)

#     observations: ObservationCfg = ObservationCfg()
#     actions: ActionCfg = ActionCfg()
#     commands: CommandsConfig = CommandsConfig()

#     rewards: RewardCfg = RewardCfg()
#     terminations: TerminationCfg = TerminationCfg()
#     events: EventCfg = EventCfg()
#     curriculum: CurriculumCfg = CurriculumCfg()

#     def __post_init__(self):
#         self.decimation = 2
#         self.episode_length_s = 20.0

#         # self.sim.dt = 1.0 / 120
#         # self.sim.render_interval = self.decimation
