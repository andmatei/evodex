from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms


def pose_rel(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform the robot position from world frame to object frame.
    """
    entity: RigidObject = env.scene[entity_cfg.name]
    frame: RigidObject = env.scene[frame_cfg.name]

    # Compute the relative position
    entity_pos_w = entity.data.root_pos_w
    entity_quat_w = entity.data.root_quat_w
    frame_pos_w = frame.data.root_pos_w
    frame_quat_w = frame.data.root_quat_w

    entity_pos_rel, entity_quat_rel = subtract_frame_transforms(
        frame_pos_w, frame_quat_w, entity_pos_w, entity_quat_w
    )

    return entity_pos_rel, entity_quat_rel


def pos_rel(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Transform the robot position from world frame to object frame.
    """
    return pose_rel(env, entity_cfg, frame_cfg)[0]


def quat_rel(
    env: ManagerBasedRLEnv,
    entity_cfg: SceneEntityCfg = SceneEntityCfg("gripper"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Transform the robot orientation from world frame to object frame.
    """
    return pose_rel(env, entity_cfg, frame_cfg)[1]
