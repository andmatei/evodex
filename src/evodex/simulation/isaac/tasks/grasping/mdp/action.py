from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg, ActionTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation


class BaseVelocityAction(ActionTerm):
    cfg: BaseVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: BaseVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[self.cfg.asset_name]

        self._actions_raw = torch.zeros((self.num_envs, 6), device=self.device)
        self._actions_processed = torch.zeros((self.num_envs, 6), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._actions_raw.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions_raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._actions_processed

    def process_actions(self, actions: torch.Tensor):
        self._actions_raw = actions.clone()
        self._actions_processed[:, :3] = actions[:, :3] * self.cfg.linear_velocity_scale
        self._actions_processed[:, 3:] = (
            actions[:, 3:] * self.cfg.angular_velocity_scale
        )

    def apply_actions(self):
        self._asset.write_root_velocity_to_sim(self._actions_processed)


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    """Configuration for controlling the base velocity of the gripper."""

    class_type: type = BaseVelocityAction

    linear_velocity_scale: float = 0.1  # Scale for linear velocity commands
    angular_velocity_scale: float = 0.1  # Scale for angular velocity commands
