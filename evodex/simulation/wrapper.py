import gymnasium as gym

from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space, unflatten, flatten


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = flatten_space(self.env.action_space)

    def action(self, action):
        return unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return flatten(self.env.action_space, action)


def flatten_env(
    env: gym.Env, observation: bool = False, action: bool = False
) -> gym.Env:
    """Wrap the environment with FlattenAction."""
    if action:
        env = FlattenAction(env)
    if observation:
        env = FlattenObservation(env)
    return env
