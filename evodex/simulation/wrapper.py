import gymnasium as gym

from gymnasium.wrappers import FlattenObservation


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


def flatten(env: gym.Env, observation: bool = False, action: bool = False) -> gym.Env:
    """Wrap the environment with FlattenAction."""
    if action:
        env = FlattenAction(env)
    if observation:
        env = FlattenObservation(env)
    return env
