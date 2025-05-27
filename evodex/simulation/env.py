import gymnasium as gym

from typing import Enum
from evodex.simulation.scenario import Scenario
from evodex.simulation.simulation import Simulation
from evodex.simulation.robot import Robot

class RobotEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a wrapper for the robot simulation environment.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        scenario,
        robot: Robot,
        mode = 'agent',
    ):
        
        super(RobotEnv, self).__init__()
        self.scenario = scenario
        self.robot = robot
        self.mode = mode
        self.sim = self._init_simulation()

        self.action_space = gym.spaces.Discrete(robot.get_dof())
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(robot.get_dof(),), dtype=float)
        
    def _init_simulation(self):
        """
        Initialize the simulation environment.
        """
        sim = Simulation(self.scenario, self.robot, mode=self.mode)
        return sim