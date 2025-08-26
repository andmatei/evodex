from .core import Scenario, ScenarioRegistry, ScenarioConfig
from .types import Observation, Goal

# Import scenarios to be registered
from . import move_cube_to_target
from . import move_to_target
