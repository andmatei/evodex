import pytest
from pydantic import ValidationError
from typing import Literal

# Assuming the ScenarioRegistry is in this path
from evodex.simulation.scenario.core import (
    Scenario,
    ScenarioConfig,
    ScenarioRegistry,
)

# --- Test Fixtures and Dummy Classes ---


# A specific config for our dummy scenario.
# The `name` field with a Literal value is crucial for registration.
class DummyConfig(ScenarioConfig):
    name: Literal["dummy_scenario"] = "dummy_scenario"
    extra_param: int = 42


# A concrete scenario class we can use for testing.
# The @register decorator runs when this module is imported.
@ScenarioRegistry.register
class DummyScenario(Scenario[DummyConfig]):
    """A simple, registerable scenario for testing purposes."""

    def __init__(self, config: DummyConfig):
        super().__init__(config)

    # No need to implement abstract methods for these tests
    def setup(self, space, robot):
        pass

    def get_reward(self, robot, action):
        return 0.0

    def is_terminated(self, robot):
        return True

    def get_observation(self, robot):
        return {}

    def render(self, screen):
        pass

    def get_achieved_goal(self, robot):
        pass

    def get_goal(self, robot):
        pass


# A fixture to ensure the registry is clean before each test
@pytest.fixture(autouse=True)
def clean_registry():
    """Ensures the registry is empty before each test runs."""
    ScenarioRegistry._scenarios.clear()
    ScenarioRegistry._configs.clear()
    # Re-register the dummy scenario for each test after clearing
    ScenarioRegistry.register(DummyScenario)
    yield
    # Cleanup after test
    ScenarioRegistry._scenarios.clear()
    ScenarioRegistry._configs.clear()


## ---------------------------------
## Tests for ScenarioRegistry
## ---------------------------------


def test_register_scenario_succeeds() -> None:
    """
    Tests that the @register decorator correctly populates both internal
    dictionaries with the correct name and types.
    """
    # GIVEN the DummyScenario is defined with the @register decorator

    # THEN the internal dictionaries should be populated correctly
    assert "dummy_scenario" in ScenarioRegistry._scenarios
    assert "dummy_scenario" in ScenarioRegistry._configs

    # AND the stored types should be correct
    assert ScenarioRegistry._scenarios["dummy_scenario"] is DummyScenario
    assert ScenarioRegistry._configs["dummy_scenario"] is DummyConfig


def test_load_scenario_from_dict_succeeds() -> None:
    """
    Tests that a scenario can be successfully loaded from a valid dictionary.
    This is the primary success-path test.
    """
    # GIVEN a valid configuration dictionary
    config_data = {
        "name": "dummy_scenario",
        "screen": {"width": 800, "height": 600},
        "robot_start_position": (10, 20),
        "extra_param": 99,  # Override the default
    }

    # WHEN we load the scenario using the registry
    instance = ScenarioRegistry.load(config_data)

    # THEN the created object should be an instance of DummyScenario
    assert isinstance(instance, DummyScenario)
    # AND its config should be a correctly parsed DummyConfig instance
    assert isinstance(instance.config, DummyConfig)
    # AND the config values should match the input data
    assert instance.config.name == "dummy_scenario"
    assert instance.config.extra_param == 99
    assert instance.config.screen.width == 800


def test_load_fails_for_unregistered_scenario() -> None:
    """
    Tests that trying to load a scenario with an unknown name raises a ValueError.
    """
    # GIVEN a config dict with a name that is not registered
    config_data = {"name": "unregistered_scenario"}

    # WHEN we attempt to load it
    # THEN a ValueError should be raised with a clear message
    with pytest.raises(
        ValueError,
        match="Scenario config class 'unregistered_scenario' not registered.",
    ):
        ScenarioRegistry.load(config_data)


def test_load_fails_for_missing_name_field() -> None:
    """
    Tests that loading a dictionary without a 'name' field raises a ValueError.
    """
    # GIVEN a config dict that is missing the 'name' key
    config_data = {
        "screen": {"width": 800, "height": 600},
        "robot_start_position": (10, 20),
    }

    # WHEN we attempt to load it
    # THEN a ValueError should be raised with a clear message
    with pytest.raises(
        ValueError, match="Scenario configuration must contain a 'name' field."
    ):
        ScenarioRegistry.load(config_data)


def test_load_fails_for_invalid_config_data() -> None:
    """
    Tests that the registry correctly raises a Pydantic ValidationError if the
    data does not match the specific config model's schema.
    """
    # GIVEN a config dict with the correct name but invalid data type for a field
    config_data = {
        "name": "dummy_scenario",
        "screen": {"width": "invalid-width", "height": 600},  # width should be an int
        "robot_start_position": (10, 20),
    }

    # WHEN we attempt to load it
    # THEN Pydantic's ValidationError should be raised
    with pytest.raises(ValidationError):
        ScenarioRegistry.load(config_data)
