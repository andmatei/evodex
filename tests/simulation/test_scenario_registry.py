import pytest
from evodex.simulation.scenario.core import (
    Scenario,
    ScenarioConfig,
    ScenarioRegistry,
    ScreenConfig,
)

# --- Test Fixtures and Dummy Classes ---


# A specific config for our dummy scenario
class DummyScenarioConfig(ScenarioConfig):
    extra_param: int = 42


# A concrete scenario class we can use for testing
@ScenarioRegistry.register
class DummyScenario(Scenario[DummyScenarioConfig]):
    """A simple, registerable scenario for testing purposes."""

    def __init__(self, config: DummyScenarioConfig):
        super().__init__(config)

    def get_observation(self):
        """Dummy implementation for observation."""
        return {"dummy_observation": True}

    def get_reward(self):
        """Dummy reset method."""
        return 0.0

    def is_terminated(self, robot, observation, current_step, max_steps):
        return True

    def setup(self, space, robot):
        """Dummy setup method."""
        pass

    def render(self, screen):
        """Dummy render method."""
        pass


# A fixture to ensure the registry is clean before each test
@pytest.fixture(autouse=True)
def clean_registry():
    """Ensures the registry is empty before each test runs."""
    ScenarioRegistry._registry.clear()
    yield  # The test runs here
    ScenarioRegistry._registry.clear()


## ---------------------------------
## Tests for ScenarioRegistry
## ---------------------------------


def test_register_and_get_scenario() -> None:
    """
    Tests that a scenario can be registered and then retrieved correctly.
    """

    # GIVEN the DummyScenario is defined with the @register decorator
    # A specific config for our dummy scenario
    class TestScenarioConfig(ScenarioConfig):
        extra_param: int = 40

    # A concrete scenario class we can use for testing
    @ScenarioRegistry.register
    class TestScenario(Scenario[TestScenarioConfig]):
        """A simple, registerable scenario for testing purposes."""

        pass

    # WHEN we get the scenario by its class name
    retrieved_class = ScenarioRegistry.get("TestScenario")

    # THEN the retrieved class should be the TestScenario class itself
    assert retrieved_class is TestScenario


def test_get_nonexistent_scenario_returns_none() -> None:
    """
    Tests that getting a scenario that doesn't exist returns None.
    """
    # GIVEN an empty registry (thanks to the clean_registry fixture)

    # WHEN we try to get a scenario that was never registered
    retrieved_class = ScenarioRegistry.get("NonExistentScenario")

    # THEN the result should be None
    assert retrieved_class is None


def test_create_scenario_success() -> None:
    """
    Tests the successful creation of a scenario instance from the registry.
    """
    # GIVEN a registered scenario
    ScenarioRegistry.register(DummyScenario)

    # AND a valid configuration object for it
    dummy_config = DummyScenarioConfig(
        name="DummyScenario",
        screen=ScreenConfig(
            width=800,  # Mock screen width
            height=600,  # Mock screen height
        ),
        robot_start_position=(0, 0),  # Mock start position
    )

    # WHEN we create the scenario using the registry
    instance = ScenarioRegistry.create(config=dummy_config)

    # THEN the created object should be an instance of DummyScenario
    assert isinstance(instance, DummyScenario)
    # AND its config should be the one we passed in
    assert instance.config is dummy_config
    assert instance.config.name == "DummyScenario"
    assert instance.config.extra_param == 42


def test_create_scenario_not_found_raises_error() -> None:
    """
    Tests that trying to create an unregistered scenario raises a ValueError.
    """
    # GIVEN an empty registry

    # WHEN we attempt to create a scenario that does not exist
    # THEN a ValueError should be raised
    with pytest.raises(ValueError) as excinfo:
        ScenarioRegistry.create(
            config=ScenarioConfig(
                name="NonExistendScenario",
                screen=ScreenConfig(width=800, height=600),
                robot_start_position=(0, 0),
            ),
        )

    # AND the error message should be informative
    assert "not found in registry" in str(excinfo.value)


def test_registry_instantiation_is_prevented() -> None:
    """
    Tests that calling the constructor of ScenarioRegistry raises a RuntimeError.
    """
    # WHEN we attempt to instantiate the registry itself
    # THEN a RuntimeError should be raised
    with pytest.raises(RuntimeError) as excinfo:
        ScenarioRegistry()

    # AND the error message should be clear
    assert "not meant to be instantiated." in str(excinfo.value)
