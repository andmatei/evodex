import pytest

# Import the functions and models to be tested
from evodex.evolution.tree import config_to_tree, tree_to_config
from evodex.simulation.isaac.robot.evolvable import (
    EvolvableRobotConfig,
    EvolvableBaseConfig,
    EvolvableFingerConfig,
    EvolvableLinkConfig,
)

from tests.utils import load_config


@pytest.fixture
def sample_robot_config() -> EvolvableRobotConfig:
    """A pytest fixture to provide a consistent, complex robot config for testing."""
    ROBOT_CONFIG_PATH = "tests/configs/test_robot.yaml"
    config = load_config(ROBOT_CONFIG_PATH)
    return EvolvableRobotConfig(**config)


def test_config_to_tree_structure(sample_robot_config):
    """
    Tests that the generated tree has the correct parent-child structure
    for both parallel and chain gene lists.
    """
    root_node = config_to_tree(sample_robot_config)

    # 1. Check root and base
    assert isinstance(root_node.data, EvolvableRobotConfig)
    assert len(root_node.children) == 3  # 1 base + 2 fingers
    base_node = next(
        c for c in root_node.children if isinstance(c.data, EvolvableBaseConfig)
    )
    assert base_node is not None
    assert base_node.parent is root_node

    # 2. Check parallel fingers
    finger_nodes = [
        c for c in root_node.children if isinstance(c.data, EvolvableFingerConfig)
    ]
    assert len(finger_nodes) == 2
    assert finger_nodes[0].parent is root_node
    assert finger_nodes[1].parent is root_node

    # 3. Check chained segments for the first finger (2 segments)
    finger_0 = finger_nodes[0]
    assert len(finger_0.children) == 1
    segment_0_0 = finger_0.children[0]
    assert isinstance(segment_0_0.data, EvolvableLinkConfig)
    assert segment_0_0.data.length == 100.0

    assert len(segment_0_0.children) == 1
    segment_0_1 = segment_0_0.children[0]
    assert isinstance(segment_0_1.data, EvolvableLinkConfig)
    assert segment_0_1.data.length == 80.0
    assert segment_0_1.parent is segment_0_0
    assert len(segment_0_1.children) == 0  # End of the chain

    # 4. Check chained segments for the second finger (3 segments)
    finger_1 = finger_nodes[1]
    assert len(finger_1.children) == 1
    segment_1_0 = finger_1.children[0]
    assert isinstance(segment_1_0.data, EvolvableLinkConfig)
    assert segment_1_0.data.length == 100.0

    assert len(segment_1_0.children) == 1
    segment_1_1 = segment_1_0.children[0]
    assert isinstance(segment_1_1.data, EvolvableLinkConfig)
    assert segment_1_1.data.length == 80.0
    assert segment_1_1.parent is segment_1_0

    assert len(segment_1_1.children) == 1
    segment_1_2 = segment_1_1.children[0]
    assert isinstance(segment_1_2.data, EvolvableLinkConfig)
    assert segment_1_2.data.length == 60.0
    assert segment_1_2.parent is segment_1_1
    assert len(segment_1_2.children) == 0  # End of the chain


def test_round_trip_conversion(sample_robot_config):
    """
    Tests the most important property: that converting a config to a tree and
    back again results in an identical copy of the original.
    """
    # 1. Convert the original config to a tree
    tree = config_to_tree(sample_robot_config)

    # 2. Convert the tree back into a new Pydantic model instance
    reconstructed_config = tree_to_config(tree)

    # 3. Assert that the reconstructed config is identical to the original.
    # Pydantic's built-in equality check (==) compares all nested values.
    assert reconstructed_config == sample_robot_config
