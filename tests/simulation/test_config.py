import pytest
from pydantic import ValidationError

from evodex.simulation.robot.config import (
    RobotConfig,
    BaseConfig,
    FingerConfig,
    SegmentConfig,
    GlobalSegmentConfig,
)

# --- Test Data Fixtures ---


@pytest.fixture
def valid_base_data():
    """Provides data for a valid BaseConfig."""
    return {"width": 0.1, "height": 0.05, "mass": 0.5}


@pytest.fixture
def valid_global_defaults():
    """Provides a valid set of default global properties."""
    return {
        "mass": 0.02,
        "motor_stiffness": 30.0,
        "motor_damping": 1.0,
        "joint_angle_limit": (0.0, 1.57),
    }


# --- Success Cases ---


def test_successful_full_config(valid_base_data, valid_global_defaults):
    """
    Tests that a complete, valid robot configuration can be parsed without errors.
    """
    robot_data = {
        "base": valid_base_data,
        "fingers": [
            {
                "name": "Thumb",
                "defaults": valid_global_defaults,
                "segments": [
                    {"length": 0.05, "width": 0.015},  # Inherits all defaults
                    {"length": 0.06, "width": 0.015, "mass": 0.025},  # Overrides mass
                ],
            }
        ],
    }

    # This should not raise any exceptions
    config = RobotConfig.model_validate(robot_data)

    # Check if defaults were applied correctly
    assert config.fingers[0].segments[0].mass == 0.02
    assert config.fingers[0].segments[0].motor_stiffness == 30.0

    # Check if override was successful
    assert config.fingers[0].segments[1].mass == 0.025
    assert config.fingers[0].segments[1].motor_stiffness == 30.0  # Inherited


def test_all_properties_defined_in_segment(valid_global_defaults):
    """
    Tests a configuration where segments define all properties and no defaults are needed.
    """
    finger_data = {
        "segments": [
            {
                "length": 0.05,
                "width": 0.012,
                **valid_global_defaults,  # Unpack all global properties here
            }
        ]
        # No 'defaults' key needed
    }

    config = FingerConfig.model_validate(finger_data)
    assert config.segments[0].mass == 0.02


# --- Failure Cases ---


def test_missing_required_global_property_raises_error(valid_base_data):
    """
    Tests that a ValueError is raised if a global property is not defined
    in the segment or in the finger's defaults.
    """
    robot_data = {
        "base": valid_base_data,
        "fingers": [
            {
                "name": "Thumb",
                "defaults": {
                    # Missing 'mass' and 'joint_angle_limit'
                    "motor_stiffness": 30.0,
                    "motor_damping": 1.0,
                },
                "segments": [{"length": 0.05, "width": 0.015}],
            }
        ],
    }

    with pytest.raises(ValidationError) as e:
        RobotConfig.model_validate(robot_data)

    # Check for a descriptive error message from our validator
    assert "joint_angle_limit" in str(e.value)
    assert "mass" in str(e.value)


def test_invalid_joint_angle_limit_raises_error():
    """Tests that joint limits with min >= max raise an error."""
    with pytest.raises(
        ValidationError,
        match="joint_angle_limit.min must be less than joint_angle_limit.max",
    ):
        GlobalSegmentConfig(joint_angle_limit=(1.0, 0.5))


def test_out_of_bounds_joint_angle_limit_raises_error():
    """Tests that joint limits outside [-pi, pi] raise an error."""
    with pytest.raises(
        ValidationError, match="joint_angle_limit must be between -pi and pi"
    ):
        GlobalSegmentConfig(joint_angle_limit=(-4.0, 1.0))

    with pytest.raises(
        ValidationError, match="joint_angle_limit must be between -pi and pi"
    ):
        GlobalSegmentConfig(joint_angle_limit=(0.0, 4.0))


def test_negative_segment_length_raises_error():
    """Tests that negative length or width in SegmentConfig fails."""
    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 0"
    ):
        SegmentConfig(
            length=-0.1,
            width=0.05,
            mass=1,
            motor_stiffness=1,
            motor_damping=1,
            joint_angle_limit=(0, 1),
        )


def test_non_positive_base_mass_raises_error():
    """Tests that non-positive mass in BaseConfig fails."""
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        BaseConfig(width=0.1, height=0.05, mass=0.0)
