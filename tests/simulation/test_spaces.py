import pytest

from evodex.simulation.robot.spaces import Action, BaseAction, ActionScale
from evodex.simulation.robot.utils import Scale

class TestActionScaling:
    @pytest.fixture
    def action_scale_config(self) -> ActionScale:
        """Provides a valid ActionScale configuration for tests."""
        return ActionScale(
            velocity=(
                Scale(target=(-0.5, 0.5)), # vx
                Scale(target=(-0.2, 0.2))  # vy
            ),
            omega=Scale(target=(-1.57, 1.57)),
            motor_rate=Scale(domain=(0, 1), target=(0, 500))
        )

    @pytest.fixture
    def sample_action(self) -> Action:
        """Provides a sample normalized action."""
        return Action(
            base=BaseAction(velocity=(-1.0, 0.5), omega=1.0),
            fingers=[[0.0, 0.5, 1.0], [0.2]]
        )

    def test_action_scaling(self, action_scale_config, sample_action):
        """
        Tests the end-to-end scaling of a full action.
        """
        scaled_action = sample_action.scale(action_scale_config)

        # Check base velocity scaling
        assert scaled_action.base.velocity[0] == pytest.approx(-0.5) # -1 in [-1,1] -> -0.5 in [-0.5,0.5]
        assert scaled_action.base.velocity[1] == pytest.approx(0.1)  # 0.5 in [-1,1] -> 0.1 in [-0.2,0.2]
        
        # Check base omega scaling
        assert scaled_action.base.omega == pytest.approx(1.57) # 1 in [-1,1] -> 1.57 in [-1.57,1.57]

        # Check finger motor rate scaling
        assert scaled_action.fingers[0][0] == pytest.approx(0)    # 0 in [0,1] -> 0 in [0,500]
        assert scaled_action.fingers[0][1] == pytest.approx(250)  # 0.5 in [0,1] -> 250 in [0,500]
        assert scaled_action.fingers[0][2] == pytest.approx(500)  # 1 in [0,1] -> 500 in [0,500]
        assert scaled_action.fingers[1][0] == pytest.approx(100)  # 0.2 in [0,1] -> 100 in [0,500]
        
    def test_chained_scaling_is_reversible(self, action_scale_config, sample_action):
        """
        Tests that scaling an action and then inverse scaling it returns the original.
        """
        # Forward scale
        scaled_action = sample_action.scale(action_scale_config)
        
        # Manually inverse scale the values
        vx_restored = action_scale_config.velocity[0].scale(scaled_action.base.velocity[0], inverse=True)
        vy_restored = action_scale_config.velocity[1].scale(scaled_action.base.velocity[1], inverse=True)
        omega_restored = action_scale_config.omega.scale(scaled_action.base.omega, inverse=True)
        
        assert vx_restored == pytest.approx(sample_action.base.velocity[0])
        assert vy_restored == pytest.approx(sample_action.base.velocity[1])
        assert omega_restored == pytest.approx(sample_action.base.omega)
