import pytest

from pydantic import ValidationError
from evodex.simulation.robot.utils import scale, Scale

class TestScale:
    def test_initialization(self):
        """Tests successful creation and default domain."""
        s = Scale(target=(0, 10))
        assert s.domain == (-1.0, 1.0)
        assert s.target == (0, 10)

    def test_forward_scaling(self):
        """Tests standard forward scaling (domain -> target)."""
        s = Scale(domain=(-1, 1), target=(0, 100))
        assert s.scale(-1) == pytest.approx(0)
        assert s.scale(0) == pytest.approx(50)
        assert s.scale(1) == pytest.approx(100)
        
    def test_inverse_scaling(self):
        """Tests inverse scaling (target -> domain)."""
        s = Scale(domain=(-1, 1), target=(0, 100))
        assert s.scale(0, inverse=True) == pytest.approx(-1)
        assert s.scale(50, inverse=True) == pytest.approx(0)
        assert s.scale(100, inverse=True) == pytest.approx(1)

    def test_asymmetric_target(self):
        """Tests scaling to a negative/asymmetric target range."""
        s = Scale(domain=(0, 1), target=(-50, 50))
        assert s.scale(0) == pytest.approx(-50)
        assert s.scale(0.5) == pytest.approx(0)
        assert s.scale(1) == pytest.approx(50)
        
    def test_validation_error(self):
        """Tests that Pydantic raises an error for missing target."""
        with pytest.raises(ValidationError):
            Scale() # Missing 'target'
