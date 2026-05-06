import numpy as np
import pytest
from pyhail.common import safe_log

# ── safe_log ─────────────────────────────────────────────────────────────────

def test_safe_log_of_e():
    assert safe_log(np.e) == pytest.approx(1.0)

def test_safe_log_of_one():
    assert safe_log(1.0) == pytest.approx(0.0)

def test_safe_log_zero_returns_floor():
    assert safe_log(0.0) == -10.0

def test_safe_log_negative_returns_floor():
    assert safe_log(-5.0) == -10.0

def test_safe_log_below_eps_returns_floor():
    # 1e-11 < default eps=1e-10
    assert safe_log(1e-11) == -10.0

def test_safe_log_above_eps_returns_log():
    # 1e-9 > default eps=1e-10
    assert safe_log(1e-9) == pytest.approx(np.log(1e-9))

def test_safe_log_custom_eps():
    # With eps=0.5, a value of 0.4 should return the floor
    assert safe_log(0.4, eps=0.5) == -10.0
    # And 0.6 should return log(0.6)
    assert safe_log(0.6, eps=0.5) == pytest.approx(np.log(0.6))

def test_safe_log_array_shape_preserved():
    x = np.array([1.0, np.e, 0.0, -1.0])
    result = safe_log(x)
    assert result.shape == x.shape

def test_safe_log_array_mixed_values():
    x = np.array([np.e, 0.0])
    result = safe_log(x)
    np.testing.assert_allclose(result, [1.0, -10.0])
