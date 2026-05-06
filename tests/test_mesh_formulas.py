import numpy as np
import pytest
from pyhail.mesh_formulas import (
    mesh_witt1998,
    mesh_mh2019_75,
    mesh_mh2019_95,
    mesh_smooth_blend,
)

# ── mesh_witt1998 ────────────────────────────────────────────────────────────

def test_witt1998_known_value():
    assert mesh_witt1998(100) == pytest.approx(2.54 * 100 ** 0.5)

def test_witt1998_zero():
    assert mesh_witt1998(0) == 0.0

def test_witt1998_negative():
    assert mesh_witt1998(-5) == 0.0

def test_witt1998_preserves_shape():
    shi = np.array([0, 1, 100])
    result = mesh_witt1998(shi)
    assert result.shape == shi.shape
    np.testing.assert_allclose(result, [0.0, 2.54, 2.54 * 10.0])

# ── mesh_mh2019_75 ───────────────────────────────────────────────────────────

def test_mh2019_75_known_value():
    # SHI=1 → 15.096 * 1^0.206 = 15.096
    assert mesh_mh2019_75(1) == pytest.approx(15.096)

def test_mh2019_75_zero():
    assert mesh_mh2019_75(0) == 0.0

def test_mh2019_75_negative():
    assert mesh_mh2019_75(-10) == 0.0

def test_mh2019_75_grows_with_shi():
    assert mesh_mh2019_75(200) > mesh_mh2019_75(100)

# ── mesh_mh2019_95 ───────────────────────────────────────────────────────────

def test_mh2019_95_known_value():
    # SHI=1 → 22.157 * 1^0.212 = 22.157
    assert mesh_mh2019_95(1) == pytest.approx(22.157)

def test_mh2019_95_zero():
    assert mesh_mh2019_95(0) == 0.0

def test_mh2019_95_negative():
    assert mesh_mh2019_95(-10) == 0.0

def test_mh2019_95_exceeds_75th_percentile():
    # 95th percentile should always be larger than 75th for SHI > 0
    for shi in [1, 50, 200, 1000]:
        assert mesh_mh2019_95(shi) > mesh_mh2019_75(shi)

# ── mesh_smooth_blend ────────────────────────────────────────────────────────

def test_smooth_blend_zero():
    assert mesh_smooth_blend(0) == 0.0

def test_smooth_blend_negative():
    assert mesh_smooth_blend(-100) == 0.0

def test_smooth_blend_at_intercept_matches_both_calibrations():
    # At the analytical intercept, both calibrations are equal, so the blend
    # must equal them regardless of the blend weight.
    SHI_INTERCEPT = (15.096 / 2.54) ** (1.0 / (0.5 - 0.206))
    expected = mesh_witt1998(SHI_INTERCEPT)
    assert mesh_smooth_blend(SHI_INTERCEPT) == pytest.approx(expected, rel=1e-6)

def test_smooth_blend_low_shi_converges_to_witt():
    # At very low SHI, w → 0 so output ≈ Witt (1998)
    shi = 1.0
    assert mesh_smooth_blend(shi) == pytest.approx(mesh_witt1998(shi), rel=0.01)

def test_smooth_blend_high_shi_converges_to_mh75():
    # At very high SHI, w → 1 so output ≈ Murillo & Homeyer 75th percentile
    shi = 10_000.0
    assert mesh_smooth_blend(shi) == pytest.approx(mesh_mh2019_75(shi), rel=0.01)

def test_smooth_blend_preserves_shape():
    shi = np.array([-1.0, 0.0, 100.0, 500.0])
    result = mesh_smooth_blend(shi)
    assert result.shape == shi.shape
    assert result[0] == 0.0
    assert result[1] == 0.0

def test_smooth_blend_narrower_transition_is_sharper():
    # A narrower transition_width makes w change more steeply around the
    # intercept.  A SHI well below the intercept should be closer to Witt.
    shi = 50.0
    wide = mesh_smooth_blend(shi, transition_width=2000)
    narrow = mesh_smooth_blend(shi, transition_width=10)
    witt = mesh_witt1998(shi)
    assert abs(narrow - witt) < abs(wide - witt)
