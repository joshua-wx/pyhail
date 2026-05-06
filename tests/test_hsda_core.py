import numpy as np
import pytest
from pyhail.hsda import trapmf, calc_ag, h_sz

# ── trapmf ───────────────────────────────────────────────────────────────────

def test_trapmf_below_a():
    assert trapmf(0.0, 1.0, 2.0, 3.0, 4.0) == 0.0

def test_trapmf_at_a():
    # x == a → not (x > a), falls to else → 0
    assert trapmf(1.0, 1.0, 2.0, 3.0, 4.0) == 0.0

def test_trapmf_rising_slope():
    assert trapmf(1.5, 1.0, 2.0, 3.0, 4.0) == pytest.approx(0.5)

def test_trapmf_at_b():
    assert trapmf(2.0, 1.0, 2.0, 3.0, 4.0) == 1.0

def test_trapmf_flat_top():
    assert trapmf(2.5, 1.0, 2.0, 3.0, 4.0) == 1.0

def test_trapmf_at_c():
    assert trapmf(3.0, 1.0, 2.0, 3.0, 4.0) == 1.0

def test_trapmf_falling_slope():
    assert trapmf(3.5, 1.0, 2.0, 3.0, 4.0) == pytest.approx(0.5)

def test_trapmf_at_d():
    # x == d → not (x < d), falls to else → 0
    assert trapmf(4.0, 1.0, 2.0, 3.0, 4.0) == 0.0

def test_trapmf_above_d():
    assert trapmf(5.0, 1.0, 2.0, 3.0, 4.0) == 0.0

def test_trapmf_degenerate_triangle():
    # b == c collapses the flat top to a point
    assert trapmf(2.0, 1.0, 2.0, 2.0, 3.0) == 1.0
    assert trapmf(1.5, 1.0, 2.0, 2.0, 3.0) == pytest.approx(0.5)

# ── calc_ag ──────────────────────────────────────────────────────────────────

# Use the alt=0 small-hail membership function as a representative fixture.
_MF_H1 = np.array([
    [45.0, 50.0, 60.0, 65.0],
    [-0.5, -0.3,  0.3,  0.5],
    [0.92, 0.96, 0.99, 1.00],
])
_W = np.array([1.0, 0.3, 0.6])
_Q = np.array([1.0, 1.0, 1.0])

def test_calc_ag_all_in_flat_top_returns_one():
    # zh=55 (in [50,60]), zdr=0 (in [-0.3,0.3]), rhv=0.97 (in [0.96,0.99])
    # → all trapmf values = 1.0, no rule-1 trigger
    result = calc_ag(_MF_H1, 55.0, 0.0, 0.97, _Q, _W)
    assert result == pytest.approx(1.0)

def test_calc_ag_rule1_min_below_threshold():
    # zh=42 is below a=45 → zh_mf=0 → min < 0.2 → rule 1 → ag=0
    result = calc_ag(_MF_H1, 42.0, 0.0, 0.97, _Q, _W)
    assert result == 0.0

def test_calc_ag_weighted_correctly():
    # zh=47.5, rising slope: zh_mf = (47.5-45)/(50-45) = 0.5
    # zdr=0 → 1.0; rhv=0.97 → 1.0; no rule 1 (0.5 >= 0.2)
    # ag = (1*1*0.5 + 0.3*1*1 + 0.6*1*1) / (1+0.3+0.6) = 1.4/1.9
    result = calc_ag(_MF_H1, 47.5, 0.0, 0.97, _Q, _W)
    assert result == pytest.approx(1.4 / 1.9)

# ── h_sz ─────────────────────────────────────────────────────────────────────

def _alt0_arrays():
    """Return numpy mf arrays and weight for alt=0 (from build_mf)."""
    mf_h1 = np.array([
        [45.0, 50.0,  60.0,  65.0],
        [-0.5, -0.3,   0.3,   0.5],
        [0.92, 0.96,  0.99,  1.00],
    ])
    mf_h2 = np.array([
        [48.0, 58.0,  63.0,  68.0],
        [-0.5, -0.3,   0.3,   0.5],
        [0.92, 0.96,  0.99,  1.00],
    ])
    mf_h3 = np.array([
        [ 50.0,  60.0, 100.0, 101.0],
        [ -8.75,  -7.75,   0.3,   0.5],
        [ -1.00,   0.00,  0.99,  1.00],
    ])
    w = np.array([1.0, 0.3, 0.6])
    q = np.array([1.0, 1.0, 1.0])
    return mf_h1, mf_h2, mf_h3, w, q

def test_h_sz_giant_hail_high_confidence():
    # zh=60 is in the flat top of h1, h2, and h3; zdr=0 and rhv=0.97 give
    # mf=1 for all variables.  All three ag=1.0; last maximum is h3 → class 3.
    mf_h1, mf_h2, mf_h3, w, q = _alt0_arrays()
    result = h_sz(60.0, 0.0, 0.97, mf_h1, mf_h2, mf_h3, q, w)
    assert result == 3

def test_h_sz_rule2_low_aggregate_forces_small_hail():
    # zh=46 gives zh_mf≈0.2 for h1 and 0 for h2/h3 → max_ag < 0.6 → class 1
    mf_h1, mf_h2, mf_h3, w, q = _alt0_arrays()
    result = h_sz(46.0, 0.0, 0.97, mf_h1, mf_h2, mf_h3, q, w)
    assert result == 1

def test_h_sz_rule3_high_zdr_forces_small_hail():
    # Construct wide zdr membership functions so zdr=2 stays in the flat top
    # and all three classes get high ag.  Without rule 3, h2 wins (ag tie
    # broken by last index).  Rule 3 downgrades to class 1 when zdr >= 2.
    mf_h1 = np.array([[45.0, 50.0, 60.0, 65.0], [-1.0, 0.0, 3.0, 4.0], [0.92, 0.96, 0.99, 1.00]])
    mf_h2 = np.array([[50.0, 55.0, 65.0, 70.0], [-1.0, 0.0, 3.0, 4.0], [0.90, 0.95, 0.99, 1.00]])
    mf_h3 = np.array([[ 55.0, 65.0, 100.0, 101.0], [-1.0, 0.0, 3.0, 4.0], [0.90, 0.95, 0.99, 1.00]])
    w = np.array([1.0, 0.3, 0.6])
    q = np.array([1.0, 1.0, 1.0])
    # zh=60 in flat-top of h1 and h2; zdr=2 in flat-top of all → high ag for h1 & h2
    result = h_sz(60.0, 2.0, 0.97, mf_h1, mf_h2, mf_h3, q, w)
    assert result == 1
