import warnings
import numpy as np
import pytest
from pyhail.mesh_ppi import main

N_RAYS = 10
N_BINS = 30
RANGE_BIN_SIZE = 2000       # m; bins span 2–60 km
LEVELS = [3000, 6000]       # m ASL: freezing level and -20 °C level
RADAR_ALTITUDE = 100.0      # m ASL

# Eight elevations so we meet the minimum-sweeps-warning threshold (≥ 8).
_ELEVATIONS_8 = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0]


def _make_inputs(n_sweeps=8, elevations=None, zh_val=55.0):
    if elevations is None:
        elevations = _ELEVATIONS_8[:n_sweeps]
    ranges = [np.arange(1, N_BINS + 1) * RANGE_BIN_SIZE for _ in range(n_sweeps)]
    azimuths = [np.linspace(0, 350, N_RAYS) for _ in range(n_sweeps)]
    refl = [np.full((N_RAYS, N_BINS), zh_val) for _ in range(n_sweeps)]
    return refl, elevations, azimuths, ranges


# ── output structure ─────────────────────────────────────────────────────────

def test_returns_four_dicts():
    result = main(*_make_inputs(), RADAR_ALTITUDE, LEVELS)
    assert len(result) == 4
    for d in result:
        assert isinstance(d, dict)

def test_all_dicts_have_data_key():
    for d in main(*_make_inputs(), RADAR_ALTITUDE, LEVELS):
        assert "data" in d

def test_shi_mesh_posh_shapes_match_sweep0():
    _, shi_dict, mesh_dict, posh_dict = main(*_make_inputs(), RADAR_ALTITUDE, LEVELS)
    expected = (N_RAYS, N_BINS)
    assert shi_dict["data"].shape == expected
    assert mesh_dict["data"].shape == expected
    assert posh_dict["data"].shape == expected

def test_ke_dict_has_one_entry_per_sweep():
    n = 8
    ke_dict, *_ = main(*_make_inputs(n_sweeps=n), RADAR_ALTITUDE, LEVELS)
    assert len(ke_dict["data"]) == n

# ── physical constraints ─────────────────────────────────────────────────────

def test_mesh_nonnegative():
    _, _, mesh_dict, _ = main(*_make_inputs(), RADAR_ALTITUDE, LEVELS)
    valid = mesh_dict["data"][~np.isnan(mesh_dict["data"])]
    assert np.all(valid >= 0)

def test_posh_bounded_0_to_100():
    _, _, _, posh_dict = main(*_make_inputs(), RADAR_ALTITUDE, LEVELS)
    valid = posh_dict["data"][~np.isnan(posh_dict["data"])]
    assert np.all(valid >= 0)
    assert np.all(valid <= 100)

def test_zero_reflectivity_gives_zero_mesh():
    _, _, mesh_dict, _ = main(*_make_inputs(zh_val=0.0), RADAR_ALTITUDE, LEVELS)
    valid = mesh_dict["data"][~np.isnan(mesh_dict["data"])]
    np.testing.assert_allclose(valid, 0.0)

# ── all mesh methods run ──────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["witt1998", "mh2019_75", "mh2019_95", "blend"])
def test_all_mesh_methods_produce_correct_shape(method):
    _, _, mesh_dict, _ = main(*_make_inputs(), RADAR_ALTITUDE, LEVELS, mesh_method=method)
    assert mesh_dict["data"].shape == (N_RAYS, N_BINS)

# ── error handling ────────────────────────────────────────────────────────────

def test_invalid_radar_band_raises():
    with pytest.raises(ValueError):
        main(*_make_inputs(), RADAR_ALTITUDE, LEVELS, radar_band="X")

def test_none_levels_raises():
    with pytest.raises(ValueError):
        main(*_make_inputs(), RADAR_ALTITUDE, None)

def test_too_few_sweeps_raises():
    # ≤ 4 sweeps triggers RuntimeError by default
    refl, elev, az, rng = _make_inputs(n_sweeps=4, elevations=[0.5, 1.0, 2.0, 4.0])
    with pytest.raises(RuntimeError):
        main(refl, elev, az, rng, RADAR_ALTITUDE, LEVELS)

def test_unknown_mesh_method_raises():
    with pytest.raises(ValueError):
        main(*_make_inputs(), RADAR_ALTITUDE, LEVELS, mesh_method="invalid")

def test_warning_issued_for_few_sweeps():
    # 5 sweeps is above the exception threshold (4) but below the warning threshold (8)
    refl, elev, az, rng = _make_inputs(n_sweeps=5, elevations=[0.5, 1.0, 2.0, 4.0, 6.0])
    with pytest.warns(UserWarning):
        main(refl, elev, az, rng, RADAR_ALTITUDE, LEVELS)
