import numpy as np
import pytest
from pyhail.hacc import main

SHAPE = (10, 20)
FZ_LEVEL = 3000   # m ASL
PRESSURE = 850    # hPa

def test_output_shape_matches_input():
    result = main(
        np.full(SHAPE, 60.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    assert result["data"].shape == SHAPE

def test_required_metadata_keys():
    result = main(
        np.full(SHAPE, 60.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    for key in ("data", "units", "long_name", "description"):
        assert key in result

def test_hacc_nonnegative_below_freezing_level():
    # zh=60 >= default threshold 55 → hail mask active; z=1000 < fz=3000 → hacc > 0
    result = main(
        np.full(SHAPE, 60.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    assert np.all(result["data"] >= 0)

def test_hacc_zero_above_freezing_level():
    # z=5000 > fz=3000 → IWC zeroed → hacc = 0
    result = main(
        np.full(SHAPE, 60.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 5000.0), FZ_LEVEL, PRESSURE,
    )
    np.testing.assert_allclose(result["data"], 0.0)

def test_hacc_zero_when_reflectivity_below_threshold():
    # zh=40 < default sp threshold 55 → hail mask inactive → hacc = 0
    result = main(
        np.full(SHAPE, 40.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    np.testing.assert_allclose(result["data"], 0.0)

def test_hacc_zero_when_hsda_sweep_has_no_hail():
    # hsda_sweep all zeros → no hail pixels → hacc = 0
    result = main(
        np.full(SHAPE, 60.0), np.zeros(SHAPE), np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    np.testing.assert_allclose(result["data"], 0.0)

def test_hsda_sweep_none_falls_back_to_reflectivity_threshold():
    # With hsda=None and zh=60 >= 55, hacc is positive; confirms the fallback path runs.
    result = main(
        np.full(SHAPE, 60.0), None, np.full(SHAPE, 25.0),
        np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE,
    )
    assert np.any(result["data"] > 0)

def test_nan_reflectivity_propagates():
    zh = np.full(SHAPE, 60.0)
    zh[0, :] = np.nan
    result = main(zh, None, np.full(SHAPE, 25.0), np.full(SHAPE, 1000.0), FZ_LEVEL, PRESSURE)
    assert np.all(np.isnan(result["data"][0, :]))
