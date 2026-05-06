import numpy as np
import pytest
from pyhail.ppi_column_maximum import main

N_RAYS = 10
N_BINS = 30
RANGE_BIN_SIZE = 2000   # m; bins span 2–60 km

_ELEVATIONS = [0.5, 1.0, 2.0, 4.0]


def _make_inputs(n_sweeps=4, elevations=None, field_values=None):
    """
    Build minimal synthetic multi-sweep inputs.

    field_values: sequence of per-sweep fill values (one per sweep).
    If omitted, each sweep i gets value (i+1)*10.
    """
    if elevations is None:
        elevations = _ELEVATIONS[:n_sweeps]
    ranges = [np.arange(1, N_BINS + 1) * RANGE_BIN_SIZE for _ in range(n_sweeps)]
    azimuths = [np.linspace(0, 350, N_RAYS) for _ in range(n_sweeps)]
    if field_values is None:
        field_values = [(i + 1) * 10.0 for i in range(n_sweeps)]
    fields = [np.full((N_RAYS, N_BINS), float(v)) for v in field_values]
    return fields, elevations, azimuths, ranges


# Helpers to disable range and altitude filtering so a single behaviour
# can be isolated per test.
_NO_RANGE = dict(min_range=0, max_range=150)
_NO_ALT   = dict(column_altitude_minimum=0, column_altitude_maximum=100_000)


# ── output structure ──────────────────────────────────────────────────────────

def test_output_shape_matches_sweep0():
    result = main(*_make_inputs(), **_NO_RANGE, **_NO_ALT)
    assert result.shape == (N_RAYS, N_BINS)

def test_output_is_ndarray():
    result = main(*_make_inputs(), **_NO_RANGE, **_NO_ALT)
    assert isinstance(result, np.ndarray)


# ── aggregation correctness ───────────────────────────────────────────────────

def test_returns_maximum_not_mean_or_last():
    # Sweep values 10, 50, 30 — max is 50, not the mean (30) or the last (30).
    fields, elevs, az, rng = _make_inputs(n_sweeps=3,
                                          elevations=[0.5, 1.0, 2.0],
                                          field_values=[10.0, 50.0, 30.0])
    result = main(fields, elevs, az, rng, **_NO_RANGE, **_NO_ALT)
    valid = result[~np.isnan(result)]
    np.testing.assert_allclose(valid, 50.0)

def test_elevation_sort_order_does_not_matter():
    # Sweeps supplied in reverse elevation order should give the same output.
    fields, elevs, az, rng = _make_inputs()
    kwargs = {**_NO_RANGE, **_NO_ALT}
    result_fwd = main(fields,        elevs,        az,        rng,        **kwargs)
    result_rev = main(fields[::-1],  elevs[::-1],  az[::-1],  rng[::-1],  **kwargs)
    np.testing.assert_array_equal(result_fwd, result_rev)


# ── range filtering ───────────────────────────────────────────────────────────

def test_nan_inside_min_range():
    # min_range=30 km; bins whose arc distance s < 30 000 m must be NaN.
    # At 0.5° elevation, s ≈ r·cos(0.5°) ≈ r, so bins with r ≤ 28 000 m
    # (indices 0–13) are well inside the exclusion zone.
    result = main(*_make_inputs(), min_range=30, max_range=150, **_NO_ALT)
    assert np.all(np.isnan(result[:, :14]))

def test_nan_outside_max_range():
    # max_range=30 km; bins with r ≥ 34 000 m (indices 16–29) are outside.
    result = main(*_make_inputs(), min_range=0, max_range=30, **_NO_ALT)
    assert np.all(np.isnan(result[:, 16:]))


# ── altitude filtering ────────────────────────────────────────────────────────

def test_altitude_upper_bound_excludes_high_beams():
    # At bin 9 (r = 20 000 m), beam altitudes above radar level are approximately:
    #   0.5° → 198 m   (below 300 m ceiling → included)
    #   1.0° → 373 m   (above 300 m ceiling → excluded)
    #   2.0° → 720 m   (excluded)
    #   4.0° → 1419 m  (excluded)
    # With field values [10, 50, 30, 40], only sweep0 (10.0) contributes.
    fields, elevs, az, rng = _make_inputs(field_values=[10.0, 50.0, 30.0, 40.0])
    result = main(fields, elevs, az, rng,
                  **_NO_RANGE,
                  column_altitude_minimum=0,
                  column_altitude_maximum=300)
    # Bin 9 is within the valid range so it must not be NaN.
    assert not np.any(np.isnan(result[:, 9]))
    np.testing.assert_allclose(result[:, 9], 10.0)

def test_altitude_lower_bound_excludes_low_beams():
    # At bin 9 (r = 20 000 m), beam altitudes above radar level are approximately:
    #   0.5° → 198 m   (below 250 m floor → excluded)
    #   1.0° → 373 m   (above floor → included)
    #   2.0° → 720 m   (included)
    #   4.0° → 1419 m  (included)
    # With field values [10, 50, 30, 40], sweep0 is excluded and sweep1 (50.0) wins.
    fields, elevs, az, rng = _make_inputs(field_values=[10.0, 50.0, 30.0, 40.0])
    result = main(fields, elevs, az, rng,
                  **_NO_RANGE,
                  column_altitude_minimum=250,
                  column_altitude_maximum=100_000)
    assert not np.any(np.isnan(result[:, 9]))
    # Sweep0 excluded; sweep1 value (50.0) is the highest remaining.
    np.testing.assert_allclose(result[:, 9], 50.0)
