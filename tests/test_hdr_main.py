import numpy as np
import pytest
from pyhail.hdr import main

SHAPE = (10, 20)

def test_returns_two_dicts():
    hdr_meta, hdr_size_meta = main(np.full(SHAPE, 55.0), np.full(SHAPE, 0.0))
    assert isinstance(hdr_meta, dict)
    assert isinstance(hdr_size_meta, dict)

def test_output_shapes_match_input():
    hdr_meta, hdr_size_meta = main(np.full(SHAPE, 55.0), np.full(SHAPE, 0.0))
    assert hdr_meta["data"].shape == SHAPE
    assert hdr_size_meta["data"].shape == SHAPE

def test_required_metadata_keys():
    hdr_meta, hdr_size_meta = main(np.full(SHAPE, 55.0), np.full(SHAPE, 0.0))
    for key in ("data", "units", "long_name"):
        assert key in hdr_meta
        assert key in hdr_size_meta

def test_hdr_formula_zdr_at_or_below_zero():
    # ZDR ≤ 0 → zdr_fun = 27; HDR = ZH - 27
    hdr_meta, _ = main(np.full(SHAPE, 60.0), np.full(SHAPE, -1.0))
    np.testing.assert_allclose(hdr_meta["data"], 33.0)

def test_hdr_formula_zdr_above_1p74():
    # ZDR > 1.74 → zdr_fun = 60; HDR = ZH - 60
    hdr_meta, _ = main(np.full(SHAPE, 60.0), np.full(SHAPE, 2.0))
    np.testing.assert_allclose(hdr_meta["data"], 0.0)

def test_hdr_size_zero_when_hdr_nonpositive():
    # HDR = 60 - 60 = 0 → hdr_size = 0
    _, hdr_size_meta = main(np.full(SHAPE, 60.0), np.full(SHAPE, 2.0))
    assert np.all(hdr_size_meta["data"] == 0)

def test_hdr_size_positive_for_hail_signal():
    # ZDR = 1.0 (between 0 and 1.74): zdr_fun = 19*1 + 27 = 46; HDR = 60 - 46 = 14 > 0
    _, hdr_size_meta = main(np.full(SHAPE, 60.0), np.full(SHAPE, 1.0))
    assert np.all(hdr_size_meta["data"] > 0)

def test_nan_propagates():
    zh = np.full(SHAPE, 55.0)
    zh[0, :] = np.nan
    hdr_meta, _ = main(zh, np.full(SHAPE, 0.0))
    assert np.all(np.isnan(hdr_meta["data"][0, :]))
