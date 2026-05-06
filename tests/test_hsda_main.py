import numpy as np
import pytest
from pyhail.hsda import main

SHAPE = (10, 20)
LEVELS = [3000, 6000]   # wbt_0c, wbt_minus25c (m ASL)
HCA_HAIL_IDX = [6]

def _zh():
    return np.full(SHAPE, 55.0)

def _zdr():
    return np.full(SHAPE, 0.0)

def _rhv():
    return np.full(SHAPE, 0.97)

def _z():
    # 2000 m ASL → alt_index 2 (wbt_0c-1000 <= 2000 < wbt_0c)
    return np.full(SHAPE, 2000.0)

def _cls_no_hail():
    return np.zeros(SHAPE)

def _cls_one_hail_pixel():
    cls = np.zeros(SHAPE)
    cls[5, 10] = 6
    return cls

def test_output_shape_matches_input():
    result = main(_zh(), _zdr(), _rhv(), _cls_no_hail(), _z(), LEVELS, HCA_HAIL_IDX)
    assert result["data"].shape == SHAPE

def test_required_metadata_keys():
    result = main(_zh(), _zdr(), _rhv(), _cls_no_hail(), _z(), LEVELS, HCA_HAIL_IDX)
    for key in ("data", "units", "long_name", "comments"):
        assert key in result

def test_no_hail_pixels_returns_all_nan():
    result = main(_zh(), _zdr(), _rhv(), _cls_no_hail(), _z(), LEVELS, HCA_HAIL_IDX)
    assert np.all(np.isnan(result["data"]))

def test_hail_pixel_is_classified():
    result = main(_zh(), _zdr(), _rhv(), _cls_one_hail_pixel(), _z(), LEVELS, HCA_HAIL_IDX)
    assert result["data"][5, 10] in (1, 2, 3)

def test_non_hail_pixels_remain_nan():
    result = main(_zh(), _zdr(), _rhv(), _cls_one_hail_pixel(), _z(), LEVELS, HCA_HAIL_IDX)
    mask = np.ones(SHAPE, dtype=bool)
    mask[5, 10] = False
    assert np.all(np.isnan(result["data"][mask]))

def test_nan_zh_at_hail_pixel_stays_nan():
    zh = _zh()
    zh[5, 10] = np.nan
    result = main(zh, _zdr(), _rhv(), _cls_one_hail_pixel(), _z(), LEVELS, HCA_HAIL_IDX)
    assert np.isnan(result["data"][5, 10])

def test_levels_order_does_not_matter():
    # Passing levels in either order should give the same result.
    r1 = main(_zh(), _zdr(), _rhv(), _cls_one_hail_pixel(), _z(), [3000, 6000], HCA_HAIL_IDX)
    r2 = main(_zh(), _zdr(), _rhv(), _cls_one_hail_pixel(), _z(), [6000, 3000], HCA_HAIL_IDX)
    np.testing.assert_array_equal(r1["data"], r2["data"])
