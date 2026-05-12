import numpy as np
import pytest
from pyhail.mesh_grid import main

DBZ_FNAME = "reflectivity"
LEVELS = [3000, 6000]  # m ASL: freezing level and -20°C level


class _FakeGrid:
    """Minimal Py-ART Grid stub for mesh_grid.main tests."""

    def __init__(self, nz=5, ny=5, nx=5, dz=1500, dy=5000, dx=5000,
                 radar_alt=100.0, zh_val=55.0):
        self.nz = nz
        self.ny = ny
        self.nx = nx

        z_vals = radar_alt + np.arange(nz) * dz + 500.0
        x_vals = (np.arange(nx) - nx // 2) * dx
        y_vals = (np.arange(ny) - ny // 2) * dy

        self.z = {"data": z_vals}
        self.x = {"data": x_vals}
        self.y = {"data": y_vals}
        self.radar_altitude = {"data": np.array([radar_alt])}

        data = np.ma.MaskedArray(
            np.full((nz, ny, nx), zh_val, dtype=float),
            mask=False,
        )
        self.fields = {DBZ_FNAME: {"data": data}}

    def get_point_longitude_latitude(self, lvl):
        return np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))

    def add_field(self, name, field_dict, replace_existing=False):
        self.fields[name] = field_dict


def _run(grid=None, **kwargs):
    if grid is None:
        grid = _FakeGrid()
    return main(grid, DBZ_FNAME, LEVELS, **kwargs)


# ── output structure ──────────────────────────────────────────────────────────

def test_returns_dict_with_four_keys():
    result = _run()
    assert len(result) == 4
    assert all("data" in v for v in result.values())


def test_output_shapes_are_3d():
    grid = _FakeGrid(nz=5, ny=5, nx=5)
    result = _run(grid)
    for fname, field in result.items():
        assert field["data"].ndim == 3, f"{fname} is not 3-D"


# ── near-radar mask ───────────────────────────────────────────────────────────

def test_near_radar_mask_invalidates_centre():
    """Grid centre (range=0) must be NaN under the default 10 km mask."""
    grid = _FakeGrid(nx=5, ny=5, dx=5000, dy=5000)
    cx, cy = grid.nx // 2, grid.ny // 2
    result = _run(grid)
    mesh = result["mesh_blend"]["data"]
    assert np.isnan(mesh[0, cy, cx])


def test_near_radar_mask_none_leaves_centre_valid():
    """near_radar_mask_range=None must not mask any data."""
    grid = _FakeGrid(nx=5, ny=5, dx=5000, dy=5000)
    cx, cy = grid.nx // 2, grid.ny // 2
    result = _run(grid, near_radar_mask_range=None)
    mesh = result["mesh_blend"]["data"]
    assert not np.isnan(mesh[0, cy, cx])


def test_near_radar_mask_applies_to_all_output_fields():
    """All four output fields must be NaN at the radar centre."""
    grid = _FakeGrid(nx=5, ny=5, dx=5000, dy=5000)
    cx, cy = grid.nx // 2, grid.ny // 2
    result = _run(grid)
    for fname, field in result.items():
        assert np.isnan(field["data"][0, cy, cx]), (
            f"{fname} is not NaN at radar centre"
        )


def test_near_radar_mask_preserves_distant_points():
    """Points beyond the mask radius must remain valid."""
    grid = _FakeGrid(nx=5, ny=5, dx=5000, dy=5000)
    # Corners sit at (±10 km, ±10 km): range ≈ 14.1 km > 10 km default mask.
    result = _run(grid)
    mesh = result["mesh_blend"]["data"]
    assert not np.all(np.isnan(mesh[0]))


def test_near_radar_mask_large_range_invalidates_all():
    """A mask radius larger than the grid diagonal must NaN every cell."""
    grid = _FakeGrid(nx=5, ny=5, dx=5000, dy=5000)
    # Grid spans ±10 km; diagonal ≈ 14.1 km; 20 km exceeds it.
    result = _run(grid, near_radar_mask_range=20000)
    mesh = result["mesh_blend"]["data"]
    assert np.all(np.isnan(mesh[0]))


def test_near_radar_mask_ke_all_levels_masked():
    """KE (3-D field) must be NaN at radar centre across all altitude levels."""
    grid = _FakeGrid(nz=5, nx=5, ny=5, dx=5000, dy=5000)
    cx, cy = grid.nx // 2, grid.ny // 2
    result = _run(grid)
    ke = result["hail_ke"]["data"]
    assert np.all(np.isnan(ke[:, cy, cx]))
